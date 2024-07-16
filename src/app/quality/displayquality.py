
import streamlit as st
from src.quality.evals import MMLU, CustomEval, PubMedQA, TruthfulQA
import pandas as pd
import asyncio
import copy
import plotly.express as px
from my_utils.ml_logging import get_logger
from src.app.quality.results import BenchmarkQualityResult

# Set up logger
logger = get_logger()

top_bar = st.empty()
results_c = st.container()
batch_c = st.container()

def create_tabs_for_non_empty_dataframes(results, results_container: st.container):
    """
    Creates Streamlit tabs for each non-empty DataFrame in the results dictionary.

    Parameters:
    - results (dict): A dictionary where keys are descriptive names of the dataframes
                      and values are the DataFrame objects.

    Each non-empty DataFrame is displayed in its own tab, labeled with the key from the results dictionary.
    If the key is "all_results", the 'test' column from the DataFrame is added to the index,
    so each row reflects one test in the index.
    If the key is "retrieval_results" or "rai_results", the DataFrame is displayed as it is.
    """
    non_empty_dataframes = {key: value for key, value in results.items() if not value.empty}

    tabs = results_container.tabs(list(non_empty_dataframes.keys()))
    for tab, (key, df) in zip(tabs, non_empty_dataframes.items()):
        with tab:
            if key == "all_results":
                df_with_test_index = df.set_index('test', append=True)
                tab.dataframe(df_with_test_index, hide_index=False)
            elif key in ["retrieval_results", "rai_results"]:
                if 'studio_url' in df.columns:
                    display_df = df.drop(columns=['studio_url'], errors='ignore')
                    studio_urls = df['studio_url']
                    tab.dataframe(display_df, hide_index=False)
                    
                    # Generate markdown links for non-empty studio_urls
                    for model, studio_url in zip(df.index, studio_urls):
                        if studio_url:  
                            tab.markdown(f"- Look at the result for the model {model} in [Azure AI studio]({studio_url})")
                else:
                    tab.dataframe(df, hide_index=False)
            else:
                tab.dataframe(df.drop("test", axis=1, errors='ignore'), hide_index=True)

def display_code_setting_sdk() -> None:
    """
    Displays the SDK setup code and provides a link to the SDK documentation.

    This function generates a code snippet that users can use to reproduce the SDK setup.
    It then displays this code snippet in a Streamlit code block and provides a markdown
    link to the SDK documentation for further details.
    """
    code = f"""
    from src.performance.latencytest import....#TODO"""
    st.code(code, language="python")
    st.markdown(
        "More details on the SDK can be found [here](https://github.com/pablosalvador10/gbb-ai-upgrade-llm)."
    )

def display_human_readable_settings(id: str, timestamp: str, settings: dict) -> None:
    """
    Displays a human-readable summary of benchmark settings.

    This function takes an identifier, a timestamp, and a settings dictionary as input.
    It generates and displays a markdown-formatted summary of the benchmark details,
    including the benchmark type, selected tests, and specific settings for each test
    if applicable (e.g., subsample percentages, categories, and custom evaluation settings).

    Parameters:
    - id (str): The unique identifier for the benchmark run.
    - timestamp (str): The timestamp when the benchmark was run.
    - settings (dict): A dictionary containing the settings for the benchmark run,
                       including selected tests and their specific configurations.

    The function does not return any value but displays the summary directly in the UI
    using Streamlit's markdown functionality.
    """
    # Benchmark Details
    benchmark_details = f"""
    **Benchmark Details:**
    - **ID:** `{id}`
    - **Timestamp:** `{timestamp}`
    """
    benchmark_selection = settings.get("benchmark_selection", [])

    summary_lines = [
        "#### Benchmark Configuration Summary",
        f"- **Benchmark Type:** Quality Benchmark",
        f"- **Tests:** {', '.join(benchmark_selection)}",
    ]

    if "MMLU" in benchmark_selection:
        mmlu_categories = settings.get("mmlu_categories", [])
        mmlu_subsample = settings.get("mmlu_subsample", 0)
        summary_lines.append(f"  - **MMLU Categories:** {', '.join(mmlu_categories)}")
        summary_lines.append(f"  - **MMLU Subsample Percentage:** {mmlu_subsample}%")

    if "MedPub QA" in benchmark_selection:
        medpub_subsample = settings.get("medpub_subsample", 0)
        summary_lines.append(f"  - **MedPub QA Subsample Percentage:** {medpub_subsample}%")

    if "Truthful QA" in benchmark_selection:
        truthful_subsample = settings.get("truthful_subsample", 0)
        summary_lines.append(f"  - **Truthful QA Subsample Percentage:** {truthful_subsample}%")

    if "Custom Evaluation" in benchmark_selection:
        custom_settings = settings.get("custom_benchmark", {})
        prompt_col = custom_settings.get("prompt_col", "None")
        ground_truth_col = custom_settings.get("ground_truth_col", "None")
        context_col = custom_settings.get("context_col", "None")
        summary_lines.append(f"  - **Custom Evaluation Prompt Column:** {prompt_col}")
        summary_lines.append(f"  - **Custom Evaluation Ground Truth Column:** {ground_truth_col}")
        summary_lines.append(f"  - **Custom Evaluation Context Column:** {context_col}")

    if "Retrieval" in benchmark_selection:
        retrieval_setting = st.session_state["settings_quality"][f"evaluation_clients_retrieval_df_BYOP"]
        summary_lines.append(f"  - Retrieval BYOP Selected: {retrieval_setting}")

    if "RAI" in benchmark_selection:
        rai_setting = st.session_state["settings_quality"][f"evaluation_clients_rai_df_BYOP"]
        summary_lines.append(f"  - RAI BYOP Selected: {rai_setting}")

    summary_lines.append("")
    summary_lines.append(f"- **Azure AI Studio Connection Details**")
    markdown_summary = """
    - **Azure AI Studio Subscription ID**: {}
    - **Azure AI Studio Resource Group Name**: {}
    - **Azure AI Studio Project Name**: {}
    """.format(
        st.session_state.get('azure_ai_studio_subscription_id', 'Not set'),
        st.session_state.get('azure_ai_studio_resource_group_name', 'Not set'),
        st.session_state.get('azure_ai_studio_project_name', 'Not set')
    )
    
    # Append the Markdown-formatted string to summary_lines
    summary_lines.append(markdown_summary)
    
    # Convert summary_lines to a markdown string
    summary_markdown = "\n".join(summary_lines)

    # Display using st.markdown
    st.markdown(benchmark_details, unsafe_allow_html=True)
    st.markdown(summary_markdown, unsafe_allow_html=True)

def ask_user_for_result_display_preference(id: str, settings: dict, results_container:st.container) -> None:
    """
    Asks the user for their preference on how to display the results of a specific run.

    This function retrieves the results of a specific run identified by `id` from the
    Streamlit session state. It then displays two expanders in a two-column layout:
    one for reproducing the run using SDK code, and another for viewing the run's
    configuration details in a human-readable format.

    Parameters:
    - id (str): The unique identifier for the run whose results are to be displayed.

    The function does not return any value but directly manipulates the Streamlit UI
    to display the requested information.
    """
    results_id = st.session_state["results_quality"][id]
    timestamp = results_id["timestamp"]
    with results_container: 
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("👨‍💻 Reproduce Run Using SDK", expanded=False):
                display_code_setting_sdk()

        with col2:
            with st.expander("👥 Run Configuration Details", expanded=False):
                display_human_readable_settings(id, timestamp, settings)
   
def display_quality_results(results_container: st.container, id: str):
    """
    Display enhanced quality benchmark results with interactive visualizations.

    :param results_container: Streamlit container to display the results.
    :param id: The unique identifier for the run.
    """

    try:
        if not isinstance(st.session_state["results_quality"], dict):
            raise TypeError("Expected 'results_quality' to be a dictionary.")
        
        results_id = st.session_state["results_quality"].get(id)
        
        # Check if results_id is correctly retrieved and is a dictionary
        if not isinstance(results_id, dict):
            raise TypeError(f"Expected 'results_id' to be a dictionary, got {type(results_id)} instead.")
        
        if results_id is None or "result" not in results_id or results_id["result"] is None:
            raise ValueError("Results are not available for the given ID.")
        
        results = results_id["result"]
        settings = results_id["settings"]
    except ValueError as e:
        st.warning(
            f"⚠️ Oops! We couldn't retrieve the data for ID: {id}. Error: {e}. Sorry for the inconvenience! 😓 Please try another ID. 🔄"
        )
        return


    results_container.markdown("## 📈 Quality Benchmark Results")
    st.toast(f"You are viewing results from Run ID: {id}", icon="ℹ️")
    results_container.markdown("""
        🧭 **Navigating the Results**

        - **Data Analysis Section**: Start here for a comprehensive analysis of the data.
        - **Visual Insights Section**: Use this section to draw conclusions by run with complex interactions.
        - **Benchmark Buddy**: Utilize this tool for an interactive, engaging "GPT" like analysis experience.

      💡 **Tip**: For a deeper understanding of Language Model (LM) evaluations, including both Large Language Models (LLMs) and Smaller Language Models (SLMs), as 
                               well as system-wide assessments, explore the article [Read the detailed exploration here.](#)
    """)

    ask_user_for_result_display_preference(id, settings, results_container)

    results_container.write("### 🔍 Data Analysis")

    create_tabs_for_non_empty_dataframes(results, results_container)

    with results_container:
        st.write("### 🤔 Visual Insights")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Understanding Results Overview", 
                                          "🔍 Detailed Scatter Analysis", 
                                          "⚙️ Retrieval System/QA", 
                                          "🛡️ RAI"])

        with tab1:
            results_all = results["all_results"]
            if not results_all.empty:
                fig = px.bar(
                    results_all,
                    x="overall_score",
                    y="test",
                    color="deployment",
                    barmode="group",
                    orientation="h",
                    title="Benchmark Results Overview",
                    hover_data=["overall_score", "deployment"]
                )
                fig.update_layout(
                    xaxis_title="Overall Score",
                    yaxis_title="Test",
                    legend_title="Deployment",
                    coloraxis_showscale=True,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("👈 Hey - you haven't fired any benchmarks yet. Please configure the benchmark settings and click 'Start Benchmark' to begin.")
        
        with tab2:
            if not results_all.empty:
                fig2 = px.line(
                    results_all,
                    x="test", 
                    y="overall_score",
                    color="deployment", 
                    markers=True,
                    title="Score Distribution by Test Across Deployments",
                    labels={'test': 'Test', 'overall_score': 'Score'}
                )
                fig2.update_layout(
                    xaxis_title="Test",
                    yaxis_title="Score",
                    legend_title="Deployment"
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("👈 No data available for scatter plot analysis. Please ensure benchmarks have been run.")

        with tab3:
            results_retrievals = results["retrieval_results"].drop(columns=['studio_url'], errors='ignore')
            if not results_retrievals.empty:
                df_reset = results_retrievals.reset_index()
                df_reset.rename(columns={df_reset.columns[0]: 'model'}, inplace=True)
                
                df_long = df_reset.melt(id_vars=['model'], var_name='Metric', value_name='Value')
                
                fig2 = px.line(
                    df_long,
                    x="Metric", 
                    y="Value",
                    color="model", 
                    markers=True,
                    title="Metric Values Across Models",
                    labels={'model': 'Model', 'Value': 'Metric Value', 'Metric': 'Metric'}
                )
                fig2.update_layout(
                    xaxis_title="Metric",
                    yaxis_title="Metric Value",
                    legend_title="Model"
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No data available for plot analysis. Please ensure runs settings is loaded correctly.")

        with tab4:
            results_rai = results["rai_results"].drop(columns=['studio_url'], errors='ignore')
            if not results_rai.empty:
                df_reset = results_rai.reset_index()
                df_reset.rename(columns={df_reset.columns[0]: 'model'}, inplace=True)
                
                df_long = df_reset.melt(id_vars=['model'], var_name='Metric', value_name='Value')
                
                fig2 = px.line(
                    df_long,
                    x="Metric", 
                    y="Value",
                    color="model", 
                    markers=True,
                    title="Metric Values Across Models",
                    labels={'model': 'Model', 'Value': 'Metric Value', 'Metric': 'Metric'}
                )
                fig2.update_layout(
                    xaxis_title="Metric",
                    yaxis_title="Metric Value",
                    legend_title="Model"
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No data available for plot analysis. Please ensure runs settings is loaded correctly.")
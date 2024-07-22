import streamlit as st
import plotly.express as px
from my_utils.ml_logging import get_logger
import traceback
import pandas as pd

# Set up logger
logger = get_logger()

def create_tabs_for_non_empty_dataframes(results, results_container: st.container):
    """
    Creates Streamlit tabs for each non-empty DataFrame in the results dictionary.
    Logs the creation of tabs and any issues encountered.
    """
    try:
        non_empty_dataframes = {key: value for key, value in results.items() if not value.empty}
        
        if not non_empty_dataframes:
            st.warning("No data available to display in tabs.")
            logger.info("No non-empty dataframes to display.")
            return

        tabs = results_container.tabs(list(non_empty_dataframes.keys()))
        for tab, (key, df) in zip(tabs, non_empty_dataframes.items()):
            with tab:
                if key == "understanding_results":
                    if 'test' in df.columns:
                        df_with_test_index = df.set_index('test', append=True)
                        tab.dataframe(df_with_test_index, hide_index=False)
                        logger.info(f"Displayed understanding_results with 'test' column for key: {key}")
                    else:
                        logger.warning("'test' column not found in understanding_results DataFrame.")
                        tab.dataframe(df, hide_index=False)
                elif key in ["retrieval_results", "rai_results"]:
                    if 'studio_url' in df.columns:
                        display_df = df.drop(columns=['studio_url'], errors='ignore')
                        studio_urls = df['studio_url']
                        tab.dataframe(display_df, hide_index=False)
                        
                        for model, studio_url in zip(df.index, studio_urls):
                            if studio_url:  
                                tab.markdown(f"- Look at the result for the model {model} in [Azure AI studio]({studio_url})")
                                logger.info(f"Displayed studio_url link for model: {model} in key: {key}")
                    else:
                        tab.dataframe(df, hide_index=False)
                else:
                    tab.dataframe(df.drop("test", axis=1, errors='ignore'), hide_index=True)
                logger.info(f"Displayed dataframe for key: {key}")
    except Exception as e:
        logger.error(f"Error in creating tabs for non-empty dataframes: {e}")
        logger.error(traceback.format_exc())
        st.error(f"An error occurred while creating tabs: {e}")

def display_code_setting_sdk() -> None:
    """
    Displays the SDK setup code and provides a link to the SDK documentation.
    Logs the display action.
    """
    try:
        code = """
        from src.performance.latencytest import....#TODO"""
        st.code(code, language="python")
        st.markdown(
            "More details on the SDK can be found [here](https://github.com/pablosalvador10/gbb-ai-upgrade-llm)."
        )
        logger.info("Displayed SDK setup code and link to documentation.")
    except Exception as e:
        logger.error(f"Error in displaying SDK setup code: {e}")
        logger.error(traceback.format_exc())
        st.error(f"An error occurred while displaying the SDK setup code: {e}")

def display_human_readable_settings(id: str, timestamp: str, settings: dict) -> None:
    """
    Displays a human-readable summary of benchmark settings.
    Logs the display action.
    """
    try:
        benchmark_details = f"""
        **Benchmark Details:**
        - **ID:** `{id}`
        - **Timestamp:** `{timestamp}`
        """
        benchmark_selection = settings.get("benchmark_selection", [])
        benchmark_selection_multiselect = settings.get("benchmark_selection_multiselect", [])
        benchmark = benchmark_selection + benchmark_selection_multiselect

        summary_lines = [
            "#### Benchmark Configuration Summary",
            f"- **Benchmark Type:** Quality Benchmark",
            f"- **Executed Tests:** {', '.join(benchmark)}",
        ]

        if "MMLU" in benchmark:
            mmlu_categories = settings.get("mmlu_categories", [])
            mmlu_subsample = settings.get("mmlu_subsample", 0)
            summary_lines.append(f"  - **MMLU Categories:** {', '.join(mmlu_categories)}")
            summary_lines.append(f"  - **MMLU Subsample Percentage:** {mmlu_subsample}%")

        if "MedPub QA" in benchmark:
            medpub_subsample = settings.get("medpub_subsample", 0)
            summary_lines.append(f"  - **MedPub QA Subsample Percentage:** {medpub_subsample}%")

        if "Truthful QA" in benchmark:
            truthful_subsample = settings.get("truthful_subsample", 0)
            summary_lines.append(f"  - **Truthful QA Subsample Percentage:** {truthful_subsample}%")

        if "Custom Evaluation" in benchmark:
            custom_settings = settings.get("custom_benchmark", {})
            prompt_col = custom_settings.get("prompt_col", "None")
            ground_truth_col = custom_settings.get("ground_truth_col", "None")
            context_col = custom_settings.get("context_col", "None")
            summary_lines.append(f"  - **Custom Evaluation Prompt Column:** {prompt_col}")
            summary_lines.append(f"  - **Custom Evaluation Ground Truth Column:** {ground_truth_col}")
            summary_lines.append(f"  - **Custom Evaluation Context Column:** {context_col}")

        if "Retrieval" in benchmark:
            retrieval_setting = st.session_state["settings_quality"]["evaluation_clients_retrieval_df_BYOP"]
            summary_lines.append(f"  - Retrieval BYOP Selected: {retrieval_setting}")

        if "RAI" in benchmark:
            rai_setting = st.session_state["settings_quality"]["evaluation_clients_rai_df_BYOP"]
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
        
        summary_lines.append(markdown_summary)
        
        summary_markdown = "\n".join(summary_lines)

        st.markdown(benchmark_details, unsafe_allow_html=True)
        st.markdown(summary_markdown, unsafe_allow_html=True)
        logger.info(f"Displayed human-readable settings for ID: {id}")
    except Exception as e:
        logger.error(f"Error in displaying human-readable settings: {e}")
        logger.error(traceback.format_exc())
        st.error(f"An error occurred while displaying the settings: {e}")

def ask_user_for_result_display_preference(id: str, settings: dict, results_container: st.container) -> None:
    """
    Asks the user for their preference on how to display the results of a specific run.
    Logs the display action.
    """
    try:
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
        logger.info(f"Asked user for result display preference for ID: {id}")
    except Exception as e:
        logger.error(f"Error in asking user for result display preference: {e}")
        logger.error(traceback.format_exc())
        st.error(f"An error occurred while asking for display preference: {e}")

def display_quality_results(results_container: st.container, id: str):
    """
    Display enhanced quality benchmark results with interactive visualizations.
    Handles exceptions and logs all actions.
    """
    try:
        if not isinstance(st.session_state["results_quality"], dict):
            raise TypeError("Expected 'results_quality' to be a dictionary.")
        
        results_id = st.session_state["results_quality"].get(id)
        
        if not isinstance(results_id, dict):
            raise TypeError(f"Expected 'results_id' to be a dictionary, got {type(results_id)} instead.")
        
        if results_id is None or "result" not in results_id or results_id["result"] is None:
            raise ValueError("Results are not available for the given ID.")
        
        results = results_id["result"]
        settings = results_id["settings"]
    except Exception as e:
        logger.error(f"Error while retrieving results for ID {id}: {e}")
        logger.error(traceback.format_exc())
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
                               well as system-wide assessments, explore the [article](#) (available soon)
    """)

    ask_user_for_result_display_preference(id, settings, results_container)
    results_container.write("### 🔍 Data Analysis")
    create_tabs_for_non_empty_dataframes(results, results_container)

    with results_container:
        st.write("### 🤔 Visual Insights")

        tab1, tab2, tab3 = st.tabs(["🧠 Understanding Overview", 
                                    "⚙️ Retrieval System/QA Comparison", 
                                    "🛡️ Responsible AI (RAI) Insights"])
        
        with tab1:
            results_all = results.get("understanding_results", pd.DataFrame())
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
                logger.info("Displayed line chart for understanding_results.")
            else:
                st.info("👈 No data available for scatter plot analysis. Please ensure benchmarks have been run.")
                logger.info("No data available for understanding_results line chart.")

        with tab2:
            results_retrievals = results.get("retrieval_results", pd.DataFrame()).drop(columns=['studio_url'], errors='ignore')
            if not results_retrievals.empty:
                df_reset = results_retrievals.reset_index()
                df_reset.rename(columns={df_reset.columns[0]: 'deployment'}, inplace=True)
                
                df_long = df_reset.melt(id_vars=['deployment'], var_name='Metric', value_name='Value')
                
                fig2 = px.line(
                    df_long,
                    x="Metric", 
                    y="Value",
                    color="deployment", 
                    markers=True,
                    title="Metric Values Across Models",
                    labels={'deployment': 'deployment', 'Value': 'Metric Value', 'Metric': 'Metric'}
                )
                fig2.update_layout(
                    xaxis_title="Metric",
                    yaxis_title="Metric Value",
                    legend_title="Deployment"
                )
                st.plotly_chart(fig2, use_container_width=True)
                logger.info("Displayed line chart for retrieval_results.")
            else:
                st.info("No data available for plot analysis. Please ensure benchmarks have been run.")
                logger.info("No data available for retrieval_results line chart.")

        with tab3:
            results_rai = results.get("rai_results", pd.DataFrame()).drop(columns=['studio_url'], errors='ignore')
            if not results_rai.empty:
                df_reset = results_rai.reset_index()
                df_reset.rename(columns={df_reset.columns[0]: 'model'}, inplace=True)
                
                df_long = df_reset.melt(id_vars=['deployment'], var_name='Metric', value_name='Value')
                
                fig2 = px.line(
                    df_long,
                    x="Metric", 
                    y="Value",
                    color="deployment", 
                    markers=True,
                    title="Metric Values Across Deployment",
                    labels={'deployment': 'Deployment', 'Value': 'Metric Value', 'Metric': 'Metric'}
                )
                fig2.update_layout(
                    xaxis_title="Metric",
                    yaxis_title="Metric Value",
                    legend_title="deployment"
                )
                st.plotly_chart(fig2, use_container_width=True)
                logger.info("Displayed line chart for rai_results.")
            else:
                st.info("No data available for plot analysis. Please ensure benchmarks have been run.")
                logger.info("No data available for rai_results line chart.")
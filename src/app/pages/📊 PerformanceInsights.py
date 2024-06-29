import asyncio
import json
import os
from typing import Any, Dict, List
import plotly.graph_objects as go

import dotenv
import pandas as pd
import streamlit as st

from src.aoai.azure_openai import AzureOpenAIManager
from src.app.outputformatting import markdown_to_docx
from src.performance.latencytest import AzureOpenAIBenchmarkNonStreaming, AzureOpenAIBenchmarkStreaming
from src.performance.aoaihelpers.stats import ModelPerformanceVisualizer
from utils.ml_logging import get_logger
from src.app.Home import add_deployment_form, display_deployments, load_default_deployment

# Load environment variables if not already loaded
dotenv.load_dotenv(".env")

# Set up logger
logger = get_logger()

st.set_option('deprecation.showPyplotGlobalUse', False)

# Initialize session state variables if they don't exist
session_vars = [
    "conversation_history",
    "ai_response",
    "chat_history",
    "messages",
    "log_messages",
    "benchmark_results",
    "deployments"
]
initial_values = {
    "conversation_history": [],
    "ai_response": "",
    "chat_history": [],
    "messages": [
        {
            "role": "assistant",
            "content": "Hey, this is your AI assistant. Please look at the AI request submit and let's work together to make your content shine!",
        }
    ],
    "log_messages": [],
    "benchmark_results": [],
    "deployments": {}
}
for var in session_vars:
    if var not in st.session_state:
        st.session_state[var] = initial_values.get(var, None)

st.set_page_config(
    page_title="Performance Insights AI Assistant",
    page_icon="📊",
)

def create_azure_openai_manager(api_key: str, endpoint: str, api_version: str, deployment_id: str) -> AzureOpenAIManager:
    """
    Create a new Azure OpenAI Manager instance.
    """
    return AzureOpenAIManager(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        chat_model_name=deployment_id,
    )

def create_benchmark_non_streaming_client(api_key: str, endpoint: str, api_version: str) -> AzureOpenAIBenchmarkNonStreaming:
    """
    Create a new benchmark client instance.
    """
    return AzureOpenAIBenchmarkNonStreaming(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )

def create_benchmark_streaming_client(api_key: str, endpoint: str, api_version: str) -> AzureOpenAIBenchmarkNonStreaming:
    """
    Create a new benchmark client instance.
    """
    return AzureOpenAIBenchmarkStreaming(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )

# Sidebar logic
with st.sidebar:
    st.markdown("## 🎯 Benchmark Selection")
    operation = st.selectbox(
        "Choose Your Benchmark:",
        ("Latency Benchmark", "Throughput Benchmark by Model"),
        help="Select the benchmark you want to perform to evaluate AI model performance.",
        placeholder="Select a Benchmark",
    )

    if operation == "Latency Benchmark":
        with st.expander("Latency Benchmark Guide 📊", expanded=False):
            st.markdown("""
                **How to Run a Latency Benchmark:**

                1. **Select model settings**: Choose models, max tokens, and iterations for your benchmark tests.
                2. **Run the benchmark**: Click 'Run Benchmark' to start and monitor progress in real-time.
                3. **Review results**: View detailed performance metrics after the benchmark completes.

                Optimize your LLM experience with precise performance insights!
            """)

        st.markdown("---")

        st.markdown("## ⚙️ Configuration Settings")
        enable_multi_region = st.checkbox("Multi-Deployment Benchmarking", help="Check this box to compare performance across multiple deployments in different regions.")
        if enable_multi_region:
            operation = st.selectbox(
                "Choose Your MaaS Deployment:",
                ("AOAI Deployment", "Other"),
                index=0,
                help="Select the benchmark you want to perform to evaluate AI model performance.",
                placeholder="Select a Benchmark",
            )
            if operation == "AOAI Deployment":
                add_deployment_form()
            else: 
                st.info("Other deployment options will be available soon.")
        else:
            load_default_deployment()

        display_deployments()

        if operation == "Latency Benchmark":
            byop_option = st.radio(
                "BYOP (Bring Your Own Prompt)",
                options=["No", "Yes"],
                help="Select 'Yes' to bring your own prompt or 'No' to use default settings."
            )

            if byop_option == "Yes":
                uploaded_file = st.file_uploader("Upload CSV", type='csv', help="Upload a CSV file with prompts for the benchmark tests.")
                if uploaded_file is not None:
                    st.write("File uploaded successfully!")
            elif byop_option == "No":
                context_tokens = st.slider(
                    "Context Tokens (Input)",
                    min_value=100,
                    max_value=5000,
                    value=1000,
                    help="Select the number of context tokens for each run.",
                )
            # Custom output tokens checkbox
            custom_output_tokens = st.checkbox("Custom Output Tokens")
            if custom_output_tokens:
                custom_tokens_input = st.text_input("Type your own max tokens (separate multiple values with commas):", help="Enter custom max tokens for each run.")
                if custom_tokens_input:
                    try:
                        max_tokens_list = [int(token.strip()) for token in custom_tokens_input.split(',')]
                    except ValueError:
                        st.error("Please enter valid integers separated by commas for max tokens.")
            else:
                options = [100, 500, 800, 1000, 1500, 2000]
                max_tokens_list = st.multiselect(
                    "Select Max Output Tokens (Generation)",
                    options=options,
                    default=[500],
                    help="Select the maximum tokens for each run."
                )
            num_iterations = st.slider(
                "Number of Iterations",
                min_value=1,
                max_value=100,
                value=50,
                help="Select the number of iterations for each benchmark test.",
            )

            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Select the temperature setting for the benchmark tests.",
            )

            prevent_server_caching = st.checkbox(
                "Prevent Server Caching",
                value=True,
                help="Enable this option to prevent server caching during the benchmark tests.",
            )

    st.markdown("---")
    st.markdown("## 🚀 Run Benchmark")
    st.markdown("Ensure all settings are correctly configured before proceeding.")
    run_benchmark = st.button("Start Benchmark")

def display_statistics(stats: List[Dict[str, Any]]) -> None:
    """
    Display benchmark statistics in a formatted manner with enhanced user interface.
    """
    st.markdown("## Benchmark Results")
    table = []
    headers = [
        "Model_MaxTokens", "is_Streaming", "Iterations", "Regions",
        "Average TTLT", "Median TTLT", "IQR TTLT", "95th Percentile TTLT", "99th Percentile TTLT", "CV TTLT",
        "Median Prompt Tokens", "IQR Prompt Tokens", "Median Completion Tokens", "IQR Completion Tokens",
        "95th Percentile Completion Tokens", "99th Percentile Completion Tokens", "CV Completion Tokens",
        "Average TBT", "Median TBT", "IQR TBT", "95th Percentile TBT", "99th Percentile TBT",
        "Average TTFT", "Median TTFT", "IQR TTFT", "95th Percentile TTFT", "99th Percentile TTFT",
        "Error Rate", "Error Types", "Successful Runs", "Unsuccessful Runs",
        "Throttle Count", "Throttle Rate", "Best Run", "Worst Run",
    ]
    for stat in stats: 
        for key, data in stats.items():
            regions = data.get("regions", [])
            regions = [r for r in regions if r is not None]
            region_string = ", ".join(set(regions)) if regions else "N/A"
            row = [
                key,
                data.get("is_streaming", "N/A"),
                data.get("number_of_iterations", "N/A"),
                region_string,
                data.get("average_ttlt", "N/A"),
                data.get("median_ttlt", "N/A"),
                data.get("iqr_ttlt", "N/A"),
                data.get("percentile_95_ttlt", "N/A"),
                data.get("percentile_99_ttlt", "N/A"),
                data.get("cv_ttlt", "N/A"),
                data.get("median_prompt_tokens", "N/A"),
                data.get("iqr_prompt_tokens", "N/A"),
                data.get("median_completion_tokens", "N/A"),
                data.get("iqr_completion_tokens", "N/A"),
                data.get("percentile_95_completion_tokens", "N/A"),
                data.get("percentile_99_completion_tokens", "N/A"),
                data.get("cv_completion_tokens", "N/A"),
                data.get("average_tbt", "N/A"),
                data.get("median_tbt", "N/A"),
                data.get("iqr_tbt", "N/A"),
                data.get("percentile_95_tbt", "N/A"),
                data.get("percentile_99_tbt", "N/A"),
                data.get("average_ttft", "N/A"),
                data.get("median_ttft", "N/A"),
                data.get("iqr_ttft", "N/A"),
                data.get("percentile_95_ttft", "N/A"),
                data.get("percentile_99_ttft", "N/A"),
                data.get("error_rate", "N/A"),
                data.get("errors_types", "N/A"),
                data.get("successful_runs", "N/A"),
                data.get("unsuccessful_runs", "N/A"),
                data.get("throttle_count", "N/A"),
                data.get("throttle_rate", "N/A"),
                json.dumps(data.get("best_run", {})) if data.get("best_run") else "N/A",
                json.dumps(data.get("worst_run", {})) if data.get("worst_run") else "N/A",
            ]
            table.append(row)

    table.sort(key=lambda x: x[3])

    df = pd.DataFrame(table, columns=headers)
    
    def color_lowest_median_time(s: pd.Series) -> List[str]:
        is_lowest = s == s.min()
        return ['background-color: green' if v else '' for v in is_lowest]
    
    styled_df = df.style.apply(color_lowest_median_time, subset=['Median Time'])
    # Add an expander with explanations for all columns in the dataframe
    with st.expander("Column Descriptions", expanded=False):
        st.markdown("""
        - **Deployment_MaxTokens**: The maximum number of tokens allowed in a single deployment.
        - **Iterations**: The number of iterations or runs performed during the analysis.
        - **Regions**: Geographic regions where the deployments were executed.
        - **Average Time**: The average time taken for completions across all runs.
        - **Median Time**: The median time taken for completions, reducing the impact of outliers.
        - **IQR Time**: Interquartile Range for time, indicating the spread of the middle 50% of the data.
        - **95th Percentile Time**: Time below which 95% of the completion times fall.
        - **99th Percentile Time**: Time below which 99% of the completion times fall.
        - **CV Time**: Coefficient of Variation for time, indicating the relative variability in completion times.
        - **Median Prompt Tokens**: The median number of tokens in the prompts used.
        - **IQR Prompt Tokens**: Interquartile Range for the number of prompt tokens.
        - **Median Completion Tokens**: The median number of tokens in the completions generated.
        - **IQR Completion Tokens**: Interquartile Range for the number of completion tokens.
        - **95th Percentile Completion Tokens**: Number of tokens below which 95% of the completion token counts fall.
        - **99th Percentile Completion Tokens**: Number of tokens below which 99% of the completion token counts fall.
        - **CV Completion Tokens**: Coefficient of Variation for completion tokens, indicating the relative variability.
        - **Error Rate**: The percentage of runs that resulted in errors.
        - **Error Types**: The types of errors encountered during the runs.
        - **Successful Runs**: The number of runs that completed successfully without errors.
        - **Unsuccessful Runs**: The number of runs that did not complete successfully due to errors.
        - **Throttle Count**: The number of times throttling occurred during the runs.
        - **Throttle Rate**: The rate at which throttling occurred.
        - **Best Run**: Details of the run with the best performance metrics.
        - **Worst Run**: Details of the run with the worst performance metrics.
        """)
    st.dataframe(styled_df)

    combined_stats = {}
    for stat in stats:
        combined_stats.update(stat)
    
    st.markdown("### Visual Insights")

    visualizer = ModelPerformanceVisualizer(data=combined_stats)
    visualizer.parse_data()

    # Adjusting layout for uniform plot sizes
    col1, col2, col3 = st.columns([1, 1, 1], gap="small")

    with col1:
        st.markdown("#### 🕒 Time Analysis")
        try:
            fig_time, fig_time2  = visualizer.plot_times()
            st.pyplot(fig_time2)
        except Exception as e:
            st.error(f"Error generating Time Analysis plot: {e}")

    with col2:
        st.markdown("#### 🪙 Token Analysis")
        try:
            fig_token = visualizer.plot_tokens()
            st.pyplot(fig_token)
        except Exception as e:
            st.error(f"Error generating Token Analysis plot: {e}")

    with col3:
        st.markdown("#### 🏆 Best vs Worst")
        try:
            fig_best_worst = visualizer.plot_best_worst_runs()
            st.pyplot(fig_best_worst)
        except Exception as e:
            st.error(f"Error generating Best and Worst Runs plot: {e}")

async def run_benchmark_tests() -> None:
    """
    Run the benchmark tests asynchronously.
    """
    try:
        deployment_clients = []
        for deployment_name, deployment in st.session_state.deployments.items():
            if deployment["stream"] == True:
                client = create_benchmark_streaming_client(
                    api_key=deployment["key"],
                    endpoint=deployment["endpoint"],
                    api_version=deployment["version"],
                )
            else:
                client = create_benchmark_non_streaming_client(
                    api_key=deployment["key"],
                    endpoint=deployment["endpoint"],
                    api_version=deployment["version"],
                )
            deployment_clients.append((client, deployment_name))

        tasks = [
            client.run_latency_benchmark_bulk(
                deployment_names=[deployment_name],
                max_tokens_list=max_tokens_list,
                iterations=num_iterations,
                context_tokens=context_tokens,
                temperature=temperature,
                prevent_server_caching=prevent_server_caching,
            )
            for client, deployment_name in deployment_clients
        ]

        await asyncio.gather(*tasks)

        stats = [
            client.calculate_and_show_statistics() for client, _ in deployment_clients
        ]
        st.session_state["benchmark_results"] = stats

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if run_benchmark:
    if not st.session_state.deployments:
        st.error("No deployments found. Please add a deployment in the sidebar.")
    else: 
        deployment_names = list(st.session_state.deployments.keys())
        st.info(
            f"""
            ### Benchmark Configuration Summary
            - **Benchmark Type:** {operation}
            - **Max Tokens:** {max_tokens_list}
            - **Number of Iterations:** {num_iterations}
            - **Context Tokens:** {context_tokens}
            - **Temperature:** {temperature}
            - **Prevent Server Caching:** {prevent_server_caching}
            - **Deployments:** {', '.join(deployment_names)}
            """
        )
    
    with st.spinner("Running benchmark tests..."):
        asyncio.run(run_benchmark_tests())

    if st.session_state["benchmark_results"]:
        display_statistics(st.session_state["benchmark_results"])
else: 
    st.info("👈 Please configure the benchmark settings and click 'Start Benchmark' to begin.")

def download_chat_history() -> None:
    """
    Provide a button to download the chat history as a JSON file.
    """
    chat_history_json = json.dumps(st.session_state.messages, indent=2)
    st.download_button(
        label="📜 Download Chat",
        data=chat_history_json,
        file_name="chat_history.json",
        mime="application/json",
        key="download-chat-history",
    )

def download_ai_response_as_docx_or_pdf() -> None:
    """
    Provide options to download the AI response as a DOCX or PDF file.
    """
    try:
        doc_io = markdown_to_docx(st.session_state.ai_response)
        file_format = st.selectbox("Select file format", ["DOCX", "PDF"])

        if file_format == "DOCX":
            st.download_button(
                label="📁 Download .docx",
                data=doc_io,
                file_name="AI_Generated_Guide.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="download-docx",
            )
        elif file_format == "PDF":
            st.download_button(
                label="📁 Download .pdf",
                data=doc_io,
                file_name="AI_Generated_Guide.pdf",
                mime="application/pdf",
                key="download-pdf",
            )
    except Exception as e:
        logger.error(f"Error generating {file_format} file: {e}")
        st.error(
            f"❌ Error generating {file_format} file. Please check the logs for more details."
        )

if st.session_state.ai_response:
    with st.sidebar:
        st.markdown("<hr/>", unsafe_allow_html=True)
        with st.expander("📥 Download Center", expanded=False):
            download_ai_response_as_docx_or_pdf()
            download_chat_history()

st.sidebar.write(
    """
<div style="text-align:center; font-size:30px; margin-top:10px;">
    ...
</div>
<div style="text-align:center; margin-top:20px;">
    <a href="https://github.com/pablosalvador10/gbb-ai-upgrade-llm" target="_blank" style="text-decoration:none; margin: 0 10px;">
        <img src="https://img.icons8.com/fluent/48/000000/github.png" alt="GitHub" style="width:40px; height:40px;">
    </a>
    <a href="https://www.linkedin.com/in/pablosalvadorlopez/?locale=en_US" target="_blank" style="text-decoration:none; margin: 0 10px;">
        <img src="https://img.icons8.com/fluent/48/000000/linkedin.png" alt="LinkedIn" style="width:40px; height:40px;">
    </a>
    <a href="#" target="_blank" style="text-decoration:none; margin: 0 10px;">
        <img src="https://img.icons8.com/?size=100&id=23438&format=png&color=000000" alt="Blog" style="width:40px; height:40px;">
    </a>
</div>
""",
    unsafe_allow_html=True,
)

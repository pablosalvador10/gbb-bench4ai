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

load_default_deployment()

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

        st.markdown("## 🤖 Deployment Center ")
        with st.expander("Add Your MaaS Deployment", expanded=False):
            operation = st.selectbox(
                "Choose Model Family:",
                ("AOAI", "Other"),
                index=0,
                help="Select the benchmark you want to perform to evaluate AI model performance.",
                placeholder="Select a Benchmark",
            )
            if operation == "AOAI":
                add_deployment_form()
            else: 
                st.info("Other deployment options will be available soon.")

        display_deployments()

        st.markdown("---")

        st.markdown("## ⚙️ Configuration Benchmark Settings")
        
        byop_option = st.radio(
            "BYOP (Bring Your Own Prompt)",
            options=["No", "Yes"],
            help="Select 'Yes' to bring your own prompt or 'No' to use default settings."
        )

        if byop_option == "Yes":
            uploaded_file = st.file_uploader("Upload CSV", type='csv', help="Upload a CSV file with prompts for the benchmark tests.")
            if uploaded_file is not None:
                st.write("File uploaded successfully!")
                df = pd.read_csv(uploaded_file)
                num_rows = len(df)
                st.write(f"Number of iterations because number of prompts {num_rows} provided")
        elif byop_option == "No":
            context_tokens = st.slider(
                "Context Tokens (Input)",
                min_value=100,
                max_value=5000,
                value=1000,
                help="Select the number of context tokens for each run.",
            )
            num_iterations = st.slider(
            "Number of Iterations",
            min_value=1,
            max_value=100,
            value=50,
            help="Select the number of iterations for each benchmark test.")
    
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
    with st.expander("AOAI Model Settings", expanded=False):
        # Temperature
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Adjust the temperature to control the randomness of the output. A higher temperature results in more random completions."
        )
        
        # Prevent Server Caching
        prevent_server_caching = st.radio(
            "Prevent Server Caching",
            ('Yes', 'No'),
            index=0,
            help="Choose 'Yes' to prevent server caching, ensuring that each request is processed freshly."
        )
        
        # Timeout
        timeout = st.number_input(
            "Timeout (seconds)",
            min_value=1,
            max_value=300,
            value=60,
            help="Set the maximum time in seconds before the request times out."
        )
        
        # Top P
        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
            help="Adjust Top P to control the nucleus sampling, filtering out the least likely candidates."
        )
        # Presence Penalty
        presence_penalty = st.slider(
            "Presence Penalty",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="Adjust the presence penalty to discourage or encourage repeated content in completions."
        )
        
        # Frequency Penalty
        frequency_penalty = st.slider(
            "Frequency Penalty",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="Adjust the frequency penalty to discourage or encourage frequent content in completions."
        )
        
    st.markdown("---")
    st.markdown("## 🚀 Run Benchmark")

    if not st.session_state.deployments:
        st.error("No deployments found. Please visit the Deployment Center to add deployments.")
    else: 
        deployment_names = list(st.session_state.deployments.keys())
        summary_placeholder = st.empty()

        # Fill the placeholder with the benchmark configuration summary
        summary_placeholder.info(
            f"""
            #### Benchmark Configuration Summary
            - **Benchmark Type:** {operation}
            - **Max Tokens:** {', '.join(map(str, max_tokens_list))}
            - **Number of Iterations:** {num_iterations}
            - **Context Tokens:** {context_tokens}
            - **Deployments:** {', '.join(deployment_names)}
            - **AOAI Model Settings:**
                - **Temperature:** {temperature}
                - **Prevent Server Caching:** {'Yes' if prevent_server_caching else 'No'} (Option to prevent server caching for fresh request processing.)
                - **Timeout:** {timeout} seconds
                - **Top P:** {top_p}
                - **Presence Penalty:** {presence_penalty}
                - **Frequency Penalty:** {frequency_penalty}
            """
        )
        st.markdown("")
        test_status_placeholder = st.empty()
        run_benchmark = st.button("Start Benchmark")

def display_statistics(stats: List[Dict[str, Any]]) -> None:
    """
    Display benchmark statistics in a formatted manner with enhanced user interface.
    """
    st.markdown("## Benchmark Results")
    table = []
    headers = [
        "Model_MaxTokens", "is_Streaming", "Iterations", "Regions",
        "Average TTLT (s)", "Median TTLT (s)", "IQR TTLT", "95th Percentile TTLT (s)", "99th Percentile TTLT (s)", "CV TTLT",
        "Median Prompt Tokens", "IQR Prompt Tokens", "Median Completion Tokens", "IQR Completion Tokens",
        "95th Percentile Completion Tokens", "99th Percentile Completion Tokens", "CV Completion Tokens",
        "Average TBT (ms)", "Median TBT (ms)", "IQR TBT", "95th Percentile TBT (ms)", "99th Percentile TBT (ms)",
        "Average TTFT (ms/s)", "Median TTFT (ms/s)", "IQR TTFT", "95th Percentile TTFT (ms/s)", "99th Percentile TTFT (ms/s)",
        "Error Rate", "Error Types", "Successful Runs", "Unsuccessful Runs",
        "Throttle Count", "Throttle Rate", "Best Run", "Worst Run",
    ]
    for stat in stats: 
        for key, data in stat.items():
            regions = data.get("regions", [])
            regions = [r for r in regions if r is not None]
            region_string = ", ".join(set(regions)) if regions else "N/A"
            row = [
                key,
                data.get("is_Streaming", "N/A"),
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
    
    styled_df = df.style.apply(color_lowest_median_time, subset=['Median TTLT'])
    # Add an expander with explanations for all columns in the dataframe
    with st.expander("Column Descriptions", expanded=False):
        st.markdown("""
        - **Model_MaxTokens**: The maximum number of tokens the model can process in a single request.
        - **is_Streaming**: Indicates whether the model uses streaming for processing requests.
        - **Iterations**: The number of iterations or runs performed during the analysis.
        - **Regions**: Geographic regions where the deployments were executed.
        - **Average TTLT**: The average Total Time to Last Token across all runs.
        - **Median TTLT**: The median Total Time to Last Token, reducing the impact of outliers.
        - **IQR TTLT**: Interquartile Range for Total Time to Last Token, indicating the spread of the middle 50% of the data.
        - **95th Percentile TTLT**: Time below which 95% of the Total Time to Last Token measurements fall.
        - **99th Percentile TTLT**: Time below which 99% of the Total Time to Last Token measurements fall.
        - **CV TTLT**: Coefficient of Variation for Total Time to Last Token, indicating the relative variability.
        - **Median Prompt Tokens**: The median number of tokens in the prompts used.
        - **IQR Prompt Tokens**: Interquartile Range for the number of prompt tokens.
        - **Median Completion Tokens**: The median number of tokens in the completions generated.
        - **IQR Completion Tokens**: Interquartile Range for the number of completion tokens.
        - **95th Percentile Completion Tokens**: Number of tokens below which 95% of the completion token counts fall.
        - **99th Percentile Completion Tokens**: Number of tokens below which 99% of the completion token counts fall.
        - **CV Completion Tokens**: Coefficient of Variation for completion tokens, indicating the relative variability.
        - **Average TBT**: The average Time Before Throttling across all runs.
        - **Median TBT**: The median Time Before Throttling, reducing the impact of outliers.
        - **IQR TBT**: Interquartile Range for Time Before Throttling, indicating the spread of the middle 50% of the data.
        - **95th Percentile TBT**: Time below which 95% of the Time Before Throttling measurements fall.
        - **99th Percentile TBT**: Time below which 99% of the Time Before Throttling measurements fall.
        - **Average TTFT**: The average Time to First Token across all runs.
        - **Median TTFT**: The median Time to First Token, reducing the impact of outliers.
        - **IQR TTFT**: Interquartile Range for Time to First Token, indicating the spread of the middle 50% of the data.
        - **95th Percentile TTFT**: Time below which 95% of the Time to First Token measurements fall.
        - **99th Percentile TTFT**: Time below which 99% of the Time to First Token measurements fall.
        - **Error Rate**: The percentage of runs that resulted in errors.
        - **Error Types**: The types of errors encountered during the runs.
        - **Successful Runs**: The number of runs that completed successfully without errors.
        - **Unsuccessful Runs**: The number of runs that did not complete successfully due to errors.
        - **Throttle Count**: The number of times throttling occurred during the runs.
        - **Throttle Rate**: The rate at which throttling occurred.
        - **Best Run**: Details of the run with the best performance metrics.
        - **Worst Run**: Details of the run with the worst performance metrics.
        """)
   
    styled_df = df.style
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
                timeout=timeout,
                top_p=top_p,
                n=1,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
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
    summary_placeholder.empty()
    test_status_placeholder.text("Benchmark Fired...")
    if not st.session_state.deployments:
        st.error("No deployments found. Please add a deployment in the sidebar.")
    else: 
        deployment_names = list(st.session_state.deployments.keys())
        st.info(
            f"""
            #### Benchmark Configuration Summary
            - **Benchmark Type:** {operation}
            - **Max Tokens:** {', '.join(map(str, max_tokens_list))}
            - **Number of Iterations:** {num_iterations}
            - **Context Tokens:** {context_tokens}
            - **Deployments:** {', '.join(deployment_names)}
            - **AOAI Model Settings:**
                - **Temperature:** {temperature}
                - **Prevent Server Caching:** {'Yes' if prevent_server_caching else 'No'} (Option to prevent server caching for fresh request processing.)
                - **Timeout:** {timeout} seconds
                - **Top P:** {top_p}
                - **Presence Penalty:** {presence_penalty}
                - **Frequency Penalty:** {frequency_penalty}
            """
        )
    
    with st.spinner("Running benchmark tests..."):
        asyncio.run(run_benchmark_tests())

    if st.session_state["benchmark_results"]:
        display_statistics(st.session_state["benchmark_results"])
        test_status_placeholder = st.empty()
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

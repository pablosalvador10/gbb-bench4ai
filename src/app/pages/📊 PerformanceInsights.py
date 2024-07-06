import asyncio
import json
import os
from typing import Any, Dict, List

import dotenv
import pandas as pd
import streamlit as st

from src.aoai.azure_openai import AzureOpenAIManager
from src.app.Home import (add_deployment_form, display_deployments,
                          load_default_deployment)
from src.app.outputformatting import markdown_to_docx
from src.app.results import BenchmarkPerformanceResult
from src.performance.aoaihelpers.stats import ModelPerformanceVisualizer
from src.performance.latencytest import (AzureOpenAIBenchmarkNonStreaming,
                                         AzureOpenAIBenchmarkStreaming)
from utils.ml_logging import get_logger

# Load environment variables if not already loaded
dotenv.load_dotenv(".env")

# Set up logger
logger = get_logger()


def initialize_session_state(vars: List[str], initial_values: Dict[str, Any]) -> None:
    """
    Initialize Streamlit session state with default values if not already set.

    :param vars: List of session state variable names.
    :param initial_values: Dictionary of initial values for the session state variables.
    """
    for var in vars:
        if var not in st.session_state:
            st.session_state[var] = initial_values.get(var, None)

    # Add a dictionary named 'settings' to the session state if it doesn't exist
    if "settings" not in st.session_state:
        st.session_state["settings"] = {}
    if "results" not in st.session_state:
        st.session_state["results"] = {}

    # Environment variables for Azure OpenAI and other services
    env_vars = {
        "AZURE_OPENAI_KEY": st.session_state.get(
            "AZURE_OPENAI_KEY", os.getenv("AZURE_OPENAI_KEY")
        ),
        "AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID": st.session_state.get(
            "AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID",
            os.getenv("AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID"),
        ),
        "AZURE_OPENAI_API_ENDPOINT": st.session_state.get(
            "AZURE_OPENAI_API_ENDPOINT", os.getenv("AZURE_OPENAI_API_ENDPOINT")
        ),
        "AZURE_OPENAI_API_VERSION": st.session_state.get(
            "AZURE_OPENAI_API_VERSION", os.getenv("AZURE_OPENAI_API_VERSION")
        ),
    }
    st.session_state.update(env_vars)

    # Initialize Azure OpenAI Manager if it hasn't been initialized yet
    if "azure_openai_manager" not in st.session_state:
        st.session_state["azure_openai_manager"] = AzureOpenAIManager(
            api_key=st.session_state["AZURE_OPENAI_KEY"],
            azure_endpoint=st.session_state["AZURE_OPENAI_API_ENDPOINT"],
            api_version=st.session_state["AZURE_OPENAI_API_VERSION"],
            chat_model_name=st.session_state[
                "AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID"
            ],
        )


session_vars = [
    "conversation_history",
    "ai_response",
    "chat_history",
    "messages",
    "log_messages",
    "benchmark_results",
    "deployments",
]
initial_values = {
    "conversation_history": [],
    "ai_response": "",
    "chat_history": [],
    "messages": [
        {
            "role": "assistant",
            "content": "Hello! I'm your AI-powered Performance Insights Guide. Feel free to ask me anything about your benchmark results or any queries you might have !",
        }
    ],
    "log_messages": [],
    "benchmark_results": [],
    "deployments": {},
}

initialize_session_state(session_vars, initial_values)

st.set_page_config(page_title="Performance Insights AI Assistant", page_icon="📊")

load_default_deployment()


def create_azure_openai_manager(
    api_key: str, endpoint: str, api_version: str, deployment_id: str
) -> AzureOpenAIManager:
    """
    Create a new Azure OpenAI Manager instance.

    :param api_key: API key for Azure OpenAI.
    :param endpoint: API endpoint for Azure OpenAI.
    :param api_version: API version for Azure OpenAI.
    :param deployment_id: Deployment ID for Azure OpenAI.
    :return: AzureOpenAIManager instance.
    """
    return AzureOpenAIManager(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        chat_model_name=deployment_id,
    )


def create_benchmark_non_streaming_client(
    api_key: str, endpoint: str, api_version: str
) -> AzureOpenAIBenchmarkNonStreaming:
    """
    Create a new benchmark client instance for non-streaming.

    :param api_key: API key for Azure OpenAI.
    :param endpoint: API endpoint for Azure OpenAI.
    :param api_version: API version for Azure OpenAI.
    :return: AzureOpenAIBenchmarkNonStreaming instance.
    """
    return AzureOpenAIBenchmarkNonStreaming(
        api_key=api_key, azure_endpoint=endpoint, api_version=api_version
    )


def create_benchmark_streaming_client(
    api_key: str, endpoint: str, api_version: str
) -> AzureOpenAIBenchmarkStreaming:
    """
    Create a new benchmark client instance for streaming.

    :param api_key: API key for Azure OpenAI.
    :param endpoint: API endpoint for Azure OpenAI.
    :param api_version: API version for Azure OpenAI.
    :return: AzureOpenAIBenchmarkStreaming instance.
    """
    return AzureOpenAIBenchmarkStreaming(
        api_key=api_key, azure_endpoint=endpoint, api_version=api_version
    )


def configure_sidebar() -> None:
    """
    Configure the sidebar with benchmark selection and deployment forms.
    """
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
                st.markdown(
                    """
                    **How to Run a Latency Benchmark:**

                    1. **Select model settings**: Choose models, max tokens, and iterations for your benchmark tests.
                    2. **Run the benchmark**: Click 'Run Benchmark' to start and monitor progress in real-time.
                    3. **Review results**: View detailed performance metrics after the benchmark completes.

                    Optimize your LLM experience with precise performance insights!
                """
                )

            st.markdown("---")

            st.markdown("## 🤖 Deployment Center ")
            with st.expander("Add Your MaaS Deployment", expanded=False):
                deployment_operation = st.selectbox(
                    "Choose Model Family:",
                    ("AOAI", "Other"),
                    index=0,
                    help="Select the benchmark you want to perform to evaluate AI model performance.",
                    placeholder="Select a Benchmark",
                )
                if deployment_operation == "AOAI":
                    add_deployment_form()
                else:
                    st.info("Other deployment options will be available soon.")

            display_deployments()

            st.markdown("---")

            configure_benchmark_settings()


def configure_benchmark_settings() -> None:
    """
    Configure benchmark settings in the sidebar.
    """
    st.markdown("## ⚙️ Configuration Benchmark Settings")

    byop_option = st.radio(
        "BYOP (Bring Your Own Prompts)",
        options=["No", "Yes"],
        help="Select 'Yes' to bring your own prompt or 'No' to use default settings.",
    )

    if byop_option == "Yes":
        configure_byop_settings()
    else:
        configure_default_settings()

    configure_aoai_model_settings()


def configure_byop_settings() -> None:
    """
    Configure BYOP (Bring Your Own Prompts) settings.
    """
    # Ensure 'settings' exists in 'st.session_state'
    if "settings" not in st.session_state:
        st.session_state["settings"] = {}

    num_iterations = 0
    context_tokens = "BYOP"
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type="csv",
        help="Upload a CSV file with prompts for the benchmark tests.",
    )

    # Add 'context_tokens' to the 'settings' dictionary
    st.session_state["settings"]["context_tokens"] = context_tokens

    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        try:
            df = pd.read_csv(uploaded_file)
            if "prompts" in df.columns:
                prompts = df["prompts"].tolist()
                num_iterations = len(prompts)
                # Add 'prompts' and 'num_iterations' to the 'settings' dictionary
                st.session_state["settings"]["prompts"] = prompts
                st.session_state["settings"]["num_iterations"] = num_iterations
                custom_output_tokens = st.checkbox("Custom Output Tokens")
                if custom_output_tokens:
                    max_tokens_list = configure_custom_tokens()
                else:
                    max_tokens_list = configure_default_tokens()
                st.session_state["settings"]["max_tokens_list"] = max_tokens_list
            else:
                st.error("The uploaded CSV file must contain a 'prompts' column.")
        except Exception as e:
            st.error(f"An error occurred while processing the uploaded file: {e}")


def configure_default_settings() -> None:
    """
    Configure default benchmark settings.
    """
    # Ensure 'settings' exists in 'st.session_state'
    if "settings" not in st.session_state:
        st.session_state["settings"] = {}

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
        help="Select the number of iterations for each benchmark test.",
    )
    prompts = None

    custom_output_tokens = st.checkbox("Custom Output Tokens")
    if custom_output_tokens:
        max_tokens_list = configure_custom_tokens()
    else:
        max_tokens_list = configure_default_tokens()

    # Add inputs to the 'settings' dictionary
    st.session_state["settings"]["context_tokens"] = context_tokens
    st.session_state["settings"]["num_iterations"] = num_iterations
    st.session_state["settings"][
        "prompts"
    ] = prompts  # This will be None unless modified later
    st.session_state["settings"]["custom_output_tokens"] = custom_output_tokens
    st.session_state["settings"]["max_tokens_list"] = max_tokens_list


def configure_custom_tokens() -> List[int]:
    """
    Configure custom tokens for benchmark settings.

    :return: List of custom max tokens.
    """
    custom_tokens_input = st.text_input(
        "Type your own max tokens (separate multiple values with commas):",
        help="Enter custom max tokens for each run.",
    )
    if custom_tokens_input:
        try:
            return [int(token.strip()) for token in custom_tokens_input.split(",")]
        except ValueError:
            st.error("Please enter valid integers separated by commas for max tokens.")
            return []
    return []


def configure_default_tokens() -> List[int]:
    """
    Configure default tokens for benchmark settings.

    :return: List of default max tokens.
    """
    options = [100, 500, 800, 1000, 1500, 2000]
    default_tokens = st.multiselect(
        "Select Max Output Tokens (Generation)",
        options=options,
        default=[500],
        help="Select the maximum tokens for each run.",
    )
    st.session_state["settings"]["default_tokens"] = default_tokens
    return default_tokens


def configure_aoai_model_settings() -> dict:
    """
    Configure AOAI model settings and return the values from each input.

    :return: A dictionary containing the settings values.
    """
    with st.expander("AOAI Model Settings", expanded=False):
        # Ensure 'settings' exists in 'session_state'
        if "settings" not in st.session_state:
            st.session_state["settings"] = {}

        st.session_state["settings"]["temperature"] = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Adjust the temperature to control the randomness of the output. A higher temperature results in more random completions.",
        )
        prevent_server_caching = st.radio(
            "Prevent Server Caching",
            ("Yes", "No"),
            index=0,
            help="Choose 'Yes' to prevent server caching, ensuring that each request is processed freshly.",
        )
        st.session_state["settings"]["prevent_server_caching"] = (
            True if prevent_server_caching == "Yes" else False
        )

        st.session_state["settings"]["timeout"] = st.number_input(
            "Timeout (seconds)",
            min_value=1,
            max_value=300,
            value=60,
            help="Set the maximum time in seconds before the request times out.",
        )

        st.session_state["settings"]["top_p"] = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
            help="Adjust Top P to control the nucleus sampling, filtering out the least likely candidates.",
        )

        st.session_state["settings"]["presence_penalty"] = st.slider(
            "Presence Penalty",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="Adjust the presence penalty to discourage or encourage repeated content in completions.",
        )

        st.session_state["settings"]["frequency_penalty"] = st.slider(
            "Frequency Penalty",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="Adjust the frequency penalty to discourage or encourage frequent content in completions.",
        )

    return st.session_state["settings"]


def display_code_setting_sdk(deployment_names) -> None:
    code = f"""
    from src.performance.latencytest import AzureOpenAIBenchmarkNonStreaming, AzureOpenAIBenchmarkStreaming

    API_KEY = {st.session_state["deployments"][deployment_names[0]]["key"]}
    ENDPOINT = {st.session_state["deployments"][deployment_names[0]]["endpoint"]}
    API_VERSION = {st.session_state["deployments"][deployment_names[0]]["version"]}

    # Create benchmark clients
    client_NonStreaming = AzureOpenAIBenchmarkNonStreaming(API_KEY,ENDPOINT,API_VERSION)
    client_Streaming = AzureOpenAIBenchmarkStreaming(API_KEY,ENDPOINT,API_VERSION)
    
    client_NonStreaming.run_latency_benchmark_bulk(
        deployment_names = [{deployment_names}],
        max_tokens_list = [{', '.join(map(str, st.session_state['settings']["max_tokens_list"]))}],
        iterations = {st.session_state['settings']["num_iterations"]},
        temperature = {st.session_state['settings']['temperature']},
        context_tokens = {st.session_state['settings']['context_tokens']},
        byop = {st.session_state['settings']['prompts']},
        prevent_server_caching = {st.session_state['settings']['prevent_server_caching']},
        timeout: {st.session_state['settings']['timeout']},
        top_p: {st.session_state['settings']['top_p']},
        n= 1,
        presence_penalty={st.session_state['settings']['presence_penalty']},
        frequency_penalty={st.session_state['settings']['frequency_penalty']}"""

    st.markdown("##### Test Settings")
    st.code(code, language="python")


def display_human_readable_settings(deployment_names, results) -> None:
    # Benchmark Details
    benchmark_details = f"""
    **Benchmark Details:**
    - **ID:** `{results["id"]}`
    - **Timestamp:** `{results["timestamp"]}`
    """

    # Benchmark Configuration Summary
    benchmark_configuration_summary = f"""
    #### Benchmark Configuration Summary
    - **Benchmark Type:** Latency Benchmark
    - **Max Tokens:** {', '.join(map(str, st.session_state['settings']["max_tokens_list"]))}
    - **Number of Iterations:** {st.session_state['settings']["num_iterations"]}
    - **Context Tokens:** {st.session_state['settings']['context_tokens']}
    - **Deployments:** {', '.join(deployment_names)}
    - **AOAI Model Settings:**
        - **Temperature:** {st.session_state['settings']['temperature']}
        - **Prevent Server Caching:** {'Yes' if st.session_state['settings']['prevent_server_caching'] else 'No'} (Option to prevent server caching for fresh request processing.)
        - **Timeout:** {st.session_state['settings']['timeout']} seconds
        - **Top P:** {st.session_state['settings']['top_p']}
        - **Presence Penalty:** {st.session_state['settings']['presence_penalty']}
        - **Frequency Penalty:** {st.session_state['settings']['frequency_penalty']}
    """

    # Display using st.markdown
    st.markdown(benchmark_details, unsafe_allow_html=True)
    st.markdown(benchmark_configuration_summary, unsafe_allow_html=True)


def ask_user_for_result_display_preference(results: BenchmarkPerformanceResult) -> None:
    col1, col2 = st.columns(2)
    deployment_names = list(st.session_state.deployments.keys())

    with col1:
        with st.expander("👨‍💻 Reproduce Run Using SDK", expanded=False):
            display_code_setting_sdk(deployment_names)

    with col2:
        with st.expander("👥 Benchmark Configuration Details", expanded=False):
            display_human_readable_settings(deployment_names, results)


async def run_benchmark_tests(
    test_status_placeholder: st.delta_generator.DeltaGenerator,
) -> None:
    """
    Run the benchmark tests asynchronously, with detailed configuration for each test.

    :param summary_placeholder: Streamlit placeholder for the summary information.
    :param test_status_placeholder: Streamlit placeholder for the test status.
    :param max_tokens_list: List of maximum tokens to use in each request.
    :param num_iterations: Number of iterations to run each benchmark test.
    :param context_tokens: Number of tokens to use as context for each request.
    :param temperature: Sampling temperature to use for each request.
    :param prompts: List of prompts to use for BYOP (Bring Your Own Prompt) tests.
    :param prevent_server_caching: Flag to prevent server-side caching of responses.
    :param timeout: Timeout in seconds for each request.
    :param top_p: Top-p sampling parameter to use for each request.
    :param presence_penalty: Presence penalty parameter to use for each request.
    :param frequency_penalty: Frequency penalty parameter to use for each request.
    """
    try:
        deployment_clients = [
            (
                create_benchmark_streaming_client(
                    deployment["key"], deployment["endpoint"], deployment["version"]
                )
                if deployment["stream"]
                else create_benchmark_non_streaming_client(
                    deployment["key"], deployment["endpoint"], deployment["version"]
                ),
                deployment_name,
            )
            for deployment_name, deployment in st.session_state.deployments.items()
        ]

        tasks = [
            client.run_latency_benchmark_bulk(
                deployment_names=[deployment_name],
                max_tokens_list=st.session_state["settings"]["max_tokens_list"],
                iterations=st.session_state["settings"]["num_iterations"],
                context_tokens=st.session_state["settings"]["context_tokens"],
                temperature=st.session_state["settings"]["temperature"],
                byop=st.session_state["settings"]["prompts"],
                prevent_server_caching=st.session_state["settings"][
                    "prevent_server_caching"
                ],
                timeout=st.session_state["settings"]["timeout"],
                top_p=st.session_state["settings"]["top_p"],
                n=1,
                presence_penalty=st.session_state["settings"]["presence_penalty"],
                frequency_penalty=st.session_state["settings"]["frequency_penalty"],
            )
            for client, deployment_name in deployment_clients
        ]

        await asyncio.gather(*tasks)

        stats = [
            client.calculate_and_show_statistics() for client, _ in deployment_clients
        ]
        stats_raw = [client.results for client, _ in deployment_clients]
        st.session_state["benchmark_results"] = stats
        st.session_state["benchmark_results_raw"] = stats_raw
        results = BenchmarkPerformanceResult(
            result=stats, settings=st.session_state["settings"]
        )
        st.session_state["results"][results.id] = results.to_dict()
        test_status_placeholder.markdown(
            f"Benchmark <span style='color: grey;'>{results.id}</span> Completed",
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


def display_results(
    stats: List[Dict[str, Any]] = None,
    results: BenchmarkPerformanceResult = None,
    id: str = None,
    stats_raw: List[Dict[str, Any]] = None,
) -> None:
    """
    Display benchmark statistics in a formatted manner with enhanced user interface.

    :param stats: List of benchmark statistics.
    :param stats_raw: List of raw benchmark statistics.
    """
    try:
        if id:
            results = st.session_state["results"][id]
            if results["result"] is None:
                raise ValueError("Results are not available for the given ID.")
            stats = results["result"]
    except ValueError as e:
        st.warning(
            f"⚠️ Oops! We couldn't retrieve the data for ID: {id}. Error: {e}. Sorry for the inconvenience! 😓 Please try another ID. 🔄"
        )
        return

    with st.container(border=False):
        ask_user_for_result_display_preference(results)
        if id:
            st.markdown("## 📈 Benchmark Results")
            st.toast(f"You are viewing results from Run ID: {id}", icon="ℹ️")
        else:
            st.markdown("## 📈 Benchmark Results")
            st.toast(f"You are viewing results from Run ID: {id}", icon="ℹ️")

        headers = [
            "Model_MaxTokens",
            "is_Streaming",
            "Iterations",
            "Regions",
            "Average TTLT (s)",
            "Median TTLT (s)",
            "IQR TTLT",
            "95th Percentile TTLT (s)",
            "99th Percentile TTLT (s)",
            "CV TTLT",
            "Median Prompt Tokens",
            "IQR Prompt Tokens",
            "Median Completion Tokens",
            "IQR Completion Tokens",
            "95th Percentile Completion Tokens",
            "99th Percentile Completion Tokens",
            "CV Completion Tokens",
            "Average TBT (ms)",
            "Median TBT (ms)",
            "IQR TBT",
            "95th Percentile TBT (ms)",
            "99th Percentile TBT (ms)",
            "Average TTFT (ms/s)",
            "Median TTFT (ms/s)",
            "IQR TTFT",
            "95th Percentile TTFT (ms/s)",
            "99th Percentile TTFT (ms/s)",
            "Error Rate",
            "Error Types",
            "Successful Runs",
            "Unsuccessful Runs",
            "Throttle Count",
            "Throttle Rate",
            "Best Run",
            "Worst Run",
        ]

        table = [
            [
                key,
                data.get("is_Streaming", "N/A"),
                data.get("number_of_iterations", "N/A"),
                ", ".join(set([r for r in data.get("regions", []) if r])) or "N/A",
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
                json.dumps(data.get("worst_run", {}))
                if data.get("worst_run")
                else "N/A",
            ]
            for stat in stats
            for key, data in stat.items()
        ]

        df = pd.DataFrame(table, columns=headers)

        with st.expander("Column Descriptions", expanded=False):
            st.markdown(
                """
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
                """
            )

        st.write(df.style)

        combined_stats = {key: val for stat in stats for key, val in stat.items()}

        # TIP: The following line is a dictionary comprehension that combines the stats from single runs into a single dictionary.
        # combined_stats_raw = {key: val for stat in stats_raw for key, val in stat.items()}

        st.write("### 🤔 Visual Insights")

        visualizer = ModelPerformanceVisualizer(data=combined_stats)
        visualizer.parse_data()

        # Create tabs for each plot
        tab1, tab2, tab3 = st.tabs(
            ["Completion Tokens", "Prompt Tokens", "Response Time Metrics"]
        )

        # Completion Tokens Plot
        with tab1:
            try:
                st.plotly_chart(
                    visualizer.plot_completion_tokens(), use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating Completion Tokens plot: {e}")

        # Prompt Tokens Plot
        with tab2:
            try:
                st.plotly_chart(
                    visualizer.plot_prompt_tokens(), use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating Prompt Tokens plot: {e}")

        # Response Time Metrics Comparison Plot
        with tab3:
            try:
                st.plotly_chart(
                    visualizer.plot_response_time_metrics_comparison(),
                    use_container_width=True,
                )
                st.toast("Make plots full screen for detailed analysis!")
            except Exception as e:
                st.error(f"Error generating Response Time Metrics plot: {e}")


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


def display_benchmark_summary(deployment_names):
    """
    Displays a summary of the benchmark configuration in the sidebar.

    Parameters:
    - deployment_names (list): A list of deployment names.
    """
    st.sidebar.info(
        f"""
        #### Benchmark Configuration Summary
        - **Benchmark Type:** Latency Benchmark
        - **Max Tokens:** {', '.join(map(str, st.session_state['settings']["max_tokens_list"]))}
        - **Number of Iterations:** {st.session_state['settings']["num_iterations"]}
        - **Context Tokens:** {st.session_state['settings']['context_tokens']}
        - **Deployments:** {', '.join(deployment_names)}
        - **AOAI Model Settings:**
            - **Temperature:** {st.session_state['settings']['temperature']}
            - **Prevent Server Caching:** {'Yes' if st.session_state['settings']['prevent_server_caching'] else 'No'} (Option to prevent server caching for fresh request processing.)
            - **Timeout:** {st.session_state['settings']['timeout']} seconds
            - **Top P:** {st.session_state['settings']['top_p']}
            - **Presence Penalty:** {st.session_state['settings']['presence_penalty']}
            - **Frequency Penalty:** {st.session_state['settings']['frequency_penalty']}
        """
    )


async def generate_ai_response(user_query, system_message):
    try:
        ai_response, _ = await asyncio.to_thread(
            st.session_state.azure_openai_manager.generate_chat_response,
            conversation_history=st.session_state.conversation_history,
            system_message_content=system_message,
            query=user_query,
            max_tokens=3000,
            stream=True,
        )
        return ai_response
    except Exception as e:
        st.error(f"An error occurred while generating the AI response: {e}")
        return None


def initialize_chatbot() -> None:
    """
    Initialize a chatbot interface for user interaction with enhanced features.
    """
    st.markdown(
        "<h3 style='text-align: center;'>BenchmarkAI Buddy 🤖</h3>",
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Container for the chatbot interface
    with st.container(height=500):
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                role = message["role"]
                content = message["content"]
                avatar_style = "🧑‍💻" if role == "user" else "🤖"
                with st.chat_message(role, avatar=avatar_style):
                    st.markdown(
                        f"<div style='padding: 10px; border-radius: 5px;'>{content}</div>",
                        unsafe_allow_html=True,
                    )

        # User input for feedback or additional instructions
        prompt = st.chat_input("What is up?")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(
                    f"<div style='padding: 10px; border-radius: 5px;'>{prompt}</div>",
                    unsafe_allow_html=True,
                )

            # Generate AI response (asynchronously)
            with st.chat_message("assistant", avatar="🤖"):
                stream = st.session_state.azure_openai_manager.openai_client.chat.completions.create(
                    model=st.session_state.azure_openai_manager.chat_model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert in benchmark analysis. The user will send you questions about past benchmarks "
                                "that are provided in Dictionary format. Your task is to understand these questions and provide detailed, "
                                "insightful, and accurate analysis based on the benchmark data. Be ready to answer questions and offer as much "
                                "analysis as possible to assist the user."
                            ),
                        }
                    ]
                    + [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    temperature=0.3,
                    max_tokens=1000,
                    seed=555,
                    stream=True,
                )
                ai_response = st.write_stream(stream)
                st.session_state.messages.append(
                    {"role": "assistant", "content": ai_response}
                )

    st.markdown(
        "<p style='text-align: center; font-style: italic;'>Your friendly BenchmarkAI Buddy 🤖 is here to help!</p>",
        unsafe_allow_html=True,
    )


def main() -> None:
    """
    Main function to run the Streamlit app.
    """
    configure_sidebar()

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🚀 Run Benchmark")
    test_status_placeholder = st.sidebar.empty()
    button_label = (
        "Start New Benchmark"
        if "results" in st.session_state and st.session_state["results"]
        else "Start Benchmark"
    )
    run_benchmark = st.sidebar.button(button_label)
    deployment_names = list(st.session_state.get("deployments", {}).keys())
    # Check if the benchmark should be run
    if run_benchmark:
        test_status_placeholder.markdown("Running Benchmark...🕒")
        with st.spinner(
            "🚀 Running benchmark... Please be patient, this might take a few minutes. 🕒"
        ):
            try:
                asyncio.run(run_benchmark_tests(test_status_placeholder))
            except Exception as e:
                st.error(f"An error occurred while running the benchmark: {e}")
    else:
        display_benchmark_summary(deployment_names)

    if "results" in st.session_state and st.session_state["results"]:
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 📊 Select a Benchmark Run")
        st.sidebar.markdown(
            "Please select a benchmark run from the list below to view its results:"
        )

        # Filter out keys for runs that are not empty
        run_keys = [
            key
            for key in st.session_state.get("results", {}).keys()
            if st.session_state["results"][key]["result"] is not None
        ]

        if run_keys:  # If there are non-empty runs
            # Ensure the latest run is selected by default
            default_index = len(run_keys) if run_keys else 0
            selected_run_key = st.sidebar.selectbox(
                "Select a Run",
                options=run_keys,
                format_func=lambda x: f"Run {x}",
                index=default_index - 1,  # Adjust for 0-based indexing
            )
            st.sidebar.markdown(
                f"You are currently viewing run: <span style='color: grey;'>**{selected_run_key}**</span>",
                unsafe_allow_html=True,
            )
            display_results(id=selected_run_key)
        else:
            st.info(
                "There are no runs available at this moment. Please try again later."
            )
    else:
        # This message is shown if there are no deployment names and no results in the session state
        st.info(
            "👈 Please configure the benchmark settings and click 'Start Benchmark' to begin."
        )

    st.write(
        """
        <div style="text-align:center; font-size:30px; margin-top:10px;">
            ...
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("")
    initialize_chatbot()

    if st.session_state.ai_response:
        with st.sidebar:
            st.markdown("<hr/>", unsafe_allow_html=True)
            with st.expander("📥 Download Center", expanded=False):
                download_ai_response_as_docx_or_pdf()
                download_chat_history()

    st.markdown("")
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


if __name__ == "__main__":
    main()

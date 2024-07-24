import asyncio
from functools import reduce
from typing import Any, Dict, List
import dotenv
import streamlit as st

from src.app.benchmarkbuddy import configure_chatbot
from src.app.Home import (create_benchmark_center, display_deployments,
                          load_default_deployment)
from src.app.performance.latencydisplay import (create_latency_display_dataframe,
                                    display_best_and_worst_run_analysis,
                                    display_error_and_throttle_metrics,
                                    display_full_dataframe,
                                    display_latency_metrics,
                                    display_token_metrics)
from src.app.performance.latencysettings import configure_benchmark_settings
from src.app.outputformatting import (download_ai_response_as_docx_or_pdf,
                                      download_chat_history)
from src.app.prompts import (SYSTEM_MESSAGE_LATENCY,
                             prompt_message_ai_benchmarking_buddy_latency)
from src.app.performance.results import BenchmarkPerformanceResult
from src.performance.aoaihelpers.stats import ModelPerformanceVisualizer
from src.app.performance.run import run_benchmark_tests
from my_utils.ml_logging import get_logger

# Load environment variables if not already loaded
dotenv.load_dotenv(".env")

logger = get_logger()

st.set_page_config(page_title="Performance Insights AI Assistant", page_icon="📊")

def initialize_session_state(vars: List[str], initial_values: Dict[str, Any]) -> None:
    """
    Initialize Streamlit session state with default values if not already set.

    :param vars: List of session state variable names.
    :param initial_values: Dictionary of initial values for the session state variables.
    """
    for var in vars:
        if var not in st.session_state:
            st.session_state[var] = initial_values.get(var, None)


session_vars = [
    "conversation_history",
    "ai_response",
    "chat_history",
    "messages",
    "log_messages",
    "benchmark_results",
    "deployments",
    "settings",
    "results",
    "disable_chatbot",
    "azure_openai_manager"
]
initial_values = {
    "conversation_history": [],
    "ai_response": "",
    "chat_history": [
        {
            "role": "assistant",
            "content": (
                "🚀 Ask away! I am all ears and ready to dive into your queries. "
                "I'm here to make sense of the numbers from your benchmarks and support you during your analysis! 😄📊"
            ),
        }
    ],
    "messages": [
        {
            "role": "system",
            "content": f"{SYSTEM_MESSAGE_LATENCY}",
        },
        {
            "role": "assistant",
            "content": (
                "🚀 Ask away! I am all ears and ready to dive into your queries. "
                "I'm here to make sense of the numbers from your benchmarks and support you during your analysis! 😄📊"
            ),
        },
    ],
    "log_messages": [],
    "benchmark_results": [],
    "deployments": {},
    "settings": {},
    "results": {},
    "disable_chatbot": True,
    "azure_openai_manager": None,
}

initialize_session_state(session_vars, initial_values)


def configure_sidebar() -> None:
    """
    Configure the sidebar with benchmark Center and deployment forms.
    """
    with st.sidebar:
        st.markdown("## 🤖 Deployment Center ")
        if st.session_state.deployments == {}:
            load_default_deployment()
        create_benchmark_center()
        display_deployments()

        st.sidebar.divider()

        st.markdown("## 🎛️ Benchmark Center")
        operation = st.selectbox(
            "Choose Your Benchmark:",
            ("Latency", "Throughput"),
            help="Select the benchmark you want to perform to evaluate AI model performance.",
            placeholder="Select a Benchmark",
        )

        if operation == "Latency":
            tab1, tab2, tab3 = st.sidebar.tabs(
                ["⚙️ Run Settings", "🤖 Buddy Settings", "📘 How-To Guide"]
            )

            with tab1:
                configure_benchmark_settings()
            with tab2:
                configure_chatbot()
            with tab3:
                with st.expander("🤖 Set-up BenchBuddy", expanded=False):
                    st.markdown(
                        """       
                        To fully activate and utilize BenchBuddy, 
                        please go under benchmark center and buddy setting follow these simple steps:
                    
                        1. **Activate Your AOAI Model**:
                            - Navigate to the "Add Your AOAI-model" section.
                            - Fill in the "Deployment id" with your Azure OpenAI deployment ID.
                            - Enter your "Azure OpenAI Key" for secure access.
                            - Specify the "API Endpoint" where your Azure OpenAI is hosted.
                            - Input the "API Version" to ensure compatibility.

                        2. **Configure Chatbot Behavior**:
                            - After adding your GPT model, go to "Benchmark Settings".
                            - Adjust settings to fine-tune the chatbot's responses and behavior.
                        """
                    )
                with st.expander("⏱️ Latency Benchmark", expanded=False):
                    st.markdown(
                        """
                            1. **Select model settings**: Choose models, max tokens, and iterations for your benchmark tests.
                            2. **Run the benchmark**: Click 'Run Benchmark' to start and monitor progress in real-time.
                            3. **Review results**: View detailed performance metrics after the benchmark completes.

                            Optimize your LLM experience with precise performance insights!
                        """
                    )
        elif operation == "Throughput":
            st.warning(
                "Throughput benchmarking is not available yet. Please select 'Latency'."
            )


def display_code_setting_sdk(deployment_names, results) -> None:
    settings = results["settings"]
    code = f"""
    from src.performance.latencytest import AzureOpenAIBenchmarkNonStreaming, AzureOpenAIBenchmarkStreaming

    API_KEY = "YOUR_API_KEY"
    ENDPOINT = "YOUR_ENDPOINT"
    API_VERSION = "YOUR_API_VERSION"

    # Create benchmark clients
    client_NonStreaming = AzureOpenAIBenchmarkNonStreaming(API_KEY,ENDPOINT,API_VERSION)
    client_Streaming = AzureOpenAIBenchmarkStreaming(API_KEY,ENDPOINT,API_VERSION)
    
    client_NonStreaming.run_latency_benchmark_bulk(
        deployment_names = [{deployment_names}],
        max_tokens_list = [{', '.join(map(str, settings["max_tokens_list"]))}],
        iterations = {settings["num_iterations"]},
        temperature = {settings['temperature']},
        context_tokens = {settings['context_tokens']},
        byop = {settings['prompts']},
        prevent_server_caching = {settings['prevent_server_caching']},
        timeout: {settings['timeout']},
        top_p: {settings['top_p']},
        n= 1,
        presence_penalty={settings['presence_penalty']},
        frequency_penalty={settings['frequency_penalty']}"""

    st.markdown("##### Test Settings")
    st.code(code, language="python")
    st.markdown(
        "More details on the SDK can be found [here](https://github.com/pablosalvador10/gbb-ai-upgrade-llm)."
    )


def display_human_readable_settings(deployment_names, results) -> None:
    # Benchmark Details
    benchmark_details = f"""
    **Benchmark Details:**
    - **ID:** `{results["id"]}`
    - **Timestamp:** `{results["timestamp"]}`
    """
    
    settings = results["settings"]
    # Benchmark Configuration Summary
    benchmark_configuration_summary = f"""
    #### Benchmark Configuration Summary
    - **Benchmark Type:** Latency Benchmark
    - **Max Tokens:** {', '.join(map(str, settings["max_tokens_list"]))}
    - **Number of Iterations:** {settings["num_iterations"]}
    - **Context Tokens:** {settings['context_tokens']}
    - **Deployments:** {', '.join(deployment_names)}
    - **AOAI Model Settings:**
        - **Temperature:** {settings['temperature']}
        - **Prevent Server Caching:** {'Yes' if settings['prevent_server_caching'] else 'No'} (Option to prevent server caching for fresh request processing.)
        - **Timeout:** {settings['timeout']} seconds
        - **Top P:** {settings['top_p']}
        - **Presence Penalty:** {settings['presence_penalty']}
        - **Frequency Penalty:** {settings['frequency_penalty']}
    """

    # Display using st.markdown
    st.markdown(benchmark_details, unsafe_allow_html=True)
    st.markdown(benchmark_configuration_summary, unsafe_allow_html=True)


def ask_user_for_result_display_preference(results: BenchmarkPerformanceResult) -> None:
    col1, col2 = st.columns(2)
    deployment_names = list(st.session_state.deployments.keys())

    with col1:
        with st.expander("👨‍💻 Reproduce Run Using SDK", expanded=False):
            display_code_setting_sdk(deployment_names, results)

    with col2:
        with st.expander("👥 Run Configuration Details", expanded=False):
            display_human_readable_settings(deployment_names, results)


def display_latency_results(
    results_container: st.container,
    stats: List[Dict[str, Any]] = None,
    results: Dict[str, Any] = None,
    id: str = None,
    stats_raw: List[Dict[str, Any]] = None,
) -> None:
    """
    Display benchmark statistics in a formatted manner with an enhanced user interface.

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

    df = create_latency_display_dataframe(stats)

    with results_container:
        st.markdown("## 📈 Latency Benchmark Results")
        st.toast(f"You are viewing results from Run ID: {id}", icon="ℹ️")
        st.markdown("")
        st.markdown(
            """
            🧭 **Navigating the Results**

            - **Data Analysis Section**: Start here for a comprehensive analysis of the data.
            - **Visual Insights Section**: Use this section to draw conclusions by run with complex interactions.
            - **Benchmark Buddy**: Utilize this tool for an interactive, engaging "GPT" like analysis experience.

            💡 **Tip**: To enhance your performance strategies, delve into optimizing and troubleshooting latency systems and metrics. Discover more in the full [article](#) (available soon)
            """
        )

        ask_user_for_result_display_preference(results)
        st.markdown("")
        st.write("### 🔍 Data Analysis")

        # Create tabs for each plot with improved titles
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Comprehensive Results",
                "Latency Insights",
                "Token Usage Analysis",
                "Error and Throttling Overview",
                "Best and Worst Run Comparison",
            ]
        )

        with tab1:
            display_full_dataframe(df)

        with tab2:
            display_latency_metrics(df)

        with tab3:
            display_token_metrics(df)

        with tab4:
            display_error_and_throttle_metrics(df)

        with tab5:
            display_best_and_worst_run_analysis(df)

        combined_stats = reduce(lambda a, b: {**a, **b}, stats, {})

        # TIP: The following line is a dictionary comprehension that combines the stats from single runs into a single dictionary.
        # combined_stats_raw = {key: val for stat in stats_raw for key, val in stat.items()}
        st.markdown("")
        st.write("### 🤔 Visual Insights")

        visualizer = ModelPerformanceVisualizer(data=combined_stats)
        visualizer.parse_data()

        # Create tabs for each plot
        tab1_viz, tab2_viz, tab3_viz, tab4_viz = st.tabs(
            [
                "Response Time Metrics",
                "Prompt Tokens",
                "Completion Tokens",
                "Best Vs Worst Run",
            ]
        )

        # Completion Tokens Plot
        with tab1_viz:
            try:
                st.plotly_chart(
                    visualizer.plot_response_time_metrics_comparison(),
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Error generating Response Time Metrics plot: {e}")

        # Prompt Tokens Plot
        with tab2_viz:
            try:
                st.plotly_chart(
                    visualizer.plot_prompt_tokens(), use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating Prompt Tokens plot: {e}")

        # Response Time Metrics Comparison Plot
        with tab3_viz:
            try:
                st.plotly_chart(
                    visualizer.plot_completion_tokens(), use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating Completion Tokens plot: {e}")

        with tab4_viz:
            try:
                st.plotly_chart(
                    visualizer.plot_best_worst_runs(),
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Error generating Response Time Metrics plot: {e}")


def display_benchmark_summary(deployment_names):
    """
    Displays a summary of the benchmark configuration in the sidebar.

    Parameters:
    - deployment_names (list): A list of deployment names.
    """
    st.sidebar.info(
        f"""
        #### Benchmark Run Configuration Summary
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
        "<h4 style='text-align: center;'>BenchBuddy 🤖</h4>",
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": (
                "🚀 Ask away! I am all ears and ready to dive into your queries. "
                "I'm here to make sense of the numbers from your benchmarks and support you during your analysis! 😄📊"
            ),
        }
    ]
    if "messages" not in st.session_state:
        st.session_state.messages = [
        {
            "role": "system",
            "content": f"{SYSTEM_MESSAGE_LATENCY}",
        },
        {
            "role": "assistant",
            "content": (
                "🚀 Ask away! I am all ears and ready to dive into your queries. "
                "I'm here to make sense of the numbers from your benchmarks and support you during your analysis! 😄📊"
            ),
        },
    ]

    respond_conatiner = st.container(height=400)

    with respond_conatiner:
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]
            avatar_style = "🧑‍💻" if role == "user" else "🤖"
            with st.chat_message(role, avatar=avatar_style):
                st.markdown(
                    f"<div style='padding: 10px; border-radius: 5px;'>{content}</div>",
                    unsafe_allow_html=True,
                )
    
    warning_issue_performance = st.empty()
    if st.session_state.get("azure_openai_manager") is None:
        warning_issue_performance.warning(
            "Oops! It seems I'm currently unavailable. 😴 Please ensure the LLM is configured correctly in the Benchmark Center and Buddy settings. Need help? Refer to the 'How To' guide for detailed instructions! 🧙"
        )
    prompt = st.chat_input("Ask away!", disabled=st.session_state.disable_chatbot)
    if prompt:
        prompt_ai_ready = prompt_message_ai_benchmarking_buddy_latency(
            st.session_state["results"], prompt
        )
        st.session_state.messages.append({"role": "user", "content": prompt_ai_ready})
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with respond_conatiner:
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
                            "content": (SYSTEM_MESSAGE_LATENCY),
                        }
                    ]
                    + [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    temperature=st.session_state["settings_buddy"]["temperature"],
                    max_tokens=st.session_state["settings_buddy"]["max_tokens"],
                    presence_penalty=st.session_state["settings_buddy"][
                        "presence_penalty"
                    ],
                    frequency_penalty=st.session_state["settings_buddy"][
                        "frequency_penalty"
                    ],
                    seed=555,
                    stream=True,
                )
                ai_response = st.write_stream(stream)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": ai_response}
                )


def main() -> None:
    """
    Main function to run the Streamlit app.
    """
    initialize_session_state(session_vars, initial_values)

    configure_sidebar()

    st.sidebar.divider()

    # Create containers for displaying benchmark results
    results_container = st.container()

    st.sidebar.markdown("## ⏱️ Runs Center")

    tab1_runs, tab2_runs = st.sidebar.tabs(
        ["🚀 Trigger Benchmark", "🗃️ Historical Benchmarks"]
    )

    # Tab for triggering benchmarks
    with tab1_runs:
        summary_container = st.container()  # Create the container
        button_label = (
            "Start New Benchmark"
            if "results" in st.session_state and st.session_state["results"]
            else "Start Benchmark"
        )
        run_benchmark = st.button(button_label)

    with results_container:
        if run_benchmark:
            summary_container.empty()
            if run_benchmark:
                with st.spinner(
                    "🚀 Running benchmark... Please be patient, this might take a few minutes. 🕒"
                ):
                    try:
                        asyncio.run(run_benchmark_tests(summary_container))
                    except Exception as e:
                        st.error(f"An error occurred while running the benchmark: {e}")
                        st.stop()
        else:
            deployment_names = list(st.session_state.deployments.keys())
            summary_container.info(
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

    selected_run_key = None
    # Tab for viewing historical benchmarks
    with tab2_runs:
        if "results" in st.session_state and st.session_state["results"]:
            st.markdown(
                "Please select a benchmark run from the list below to view its results:"
            )

            run_keys = [
                key
                for key in st.session_state.get("results", {}).keys()
                if st.session_state["results"][key]["result"] is not None
            ]

            if run_keys:
                default_index = len(run_keys) if run_keys else 0
                selected_run_key = st.selectbox(
                    "Select a Run",
                    options=run_keys,
                    format_func=lambda x: f"Run {x}",
                    index=default_index - 1,  # Select the last run by default
                )
                st.markdown(
                    f"You are currently viewing run: <span style='color: grey;'>**{selected_run_key}**</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.info(
                    "There are no runs available at this moment. Please try again later."
                )
        else:
            st.warning(
                "There are no runs available at this moment. Please try again later."
            )
            results_container.info(
                "👈 Hey - you haven't fired any benchmarks yet. Please configure the benchmark settings and click 'Start Benchmark' to begin."
            )

    if selected_run_key:
        display_latency_results(
            results_container=results_container, id=selected_run_key
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
            <a href="https://github.com/pablosalvador10" target="_blank" style="text-decoration:none; margin: 0 10px;">
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

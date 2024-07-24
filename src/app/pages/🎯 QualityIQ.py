import asyncio
from typing import Any, Dict, List

import dotenv
import streamlit as st

from src.app.quality.runs import run_benchmark_tests

from src.app.benchmarkbuddy import configure_chatbot
from src.app.quality.llm_slm_settings import (understanding_configuration, 
                                              configure_retrieval_settings,
                                              configure_rai_settings)
from my_utils.ml_logging import get_logger
from src.app.Home import (create_benchmark_center, display_deployments,
                          load_default_deployment)
from src.app.prompts import (SYSTEM_MESSAGE_QUALITY,
                             prompt_message_ai_benchmarking_buddy_quality)

from src.app.quality.resources import display_resources
from src.app.quality.displayquality import display_quality_results

# Load environment variables
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


session_vars = [
    "conversation_history_quality",
    "ai_response_quality",
    "chat_history_quality",
    "messages_quality",
    "log_messages_quality",
    "deployments",
    "settings_quality",
    "results_quality",
    "disable_chatbot",
    "azure_openai_manager"
]
initial_values = {
    "conversation_history_quality": [],
    "ai_response_quality": "",
    "chat_history_quality": [
        {
            "role": "assistant",
            "content": (
                "🚀 Ask away! I am all ears and ready to dive into your queries. "
                "I'm here to make sense of the numbers from your benchmarks and support you during your analysis! 😄📊"
            ),
        }
    ],
    "messages_quality": [
        {
            "role": "system",
            "content": f"{SYSTEM_MESSAGE_QUALITY}",
        },
        {
            "role": "assistant",
            "content": (
                "🚀 Ask away! I am all ears and ready to dive into your queries. "
                "I'm here to make sense of the numbers from your benchmarks and support you during your analysis! 😄📊"
            ),
        },
    ],
    "log_messages_quality": [],
    "deployments": {},
    "settings_quality": {},
    "results_quality": {},
    "disable_chatbot": True,
    "azure_openai_manager": None,
}

initialize_session_state(session_vars, initial_values)

# initialize metrics list
metrics_list = ["Accuracy", "Answer Similarity"]
context_metrics_list = ["Context Similarity"]

def configure_sidebar() -> None:
    """
    Configure the sidebar with benchmark Center and deployment forms, allowing users to choose between evaluating a Large Language Model (LLM) or a System based on LLM.
    """

    with st.sidebar:
        st.markdown("## 🤖 Deployment Center ")
        if st.session_state.deployments == {}:
            load_default_deployment()
        create_benchmark_center()
        display_deployments()

        st.sidebar.divider()

        st.markdown("## 🎛️ Benchmark Center")
       
        tab1, tab2, tab3 = st.sidebar.tabs(
                ["⚙️ Run Settings", "🤖 Buddy Settings", "📘 How-To Guide"]
            )
        with tab1:
            st.markdown("""
            ## Focus your LLM/SLM benchmark

            - **🧠 Understanding**: Evaluate the reasoning and overall performance using well-known datasets like MMLU, MedPub, and TruthfulQA.
            - **⚙️ Retrieval System/QA**: Assess an LLM-based system as a whole, considering context and understand domani-specific accuracy.
            - **🛡️ RAI (Responsible AI)**: Ensure the model meets responsible AI standards.
            """)

            # Create inner tabs for detailed configuration
            tab1_inner, tab2_inner, tab3_inner = st.tabs(["🧠 Understanding", "⚙️ Retrieval System/QA", "🛡️ RAI"])

            with tab1_inner:
                with st.expander("📊 What metrics are we evaluating?", expanded=False): 
                    st.markdown("""
                        - **MMLU**: Measures domain knowledge across multiple fields.
                        - **MedPub**: Evaluates understanding of medical literature.
                        - **TruthfulQA**: Tests for truthfulness in responses.
                    """)
                understanding_configuration()

            with tab2_inner:
                # Add any additional configuration or content for System
                with st.expander("📊 What metrics are we evaluating?", expanded=False): 
                    st.markdown("""
                    - `f1_score`: Range [0, 1]
                    - `gpt_groundedness`: Range [0, 5]
                    - `gpt_relevance`: Range [0, 5]
                    - `gpt_coherence`: Range [0, 5]
                    - `gpt_fluency`: Range [0, 5]
                    - `gpt_similarity`: Range [0, 5]
                    """)
                configure_retrieval_settings()

            with tab3_inner:
                with st.expander("📊 What metrics are we evaluating?", expanded=False):
                    st.markdown("""
                    - `self_harm_defect_rate`: Range [0, 1]
                    - `sexual_defect_rate`: Range [0, 1]
                    - `hate_unfairness_defect_rate`: Range [0, 1]
                    - `violence_defect_rate`: Range [0, 1]
                    """)
                configure_rai_settings()
                          
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
            display_resources()

top_bar = st.empty()


def initialize_chatbot() -> None:
    """
    Initialize a chatbot interface for user interaction with enhanced features.
    """
    #FIXME: adapt me to QualityMetrics
    st.markdown(
        "<h4 style='text-align: center;'>BenchBuddy 🤖</h4>",
        unsafe_allow_html=True,
    )

    if "chat_history_quality" not in st.session_state:
        st.session_state.chat_history_quality = [
        {
            "role": "assistant",
            "content": (
                "🚀 Ask away! I am all ears and ready to dive into your queries. "
                "I'm here to make sense of the numbers from your benchmarks and support you during your analysis! 😄📊"
            ),
        }
    ]
    if "messages_quality" not in st.session_state:
        st.session_state.messages_quality = [
        {
            "role": "system",
            "content": f"{SYSTEM_MESSAGE_QUALITY}",
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
        for message in st.session_state.chat_history_quality:
            role = message["role"]
            content = message["content"]
            avatar_style = "🧑‍💻" if role == "user" else "🤖"
            with st.chat_message(role, avatar=avatar_style):
                st.markdown(
                    f"<div style='padding: 10px; border-radius: 5px;'>{content}</div>",
                    unsafe_allow_html=True,
                )


    warning_issue_quality = st.empty()
    if st.session_state.get("azure_openai_manager") is None:
        warning_issue_quality.warning(
            "Oops! It seems I'm currently unavailable. 😴 Please ensure the LLM is configured correctly in the Benchmark Center and Buddy settings. Need help? Refer to the 'How To' guide for detailed instructions! 🧙"
        )

    prompt = st.chat_input("Ask away!", disabled=st.session_state.disable_chatbot)
    if prompt:
        prompt_ai_ready = prompt_message_ai_benchmarking_buddy_quality(
            st.session_state["results_quality"], prompt
        )
        st.session_state.messages_quality.append({"role": "user", "content": prompt_ai_ready})
        st.session_state.chat_history_quality.append({"role": "user", "content": prompt})
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
                            "content": (SYSTEM_MESSAGE_QUALITY),
                        }
                    ]
                    + [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages_quality
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
                st.session_state.chat_history_quality.append(
                    {"role": "assistant", "content": ai_response}
                )

def display_configuration_summary(summary_container:st.container):
    """
    Display the current benchmark configuration summary.
    """
    settings = st.session_state.get("settings_quality", {})
    benchmark_selection = settings.get("benchmark_selection", [])
    benchmark_selection_multiselect = settings.get("benchmark_selection_multiselect", [])
    benchmark = benchmark_selection + benchmark_selection_multiselect
    
    summary_lines = [
        "#### Benchmark Configuration Summary",
        f"- **Benchmark Type:** Quality Benchmark",
        f"- **Tests Selected for Run:** {', '.join(benchmark)}",
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
        if f"evaluation_clients_retrieval_BYOP" not in st.session_state["settings_quality"]:
            st.session_state["settings_quality"][f"evaluation_clients_retrieval_df_BYOP"] = False
        
        retrieval_setting = st.session_state["settings_quality"][f"evaluation_clients_retrieval_df_BYOP"]
        summary_lines.append(f"  - Retrieval BYOP Selected: {retrieval_setting}")

    if "RAI" in benchmark:
        if f"evaluation_clients_rai_BYOP" not in st.session_state["settings_quality"]:
            st.session_state["settings_quality"][f"evaluation_clients_rai_df_BYOP"] = False
        
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
    
    # Display the updated summary
    summary_container.info("\n".join(summary_lines))


def main():
    
    #FIXME: Add page title and icon
    #st.set_page_config(page_title="Quality Benchmarking", page_icon="🎯 ")

    initialize_session_state(session_vars, initial_values)

    configure_sidebar()

    results_container = st.container()

    st.sidebar.divider()

    st.sidebar.markdown("## ⏱️ Runs Center")

    tab1_runs, tab2_runs = st.sidebar.tabs(
        ["🚀 Trigger Benchmark", "🗃️ Historical Benchmarks"]
    )

    with tab1_runs:
        summary_container = st.container()  # Create the container
        if "results_quality" in st.session_state and st.session_state["results_quality"]:
            button_label = "Start New Benchmark"
        else:
            button_label = "Start Benchmark"
        run_benchmark = st.button(button_label)
        with results_container:
            if run_benchmark:
                summary_container.warning(
                        "Warning: Editing sidebar while benchmark is running will kill the job."
                    )
                with st.spinner(
                    "🚀 Running benchmark... Please be patient, this might take a few minutes. 🕒"
                ):
                    asyncio.run(run_benchmark_tests())
                summary_container.empty()
            else:
                display_configuration_summary(summary_container)

    selected_run_key = None
    # Tab for viewing historical benchmarks
    with tab2_runs:
        if "results_quality" in st.session_state and st.session_state["results_quality"]:
            st.markdown(
                "Please select a benchmark run from the list below to view its results:"
            )

            run_keys = [
                key
                for key in st.session_state.get("results_quality", {}).keys()
                if st.session_state["results_quality"][key] is not None
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
        display_quality_results(
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

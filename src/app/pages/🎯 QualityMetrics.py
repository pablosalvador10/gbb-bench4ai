import asyncio
from typing import Any, Dict, List

import dotenv
import copy
import pandas as pd
import streamlit as st
from src.app.quality.results import BenchmarkQualityResult

from src.quality.evals import MMLU, CustomEval, PubMedQA, TruthfulQA
from src.app.benchmarkbuddy import configure_chatbot
from src.app.quality.llm_slm_settings import (understanding_configuration, 
                                              configure_retrieval_settings,
                                              configure_rai_settings)
from utils.ml_logging import get_logger
from src.app.Home import (create_benchmark_center, display_deployments,
                          load_default_deployment)
from src.app.prompts import (SYSTEM_MESSAGE_LATENCY,
                             prompt_message_ai_benchmarking_buddy_latency)

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
                with st.expander("📊 Metrics", expanded=False): 
                    st.markdown("""
                    - **Datasets**: MMLU, MedPub, TruthfulQA
                    """)
                # Call the function for LLM/SLM benchmark configuration
                understanding_configuration()

            with tab2_inner:
                # Add any additional configuration or content for System
                with st.expander("📊 Metrics", expanded=False): 
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
                with st.expander("📊 Metrics", expanded=False): 
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
            with st.expander("🤖 Set-up BenchmarkAI Buddy", expanded=False):
                st.markdown(
                    """       
                    To fully activate and utilize BenchmarkAI Buddy, 
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

# Function to get the task list for the selected benchmark
def get_task_list(test: str = None):
    objects = []

    # Inside the get_task_list function, before creating objects
    if "settings_quality" in st.session_state:
        settings = st.session_state["settings_quality"]
    else: 
        st.error("No settings found in session state.")

    for deployment_name, deployment in st.session_state.deployments.items():
        deployment_config = {
            "key": deployment.get("key"),
            "endpoint": deployment.get("endpoint"),
            "model": deployment_name,
            "version": deployment.get("version"),
        }
        if test == "mmlu":
            mmlu_categories = settings.get("mmlu_categories", [])
            mmlu_subsample = settings.get("mmlu_subsample", 100)
            obj = MMLU(
                deployment_config=deployment_config,
                sample_size=mmlu_subsample / 100,
                log_level="INFO",
                categories=mmlu_categories,
            )
            data = obj.load_data(dataset="cais/mmlu", subset="all", split="test")
            data = obj.transform_data(df=data)
        elif test == "medpub":
            medpub_subsample = settings.get("medpub_subsample", 100)
            obj = PubMedQA(
                deployment_config=deployment_config,
                sample_size=medpub_subsample / 100,
                log_level="ERROR",
            )
            data = obj.load_data(
                dataset="qiaojin/PubMedQA",
                subset="pqa_labeled",
                split="train",
                flatten=True,
            )
            data = obj.transform_data(df=data)
        elif test == "truthfulqa":
            truthful_subsample = settings.get("truthful_subsample", 100)
            obj = TruthfulQA(
                deployment_config=deployment_config,
                sample_size=truthful_subsample / 100,
                log_level="ERROR",
            )
            data = obj.load_data(
                dataset="truthful_qa", subset="multiple_choice", split="validation"
            )
            data = obj.transform_data(df=data)
        elif test == "custom":
            custom_metrics = settings.get("custom_benchmark", {}).get("metrics_list", [])
            custom_subsample = settings.get("custom_subsample", 100)
            custom_df = settings.get("custom_benchmark", {}).get("custom_df", pd.DataFrame())
            obj = CustomEval(
                deployment_config=deployment_config,
                metrics_list=custom_metrics,
                sample_size=custom_subsample / 100,
                log_level="ERROR",
            )
            data = obj.transform_data(df=custom_df)

        objects.append(obj)

    tasks = [asyncio.create_task(obj.test(data=data)) for obj in objects]
    return tasks

import concurrent.futures

def run_retrieval_quality_for_client(client):
    try:
        results_retrieval = client.run_retrieval_quality(data_input=st.session_state["evaluation_clients_retrieval_df"])
        # Assuming client.model_config.azure_deployment exists and is unique for each client
        deployment_name = client.model_config.azure_deployment
        return (deployment_name, results_retrieval)
    except Exception as e:
        print(f"An error occurred: {e}")
        return (client.model_config.azure_deployment, None)

def run_retrieval_quality_in_parallel():
    clients = st.session_state["evaluation_clients_retrieval"]
    results_gpt_evals = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(run_retrieval_quality_for_client, clients))
    
    for deployment_name, result in results:
        if result is not None:
            results_gpt_evals[deployment_name] = result["metrics"]
    
    return results_gpt_evals

# Define an asynchronous function to run benchmark tests and log progress
async def run_benchmark_tests():
    try:
        results = []
        if "results_quality" not in st.session_state:
            st.session_state["results_quality"] = {}
        
        #FIXME: decouple
        results_df = pd.DataFrame()
        results_df = pd.DataFrame()
        truthful_results = pd.DataFrame()
        mmlu_results = pd.DataFrame()
        medpub_results = pd.DataFrame()

        if "settings_quality" in st.session_state:
            settings = st.session_state["settings_quality"]
            if "benchmark_selection" in settings:
                if "MMLU" in settings["benchmark_selection"]:
                    mmlu_tasks = get_task_list(test="mmlu")
                    mmlu_stats = await asyncio.gather(*mmlu_tasks)
                    mmlu_results = pd.concat(mmlu_stats)
                    results.append(mmlu_results)

                if "MedPub QA" in settings["benchmark_selection"]:
                    logger.info("Running MedPub QA benchmark")
                    medpub_tasks = get_task_list(test="medpub")
                    medpub_stats = await asyncio.gather(*medpub_tasks)
                    medpub_results = pd.concat(medpub_stats)
                    results.append(medpub_results)

                if "Truthful QA" in settings["benchmark_selection"]:
                    logger.info("Running Truthful QA benchmark")
                    truthful_tasks = get_task_list(test="truthfulqa")
                    truthful_stats = await asyncio.gather(*truthful_tasks)
                    truthful_results = pd.concat(truthful_stats)
                    results.append(truthful_results)
                    

                if "Custom Evaluation" in settings["benchmark_selection"]:
                    logger.info("Running Custom Evaluation")
                    custom_tasks = get_task_list(test="custom")
                    custom_stats = await asyncio.gather(*custom_tasks)
                    custom_results = pd.concat(custom_stats)
                    results.append(custom_results)

                if "retrieval" in settings["benchmark_selection"]:
                    logger.info("Running Retrieval System benchmark")
                    # Execute the function and store results in a dictionary
                    results_gpt_evals_dict = run_retrieval_quality_in_parallel()
                    #results_gpt_evals_dict = st.session_state[].run_retrieval_quality(data_input=st.session_state["evaluation_clients_retrieval_df"])
                    st.info(results_gpt_evals_dict)

                results_df = pd.concat(results)
                results_df = results_df if isinstance(results_df, pd.DataFrame) else pd.DataFrame()
                truthful_results = truthful_results if isinstance(truthful_results, pd.DataFrame) else pd.DataFrame()
                mmlu_results = mmlu_results if isinstance(mmlu_results, pd.DataFrame) else pd.DataFrame()
                medpub_results = medpub_results if isinstance(medpub_results, pd.DataFrame) else pd.DataFrame()
                results = {
                    "all_results": results_df,
                    "truthful_results": truthful_results,
                    "mmlu_results": mmlu_results,
                    "medpub_results": medpub_results,
                }
                # Create a deep copy of the settings to ensure it remains unchanged
                settings_snapshot = copy.deepcopy(settings)

                # Use the deep copied settings when creating the BenchmarkQualityResult instance
                results_quality = BenchmarkQualityResult(result=results, settings=settings_snapshot)
                st.session_state["results_quality"][results_quality.id] = results_quality.to_dict()

    except Exception as e:
        top_bar.error(f"An error occurred: {str(e)}")

def initialize_chatbot() -> None:
    """
    Initialize a chatbot interface for user interaction with enhanced features.
    """
    #FIXME: adapt me to QualityMetrics
    st.markdown(
        "<h4 style='text-align: center;'>BenchmarkAI Buddy 🤖</h4>",
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
        prompt_ai_ready = prompt_message_ai_benchmarking_buddy_latency(
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
                            "content": (SYSTEM_MESSAGE_LATENCY),
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
    
    summary_lines = [
        "#### Benchmark Configuration Summary",
        f"- **Benchmark Type:** Quality Benchmark",
        f"- **Tests:** {', '.join(benchmark_selection)}",
    ]

    if "MMLU" in benchmark_selection:
        mmlu_categories = settings.get("mmlu_categories", [])
        mmlu_subsample = settings.get("mmlu_subsample", 0)
        summary_lines.append(f"  - **MMLU Categories:** {', '.join(mmlu_categories)}")
        summary_lines.append(f"  - **MMLU Subsample:** {mmlu_subsample}%")

    if "MedPub QA" in benchmark_selection:
        medpub_subsample = settings.get("medpub_subsample", 0)
        summary_lines.append(f"  - **MedPub QA Subsample:** {medpub_subsample}%")

    if "Truthful QA" in benchmark_selection:
        truthful_subsample = settings.get("truthful_subsample", 0)
        summary_lines.append(f"  - **Truthful QA Subsample:** {truthful_subsample}%")

    if "Custom Evaluation" in benchmark_selection:
        custom_settings = settings.get("custom_benchmark", {})
        prompt_col = custom_settings.get("prompt_col", "None")
        ground_truth_col = custom_settings.get("ground_truth_col", "None")
        context_col = custom_settings.get("context_col", "None")
        summary_lines.append(f"  - **Custom Prompt Column:** {prompt_col}")
        summary_lines.append(f"  - **Custom Ground Truth Column:** {ground_truth_col}")
        summary_lines.append(f"  - **Custom Context Column:** {context_col}")

    summary_container.info("\n".join(summary_lines))


def main():
    
    #FIXME: Add page title and icon
    #st.set_page_config(page_title="Quality Benchmarking", page_icon="🎯 ")

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
        if run_benchmark:
            if run_benchmark:
                with st.spinner(
                    "🚀 Running benchmark... Please be patient, this might take a few minutes. 🕒"
                ):
                    top_bar.warning(
                        "Warning: Editing sidebar while benchmark is running will kill the job."
                    )
                    asyncio.run(run_benchmark_tests())
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

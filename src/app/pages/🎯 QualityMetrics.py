import asyncio
from typing import Any, Dict, List

import dotenv
import pandas as pd
import streamlit as st
from src.app.quality.results import BenchmarkQualityResult

from src.quality.evals import MMLU, CustomEval, PubMedQA, TruthfulQA
from src.app.benchmarkbuddy import configure_chatbot
from src.app.quality.llm_slm_settings import slm_llm_benchmark_configuration
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
        operation = st.selectbox(
            "Choose Your Evaluation Focus:",
            ("LLM/SLM", "System"),
            help="""Select the focus of your benchmark:
                    - 'LLM' to evaluate the performance of a standalone Large Language Model. This includes metrics like accuracy, fluency, and coherence.
                    - 'System' to assess an LLM-based system as a whole, considering aspects such as integration, latency, throughput, and user experience.""",
            placeholder="Select a Focus",
        )

        if operation == "LLM/SLM":
            tab1, tab2, tab3 = st.sidebar.tabs(
                    ["⚙️ Run Settings", "🤖 Buddy Settings", "📘 How-To Guide"]
                )
            with tab1:
                slm_llm_benchmark_configuration()
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
        elif operation == "System":
            st.warning(
                "Throughput benchmarking is not available yet. Please select 'Latency'."
            )

top_bar = st.empty()
results_c = st.container()
batch_c = st.container()

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
                    mmlu_categories = settings.get("mmlu_categories", [])
                    mmlu_subsample = settings.get("mmlu_subsample", 100)
                    batch_c.markdown("#### MMLU Results")
                    batch_c.write(f"Subsample: {mmlu_subsample}% of each category")
                    batch_c.write(f"Categories: {str(mmlu_categories)}")
                    batch_c.dataframe(mmlu_results.drop("test", axis=1), hide_index=True)
                    results.append(mmlu_results)

                if "MedPub QA" in settings["benchmark_selection"]:
                    logger.info("Running MedPub QA benchmark")
                    medpub_tasks = get_task_list(test="medpub")
                    medpub_stats = await asyncio.gather(*medpub_tasks)
                    medpub_results = pd.concat(medpub_stats)
                    medpub_subsample = settings.get("medpub_subsample", 100)
                    batch_c.markdown("#### MedPub QA Results")
                    batch_c.write(
                        f"Sample Size: {int((medpub_subsample/100)*1000)} ({medpub_subsample}% of 1,000 samples)"
                    )
                    batch_c.dataframe(medpub_results.drop("test", axis=1), hide_index=True)
                    results.append(medpub_results)

                if "Truthful QA" in settings["benchmark_selection"]:
                    logger.info("Running Truthful QA benchmark")
                    truthful_tasks = get_task_list(test="truthfulqa")
                    truthful_stats = await asyncio.gather(*truthful_tasks)
                    truthful_results = pd.concat(truthful_stats)
                    truthful_subsample = settings.get("truthful_subsample", 100)
                    batch_c.markdown("#### Truthful QA Results")
                    batch_c.write(
                        f"Sample Size: {int((truthful_subsample/100)*814)} ({truthful_subsample}% of 814 samples)"
                    )
                    batch_c.dataframe(truthful_results.drop("test", axis=1), hide_index=True)
                    results.append(truthful_results)
                    

                if "Custom Evaluation" in settings["benchmark_selection"]:
                    logger.info("Running Custom Evaluation")
                    custom_tasks = get_task_list(test="custom")
                    custom_stats = await asyncio.gather(*custom_tasks)
                    custom_results = pd.concat(custom_stats)
                    custom_subsample = settings.get("custom_subsample", 100)
                    custom_df = settings.get("custom_benchmark", {}).get("custom_df", pd.DataFrame())
                    batch_c.markdown("#### Custom Evaluation Results")
                    batch_c.write(
                        f"Sample Size: {int((custom_subsample/100)*custom_df.shape[0])} ({custom_subsample}% of {custom_df.shape[0]} samples)"
                    )
                    batch_c.dataframe(custom_results, hide_index=True)
                    results.append(custom_results)

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
            results_quality = BenchmarkQualityResult(result=results, settings=settings)
            st.session_state["results_quality"][results_quality.id] = results_quality.result

    except Exception as e:
        top_bar.error(f"An error occurred: {str(e)}")

# Main layout for initial submission

def initialize_chatbot() -> None:
    """
    Initialize a chatbot interface for user interaction with enhanced features.
    """
    st.markdown(
        "<h4 style='text-align: center;'>BenchmarkAI Buddy 🤖</h4>",
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []

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

    # User input for feedback or additional instructions
    warning_issue = st.empty()
    if "azure_openai_manager" not in st.session_state:
        warning_issue.warning(
            "Oops! I'm taking a nap right now. 😴 To wake me up, please set up the LLM in the Benchmark center and Buddy settings. Stuck? The 'How To' guide has all the secret wake-up spells! 🧙‍♂️"
        )

    prompt = st.chat_input("Ask away!", disabled=st.session_state.disable_chatbot)
    if prompt:
        prompt_ai_ready = prompt_message_ai_benchmarking_buddy_latency(
            st.session_state["results_quality"], prompt
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
                    "Running benchmark tests. Outputs will appear as benchmarks complete. This may take a while..."
                ):
                    top_bar.warning(
                        "Warning: Editing sidebar while benchmark is running will kill the job."
                    )
                    asyncio.run(run_benchmark_tests())
        else:
            deployment_names = list(st.session_state.deployments.keys())
            summary_container.info(
                f"""
                #### Benchmark Configuration Summary
                - **Benchmark Type:** Quality Benchmark
                - **Deployments:** {', '.join(deployment_names)}
                """
            )

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




if __name__ == "__main__":
    main()

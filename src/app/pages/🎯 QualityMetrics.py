import asyncio
import os
from typing import Any, Dict, Optional

import dotenv
import pandas as pd
import plotly.express as px
import streamlit as st

from multiprocessing import Pool, cpu_count
from src.quality.evals import MMLU, CustomEval, PubMedQA, TruthfulQA
from utils.ml_logging import get_logger
from functools import partial


def initialize_session_state(defaults: Dict[str, Any]) -> None:
    """
    Initialize Streamlit session state with default values if not already set.

    Args:
        defaults (Dict[str, Any]): Dictionary of default values.
    """
    for var, value in defaults.items():
        if var not in st.session_state:
            st.session_state[var] = value


def load_default_deployment(
    name: Optional[str] = None,
    key: Optional[str] = None,
    endpoint: Optional[str] = None,
    version: Optional[str] = None,
) -> None:
    """
    Load default deployment settings, optionally from provided parameters.
    Ensures that a deployment with the same name does not already exist.
    """
    # Ensure deployments is a dictionary
    if "deployments" not in st.session_state or not isinstance(
        st.session_state.deployments, dict
    ):
        st.session_state.deployments = {}

    # Check if the deployment name already exists
    deployment_name = (
        name if name else os.getenv("AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID")
    )
    if deployment_name in st.session_state.deployments:
        return  # Exit the function if deployment already exists

    default_deployment = {
        "name": deployment_name,
        "key": key if key else os.getenv("AZURE_OPENAI_KEY"),
        "endpoint": endpoint if endpoint else os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        "version": version if version else os.getenv("AZURE_OPENAI_API_VERSION"),
        "stream": False,
    }


def add_deployment_aoai_form() -> None:
    """
    Render the form to add a new Azure OpenAI deployment.
    """
    with st.form("add_deployment_aoai_form"):
        deployment_name = st.text_input(
            "Deployment id",
            help="Enter the deployment ID for Azure OpenAI.",
            placeholder="e.g., chat-gpt-1234abcd",
        )
        deployment_key = st.text_input(
            "Azure OpenAI Key",
            help="Enter your Azure OpenAI key.",
            type="password",
            placeholder="e.g., sk-ab*****..",
        )
        deployment_endpoint = st.text_input(
            "API Endpoint",
            help="Enter the API endpoint for Azure OpenAI.",
            placeholder="e.g., https://api.openai.com/v1",
        )
        deployment_version = st.text_input(
            "API Version",
            help="Enter the API version for Azure OpenAI.",
            placeholder="e.g., 2024-02-15-preview",
        )
        is_streaming = st.radio(
            "Streaming",
            (True, False),
            index=1,
            format_func=lambda x: "Yes" if x else "No",
            help="Select 'Yes' if the model will be tested with output in streaming mode.",
        )
        submitted = st.form_submit_button("Add Deployment")

        if submitted:
            if (
                deployment_name
                and deployment_key
                and deployment_endpoint
                and deployment_version
            ):
                if deployment_name not in st.session_state.deployments:
                    st.session_state.deployments[deployment_name] = {
                        "key": deployment_key,
                        "endpoint": deployment_endpoint,
                        "version": deployment_version,
                        "stream": is_streaming,
                    }
                    st.success(f"Deployment '{deployment_name}' added successfully.")
                else:
                    st.error(
                        f"A deployment with the name '{deployment_name}' already exists."
                    )
            else:
                st.error("Please fill in all fields.")


def display_deployments() -> None:
    """
    Display and manage existing Azure OpenAI deployments.
    """
    if "deployments" in st.session_state:
        st.markdown("#### Loaded AOAI Deployments")
        for deployment_name, deployment in st.session_state.deployments.items():
            with st.expander(deployment_name):
                updated_name = st.text_input(
                    "Name", value=deployment_name, key=f"name_{deployment_name}"
                )
                updated_key = st.text_input(
                    "Key",
                    value=deployment.get("key", ""),
                    type="password",
                    key=f"key_{deployment_name}",
                )
                updated_endpoint = st.text_input(
                    "Endpoint",
                    value=deployment.get("endpoint", ""),
                    key=f"endpoint_{deployment_name}",
                )
                updated_version = st.text_input(
                    "Version",
                    value=deployment.get("version", ""),
                    key=f"version_{deployment_name}",
                )
                updated_stream = st.radio(
                    "Streaming",
                    (True, False),
                    format_func=lambda x: "Yes" if x else "No",
                    index=0 if deployment.get("stream", False) else 1,
                    key=f"stream_{deployment_name}",
                    help="Select 'Yes' if the model will be tested with output in streaming mode.",
                )

                if st.button("Update Deployment", key=f"update_{deployment_name}"):
                    if updated_name != deployment_name:
                        st.session_state.deployments.pop(deployment_name)
                        st.session_state.deployments[updated_name] = {
                            "key": updated_key,
                            "endpoint": updated_endpoint,
                            "version": updated_version,
                            "stream": updated_stream,
                        }
                    else:
                        st.session_state.deployments[deployment_name] = {
                            "key": updated_key,
                            "endpoint": updated_endpoint,
                            "version": updated_version,
                            "stream": updated_stream,
                        }
                    st.rerun()

                if st.button("Remove Deployment", key=f"remove_{deployment_name}"):
                    del st.session_state.deployments[deployment_name]
                    st.rerun()
    else:
        st.error("No deployments found. Please add a deployment in the sidebar.")


# Function to get the task list for the selected benchmark
def run_test_for_deployment(deployment_config: Dict, test: str = None):

    if test == "mmlu":
        obj = MMLU(
            deployment_config=deployment_config,
            sample_size=mmlu_subsample / 100,
            log_level="INFO",
            categories=mmlu_categories,
        )
        data = obj.load_data(dataset="cais/mmlu", subset="all", split="test")
        data = obj.transform_data(df=data)
        return obj.test(data)
    if test == "medpub":
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
        return obj.test(data)
    if test == "truthfulqa":
        obj = TruthfulQA(
            deployment_config=deployment_config,
            sample_size=truthful_subsample / 100,
            log_level="ERROR",
        )
        data = obj.load_data(
            dataset="truthful_qa", subset="multiple_choice", split="validation"
        )
        data = obj.transform_data(df=data)
        return obj.test(data)
    if test == "custom":
        obj = CustomEval(
            deployment_config=deployment_config,
            metrics_list=custom_metrics,
            sample_size=custom_subsample / 100,
            log_level="ERROR",
        )
        data = obj.transform_data(df=custom_df)
        return obj.test(data)


    return None


# Define an asynchronous function to run benchmark tests and log progress
def run_benchmark_tests():

    deployment_configs = []
    for deployment_name, deployment in st.session_state.deployments.items():
        deployment_config = {
            "key": deployment.get("key"),
            "endpoint": deployment.get("endpoint"),
            "model": deployment_name,
            "version": deployment.get("version"),
        }
        deployment_configs.append(deployment_config)
    
    results = []
    if mmlu_select:
        pool = Pool(processes=cpu_count() - 1)
        mmlu_stats = pool.map(partial(run_test_for_deployment, test="mmlu"), deployment_configs)
        mmlu_results = pd.concat(mmlu_stats)
        logger.debug(f"MMLU results: {mmlu_results}")

        batch_c.markdown("#### MMLU Results")
        batch_c.write(f"Subsample: {mmlu_subsample}% of each category")
        batch_c.write(f"Categories: {str(mmlu_categories)}")
        batch_c.dataframe(mmlu_results.drop("test", axis=1), hide_index=True)
        results.append(mmlu_results)

    if medpub_select:
        pool = Pool(processes=cpu_count() - 1)
        medpub_stats = pool.map(partial(run_test_for_deployment, test="medpub"), deployment_configs)
        medpub_results = pd.concat(medpub_stats)
        logger.debug(f"MedPub results: {medpub_results}")

        batch_c.markdown("#### MedPub QA Results")
        batch_c.write(f"Subsample: {medpub_subsample}%")
        batch_c.dataframe(medpub_results.drop("test", axis=1), hide_index=True)
        results.append(medpub_results)

    if truthful_select:
        pool = Pool(processes=cpu_count() - 1)
        truthful_stats = pool.map(partial(run_test_for_deployment, test="truthfulqa"), deployment_configs)
        truthful_results = pd.concat(truthful_stats)
        logger.debug(f"Truthful QA results: {truthful_results}")

        batch_c.markdown("#### Truthful QA Results")
        batch_c.write(f"Subsample: {truthful_subsample}%")
        batch_c.dataframe(truthful_results.drop("test", axis=1), hide_index=True)
        results.append(truthful_results)
        
    if custom_select:
        pool = Pool(processes=cpu_count() - 1)
        custom_stats = pool.map(partial(run_test_for_deployment, test="custom"), deployment_configs)
        custom_results = pd.concat(custom_stats)

        batch_c.markdown("#### Custom Evaluation Results")
        batch_c.write(
            f"Sample Size: {int((custom_subsample/100)*custom_df.shape[0])} ({custom_subsample}% of {custom_df.shape[0]} samples)"
        )
        batch_c.dataframe(custom_results, hide_index=True)
        results.append(custom_results)

    results_df = pd.concat(results)

    results_c.markdown("## Benchmark Results")
    fig = px.bar(
        results_df,
        x="overall_score",
        y="test",
        color="deployment",
        barmode="group",
        orientation="h",
    )
    results_c.plotly_chart(fig)
    top_bar.success("Benchmark tests completed successfully! 🎉")

    return


# Load environment variables
dotenv.load_dotenv(".env")

# Set up logger
logger = get_logger()


st.set_page_config(
    page_title="Quality Metrics AI Assistant",
    page_icon="🎯",
)

# Check if environment variables have been loaded
if not st.session_state.get("env_vars_loaded", False):
    env_vars = {
        "AZURE_OPENAI_KEY": os.getenv("AZURE_OPENAI_KEY"),
        "AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID": os.getenv(
            "AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID"
        ),
        "AZURE_OPENAI_API_ENDPOINT": os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
    }
    initialize_session_state(env_vars)

# initialize metrics list
metrics_list = ["Accuracy", "Answer Similarity"]
context_metrics_list = ["Context Similarity"]

# Main layout for initial submission

top_bar = st.empty()
results_c = st.container()
batch_c = st.container()

# Sidebar to setup model configuration settings
with st.sidebar:
    st.markdown("## 🤖 Deployment Center ")

    load_default_deployment()

    with st.expander("Add Your MaaS Deployment", expanded=False):
        operation = st.selectbox(
            "Choose Model Family:",
            ("AOAI", "Other"),
            index=0,
            help="Select the benchmark you want to perform to evaluate AI model performance.",
            placeholder="Select a Benchmark",
        )
        if operation == "AOAI":
            add_deployment_aoai_form()
        else:
            st.info("Other deployment options will be available soon.")

    display_deployments()

    st.markdown("---")

    st.markdown("## 🎯 Benchmark Configuration")

    st.markdown("Select the benchmark(s) you'd like to run:")
    mmlu_select = st.checkbox("MMLU")
    medpub_select = st.checkbox("MedPub QA")
    truthful_select = st.checkbox("Truthful QA")
    custom_select = st.checkbox("Custom Evaluation")

    if mmlu_select:
        st.write("**MMLU Benchmark Settings**")

        # Sample to categories
        mmlu_categories = st.multiselect(
            "Select MMLU subcategories to run",
            ["STEM", "Medical", "Business", "Social Sciences", "Humanities", "Other"],
            help="Select subcategories of the MMLU benchmark you'd like to run.",
        )

        # Subsample
        mmlu_subsample = st.slider(
            "Select MMLU benchmark subsample for each selected category %. (14,402 total samples)",
            min_value=0,
            max_value=100,
        )

    if medpub_select:
        st.write("**MedPub QA Benchmark Settings**")
        medpub_subsample = st.slider(
            "Select MedPub QA benchmark subsample %. (1,000 total samples)",
            min_value=0,
            max_value=100,
        )

    if truthful_select:
        st.write("**Truthful QA Benchmark Settings**")
        truthful_subsample = st.slider(
            "Select Truthful QA benchmark subsample %. (814 total samples)",
            min_value=0,
            max_value=100,
        )

    if custom_select:
        st.write("**Custom Benchmark Settings**")
        uploaded_file = st.file_uploader(
            "Upload CSV data",
            type=["csv"],
            help="Upload a CSV file with custom data for evaluation. CSV columns should be 'prompt', 'ground_truth', and 'context'. Context is optional",
        )
        if uploaded_file is not None:
            # To read file as df:
            custom_df = pd.read_csv(uploaded_file)
            cols = custom_df.columns.tolist()
            cols.append("None")

            prompt_col = st.selectbox(
                label="Select 'prompt' column", options=cols, index=cols.index("None")
            )
            ground_truth_col = st.selectbox(
                label="Select 'ground_truth' column",
                options=cols,
                index=cols.index("None"),
            )
            context_col = st.selectbox(
                label="Select 'context' column (optional)",
                options=cols,
                index=cols.index("None"),
                help="Select the context column if available. Otherwise leave as 'None'",
            )

            custom_df.rename(
                columns={prompt_col: "prompt", ground_truth_col: "ground_truth"},
                inplace=True,
            )

            if context_col != "None":
                custom_df.rename(columns={context_col: "context"}, inplace=True)
                metrics_list = metrics_list + context_metrics_list

            custom_subsample = st.slider(
                f"Select Custom benchmark subsample %. {custom_df.shape[0]} rows found",
                min_value=0,
                max_value=100,
            )
            custom_metrics = st.multiselect(
                label="Select metrics:",
                options=metrics_list,
                help="Select metrics for your custom evaluation.",
            )

    run_benchmark = st.button("Run Benchmark 🚀")

# Button to start the benchmark tests
if run_benchmark:
    with st.spinner(
        "Running benchmark tests. Outputs will appear as benchmarks complete. This may take a while..."
    ):
        top_bar.warning(
            "Warning: Editing sidebar while benchmark is running will kill the job."
        )
        run_benchmark_tests()

else:
    top_bar.info("👈 Please configure the benchmark settings to begin.")


# Footer Section

st.markdown("#### 📚 Resources and Information")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Benchmark Guide",
        "How to Add New Deployments",
        "Motivation?",
        "Learn More About Public Benchmarks",
        "Learn More About Metrics for Custom Eval",
        "What Else Can I do in this App?",
    ]
)

# Benchmark Guide
with tab1:
    st.markdown(
        """
        Ready to test the quality of your LLMs? Our benchmarking tool makes it easy! 📊✨

        Here's how it works:
        1. **Select your model settings**: Choose the tests to run, the models to evaluate, and other input parameters in the side bar.
        2. **Run the benchmark**: Hit the 'Run Benchmark' to run your selected evaluations. Wait for tests to complete.
        3. **Review the results**: Once the benchmark is complete, view detailed results on this page.

        Let's get started and optimize your LLM experience!
        """
    )

# How to Add New Deployments
with tab2:
    st.markdown(
        """
        Adding new deployments allows you to compare performance across multiple Azure OpenAI deployments in different regions. Here's a step-by-step guide on how to add and manage your deployments:

        ### Step 1: Enable Multi-Deployment
        - Check the **Add New Deployment** box at the top of the sidebar. This enables the option to add multiple deployments.

        ### Step 2: Add Deployment Details
        - Fill in the form with the following details:
            - **Azure OpenAI Key**: Your Azure OpenAI key. This is sensitive information, so it's treated as a password.
            - **API Endpoint**: The endpoint URL for Azure OpenAI.
            - **API Version**: The version of the Azure OpenAI API you're using.
            - **Chat Model Name Deployment ID**: The specific ID for your chat model deployment.
            - **Streaming**: Choose 'Yes' if you want the model to output in streaming mode.
        - Click **Add Deployment** to save your deployment to the session state.

        ### Step 3: View and Manage Deployments
        - Once added, your deployments are listed under **Loaded AOAI Deployments**.
        - Click on a deployment to expand and view its details.
        - You can update any of the deployment details here and click **Update Deployment** to save changes.

        ### How Deployments are Managed
        - Deployments are stored in the Streamlit `session_state`, allowing them to persist across page reloads and be accessible across different pages of the app.
        - You can add multiple deployments and manage them individually from the sidebar.
        - This flexibility allows you to easily compare the performance of different deployments and make adjustments as needed.

        ### Updating Deployments Across Pages
        - Since deployments are stored in the `session_state`, any updates made to a deployment from one page are reflected across the entire app.
        - This means you can seamlessly switch between different deployments or update their configurations without losing context.

        Follow these steps to efficiently manage your Azure OpenAI deployments and leverage the power of multi-deployment benchmarking.
        """,
        unsafe_allow_html=True,
    )

# Motivation?
with tab3:
    st.markdown(
        """
        Public benchmarks are often used to assess foundation model performance across a wide variety of tasks. 
        However, the fine print of many of these test reveals inconsistent prompting methodology which leads to confusing and unreliable results.   
        
        The goal of this repository is to provide a **transparent**, **flexible**, and **standadized** method to repeatably compare different foundation models or model versions. 
        This repository is designed to be executed to uniquely compare _my_ existing model to _my_ challenger model(s) as opposed to relying on public benchmarks executed on a model instantiation managed by some other entity.
        """
    )

# Learn More About Public Benchmarks
with tab4:
    st.markdown(
        """
        **MMLU**  
        
        This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge. The test spans subjects in the humanities, social sciences, hard sciences, and other areas that are important for some people to learn. This covers 57 tasks including elementary mathematics, US history, computer science, law, and more. To attain high accuracy on this test, models must possess extensive world knowledge and problem solving ability.  
        
        We have grouped some of the tasks into broader categories for easier targeted execution. These categories are: STEM, Medical, Business, Social Sciences, Humanities, and Other.
        
        [Paper](https://arxiv.org/pdf/2009.03300) | [HuggingFace Dataset](https://huggingface.co/datasets/cais/mmlu)
        
        **Truthful QA**
        
        TruthfulQA is a benchmark to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance, and politics. Questions are crafted so that some humans would answer falsely due to a false belief or misconception. To perform well, models must avoid generating false answers learned from imitating human texts.  
        
        [Paper](https://arxiv.org/pdf/2109.07958) | [HuggingFace Dataset](https://huggingface.co/datasets/truthfulqa/truthful_qa) | [GitHub](https://github.com/sylinrl/TruthfulQA)
            
        **PubMedQA**
        
        The task of PubMedQA is to answer research questions with yes/no/maybe _(e.g.: Do preoperative statins reduce atrial fibrillation after coronary artery bypass grafting?)_ using the corresponding abstracts. PubMedQA has 1k expert labeled instances. 
        
        [Paper](https://arxiv.org/pdf/1909.06146`) | [HuggingFace Dataset](https://huggingface.co/datasets/qiaojin/PubMedQA) | [Website](https://pubmedqa.github.io/) | [GitHub](https://github.com/pubmedqa/pubmedqa)
        """
    )

# Learn More About Metrics for Custom Eval
with tab5:
    st.markdown(
        """
        **Accuracy**: Number of correct predictions divided by the total number of predictions. Model outputs must be exact matches to ground truth.
          
        **Answer Similarity**: The similarity between the generated answer and the ground truth answer. This metric is calculated using the Sentence Transformers library, which provides a pre-trained model for computing sentence embeddings and calculating the cosine similarity.
          
        **Context Similarity**: The similarity between the generated answer and the context. This metric is calculated using the Sentence Transformers library, which provides a pre-trained model for computing sentence embeddings and calculating the cosine similarity.
        """
    )

# What Else Can I do in this App?
with tab6:
    st.markdown(
        """
        Dive into the capabilities of our application:

        - **Multi-Region Latency Benchmark**: Test the response time of various models across different regions. This feature helps you identify the fastest model for your needs, ensuring efficient performance no matter where you are.
        - **Throughput Test by Model**: Evaluate how many requests a model can handle over a set period. This is crucial for understanding a model's capacity and ensuring it can handle your workload without slowing down.
        - **Quality Benchmarks**: Run quality tests on your models to assess their performance in a way that YOU control.

        Our tool is designed to give you a comprehensive understanding of model performance, helping you make informed decisions. To begin, simply select an option from the sidebar. Let's optimize your AI model selection together! 👍
        """
    )

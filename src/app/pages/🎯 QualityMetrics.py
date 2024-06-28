import asyncio
import os
import pandas as pd
import plotly.express as px

import dotenv
import streamlit as st

from src.aoai.azure_openai import AzureOpenAIManager
from src.quality.evals import MMLU, TruthfulQA, PubMedQA, CustomEval
from utils.ml_logging import get_logger
from typing import Dict, Any, Optional


def initialize_session_state(defaults: Dict[str, Any]) -> None:
    """
    Initialize Streamlit session state with default values if not already set.

    Args:
        defaults (Dict[str, Any]): Dictionary of default values.
    """
    for var, value in defaults.items():
        if var not in st.session_state:
            st.session_state[var] = value

def load_default_deployment(name: Optional[str] = None, 
                            key: Optional[str] = None, 
                            endpoint: Optional[str] = None, 
                            version: Optional[str] = None) -> None:
    """
    Load default deployment settings, optionally from provided parameters.
    Ensures that a deployment with the same name does not already exist.
    """
    # Ensure deployments is a dictionary
    if 'deployments' not in st.session_state or not isinstance(st.session_state.deployments, dict):
        st.session_state.deployments = {}

    # Check if the deployment name already exists
    deployment_name = name if name else os.getenv("AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID")
    if deployment_name in st.session_state.deployments:
        return  # Exit the function if deployment already exists

    default_deployment = {
        "name": deployment_name,
        "key": key if key else os.getenv("AZURE_OPENAI_KEY"),
        "endpoint": endpoint if endpoint else os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        "version": version if version else os.getenv("AZURE_OPENAI_API_VERSION"),
        "stream": False
    }

    if all(value is not None for value in default_deployment.values() if value != False):
        st.session_state.deployments[default_deployment["name"]] = default_deployment
    else:
        st.error("Default deployment settings are missing.")

def add_deployment_form() -> None:
    """
    Render the form to add a new Azure OpenAI deployment.
    """
    with st.form("add_deployment_form"):
        deployment_name = st.text_input(
            "Deployment id",
            help="Enter the deployment ID for Azure OpenAI.",
            placeholder="e.g., chat-gpt-1234abcd"
        )
        deployment_key = st.text_input(
            "Azure OpenAI Key",
            help="Enter your Azure OpenAI key.",
            type="password",
            placeholder="e.g., sk-ab*****.."
        )
        deployment_endpoint = st.text_input(
            "API Endpoint",
            help="Enter the API endpoint for Azure OpenAI.",
            placeholder="e.g., https://api.openai.com/v1"
        )
        deployment_version = st.text_input(
            "API Version",
            help="Enter the API version for Azure OpenAI.",
            placeholder="e.g., 2024-02-15-preview"
        )
        is_streaming = st.radio(
            "Streaming",
            (True, False),
            index=1,
            format_func=lambda x: "Yes" if x else "No",
            help="Select 'Yes' if the model will be tested with output in streaming mode."
        )
        submitted = st.form_submit_button("Add Deployment")
        
        if submitted:
            if deployment_name and deployment_key and deployment_endpoint and deployment_version:
                if deployment_name not in st.session_state.deployments:
                    st.session_state.deployments[deployment_name] = {
                        "key": deployment_key,
                        "endpoint": deployment_endpoint,
                        "version": deployment_version,
                        "stream": is_streaming
                    }
                    st.success(f"Deployment '{deployment_name}' added successfully.")
                else:
                    st.error(f"A deployment with the name '{deployment_name}' already exists.")
            else:
                st.error("Please fill in all fields.")

def display_deployments() -> None:
    """
    Display and manage existing Azure OpenAI deployments.
    """
    if 'deployments' in st.session_state:
        st.markdown("#### Loaded AOAI Deployments")
        for deployment_name, deployment in st.session_state.deployments.items():
            with st.expander(deployment_name):
                updated_name = st.text_input(f"Name", value=deployment_name, key=f"name_{deployment_name}")
                updated_key = st.text_input(f"Key", value=deployment.get('key', ''), type="password", key=f"key_{deployment_name}")
                updated_endpoint = st.text_input(f"Endpoint", value=deployment.get('endpoint', ''), key=f"endpoint_{deployment_name}")
                updated_version = st.text_input(f"Version", value=deployment.get('version', ''), key=f"version_{deployment_name}")
                updated_stream = st.radio(
                    "Streaming",
                    (True, False),
                    format_func=lambda x: "Yes" if x else "No",
                    index=0 if deployment.get('stream', False) else 1,
                    key=f"stream_{deployment_name}",
                    help="Select 'Yes' if the model will be tested with output in streaming mode."
                )

                if st.button(f"Update Deployment", key=f"update_{deployment_name}"):
                    if updated_name != deployment_name:
                        st.session_state.deployments.pop(deployment_name)
                        st.session_state.deployments[updated_name] = {
                            "key": updated_key,
                            "endpoint": updated_endpoint,
                            "version": updated_version,
                            "stream": updated_stream
                        }
                    else:
                        st.session_state.deployments[deployment_name] = {
                            "key": updated_key,
                            "endpoint": updated_endpoint,
                            "version": updated_version,
                            "stream": updated_stream
                        }
                    st.experimental_rerun()

                if st.button(f"Remove Deployment", key=f"remove_{deployment_name}"):
                    del st.session_state.deployments[deployment_name]
                    st.experimental_rerun()
    else:
        st.error("No deployments found. Please add a deployment in the sidebar.")

# Function to get the task list for the selected benchmark
def get_task_list(test:str=None):
    objects = []

    for deployment_name, deployment in st.session_state.deployments.items():
        deployment_config = {
            "key": deployment.get('key'),
            "endpoint": deployment.get('endpoint'),
            "model": deployment_name,
            "version": deployment.get('version'),
        }
        if test == "mmlu":
            obj = MMLU(
                deployment_config=deployment_config,
                sample_size=mmlu_subsample/100,
                log_level="ERROR",
                categories=mmlu_categories
            )
        if test == "medpub":
            obj = PubMedQA(
                deployment_config=deployment_config,
                sample_size=medpub_subsample/100,
                log_level="ERROR"
            )
        if test == "truthfulqa":
            obj = TruthfulQA(
                deployment_config=deployment_config,
                sample_size=truthful_subsample/100,
                log_level="ERROR"
            )

        if test == "custom":
            obj = CustomEval(
                deployment_config=deployment_config,
                custom_data=custom_df,
                metrics_list=custom_metrics,
                sample_size=custom_subsample/100,
                log_level="ERROR"
            )

        objects.append(obj)
    
    tasks = [asyncio.create_task(obj.test()) for obj in objects]
    return tasks

# Define an asynchronous function to run benchmark tests and log progress
async def run_benchmark_tests():
    try:
        results = []

        if mmlu_select:
            
            mmlu_tasks = get_task_list(test="mmlu")
            mmlu_stats = await asyncio.gather(*mmlu_tasks)
            mmlu_results = pd.concat(mmlu_stats)

            batch_c.markdown("#### MMLU Results")
            batch_c.write(f"Subsample: {mmlu_subsample}% of each category")
            batch_c.write(f"Categories: {str(mmlu_categories)}")
            batch_c.dataframe(mmlu_results.drop('test', axis = 1), hide_index=True)
            results.append(mmlu_results)
        
        if medpub_select:
            logger.info("Running MedPub QA benchmark")
            medpub_tasks = get_task_list(test="medpub")
            medpub_stats = await asyncio.gather(*medpub_tasks)
            medpub_results = pd.concat(medpub_stats)

            batch_c.markdown("#### MedPub QA Results")
            batch_c.write(f"Sample Size: {int((medpub_subsample/100)*1000)} ({medpub_subsample}% of 1,000 samples)")
            batch_c.dataframe(medpub_results.drop('test', axis = 1), hide_index=True)
            results.append(medpub_results)
        
        if truthful_select:
            logger.info("Running Truthful QA benchmark")
            truthful_tasks = get_task_list(test="truthfulqa")
            truthful_stats = await asyncio.gather(*truthful_tasks)
            truthful_results = pd.concat(truthful_stats)

            batch_c.markdown("#### Truthful QA Results")
            batch_c.write(f"Sample Size: {int((truthful_subsample/100)*814)} ({truthful_subsample}% of 814 samples)")
            batch_c.dataframe(truthful_results.drop('test', axis = 1), hide_index=True)
            results.append(truthful_results)

        if custom_select:
            logger.info("Running Custom Evaluation")
            custom_tasks = get_task_list(test="custom")
            custom_stats = await asyncio.gather(*custom_tasks)
            custom_results = pd.concat(custom_stats)

            batch_c.markdown("#### Custom Evaluation Results")
            batch_c.write(f"Sample Size: {int((custom_subsample/100)*custom_df.shape[0])} ({custom_subsample}% of {custom_df.shape[0]} samples)")
            batch_c.dataframe(custom_results, hide_index=True)
            results.append(custom_results)

        results_df = pd.concat(results)

        results_c.markdown("## Benchmark Results")
        fig = px.bar(results_df, x='overall_score', y='test', color='deployment', barmode='group', orientation='h')
        results_c.plotly_chart(fig)
        top_bar.success("Benchmark tests completed successfully! 🎉")
        # results_c.dataframe(results_df, hide_index=True)

    except Exception as e:
        top_bar.error(f"An error occurred: {str(e)}")

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
metrics_list = ['Accuracy', 'Answer Similarity']
context_metrics_list = ['Context Similarity']

# Main layout for initial submission

top_bar = st.empty()
results_c = st.container()
batch_c = st.container()

# Sidebar to setup model configuration settings
with st.sidebar:
    st.markdown("## ⚙️ Configuration Settings")
        
    with st.expander("📘 How to Add New Deployments", expanded=False):
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
            unsafe_allow_html=True
        )
    
    enable_multi_deployment = st.checkbox("Load New Deployment", help="Check this box to compare performance across multiple deployments in different regions.")
    
    if enable_multi_deployment:
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

    st.markdown("## 🎯 Benchmark Settings")

    st.markdown("Select the benchmark(s) you'd like to run:")
    mmlu_select = st.checkbox("MMLU")
    medpub_select = st.checkbox("MedPub QA")
    truthful_select = st.checkbox("Truthful QA")
    custom_select = st.checkbox("Custom Evaluation")
    
    if mmlu_select:
        st.write("**MMLU Benchmark Settings**")

        # Sample to categories
        mmlu_categories = st.multiselect("Select MMLU subcategories to run",
                                        ['STEM', 'Medical', 'Business', 'Social Sciences', 'Humanities', 'Other'],
                                        help="Select subcategories of the MMLU benchmark you'd like to run.")

        # Subsample
        mmlu_subsample = st.slider('Select MMLU benchmark subsample for each selected category %. (14,402 total samples)', min_value=0, max_value=100)

    if medpub_select:
        st.write("**MedPub QA Benchmark Settings**")
        medpub_subsample = st.slider('Select MedPub QA benchmark subsample %. (1,000 total samples)', min_value=0, max_value=100)

    if truthful_select:
        st.write("**Truthful QA Benchmark Settings**")
        truthful_subsample = st.slider('Select Truthful QA benchmark subsample %. (814 total samples)', min_value=0, max_value=100)

    if custom_select:
        st.write("**Custom Benchmark Settings**")
        uploaded_file = st.file_uploader("Upload CSV data", type=['csv'], help="Upload a CSV file with custom data for evaluation. CSV columns should be 'prompt', 'ground_truth', and 'context'. Context is optional")
        if uploaded_file is not None:
            # To read file as df:
            custom_df = pd.read_csv(uploaded_file)
            cols = custom_df.columns.tolist()
            cols.append("None")

            prompt_col = st.selectbox(label="Select 'prompt' column", options=cols, index=cols.index("None"))
            ground_truth_col = st.selectbox(label="Select 'ground_truth' column", options=cols, index=cols.index("None"))
            context_col = st.selectbox(label="Select 'context' column (optional)", options=cols, index=cols.index("None"), help="Select the context column if available. Otherwise leave as 'None'")
            
            custom_df.rename(columns={prompt_col: 'prompt', ground_truth_col: 'ground_truth'}, inplace=True)

            if context_col != "None":
                custom_df.rename(columns={context_col: 'context'}, inplace=True)
                metrics_list = metrics_list + context_metrics_list
            
            custom_subsample = st.slider(f'Select Custom benchmark subsample %. {custom_df.shape[0]} rows found', min_value=0, max_value=100)
            custom_metrics = st.multiselect(label="Select metrics:",options = metrics_list, help = "Select metrics for your custom evaluation.")

    run_benchmark = st.button("Run Benchmark 🚀")

# Button to start the benchmark tests
if run_benchmark:
    with st.spinner("Running benchmark tests. Outputs will appear as benchmarks complete. This may take a while..."):
        top_bar.warning("Warning: Editing sidebar while benchmark is running will kill the job.")
        asyncio.run(run_benchmark_tests())

else:
    top_bar.info("👈 Please configure the benchmark settings to begin.")


# Footer Section

st.markdown("#### 📚 Resources and Information:")

with st.expander("Benchmark Guide 📊", expanded=False):
        st.markdown(
            """
            Ready to test the  quality of your LLMs? Our benchmarking tool makes it easy! 📊✨

            Here's how it works:
            1. **Select your model settings**: Choose the tests to run, the models to evaluate, and other input parameters in the side bar.
            2. **Run the benchmark**: Hit the 'Run Benchmark' to run your selected evaluations. Wait for tests to complete.
            3. **Review the results**: Once the benchmark is complete, view detailed results on this page.

            Let's get started and optimize your LLM experience!
            """
        )

with st.expander("Motivation? ⚡", expanded=False):
    st.markdown(
        """
        Public benchmarks are often used to assess foundation model performance across a wide variety of tasks. 
        However, the fine print of many of these test reveals inconsistent prompting methodology which leads to confusing and unreliable results.   
        
        The goal of this repository is to provide a **transparent**, **flexible**, and **standadized** method to repeatably compare different foundation models or model versions. 
        This repsoitory is designed to be executed to uniquely compare _my_ existing model to _my_ challenger model(s) as opposed to relying on public benchmarks executed on a model instantiation managed by some other entity.
        """
    )

with st.expander("Learn More About Public Benchmarks 📖", expanded=False):
    st.markdown(
        """
            **MMLU**  
            
            This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge. The test spans subjects in the humanities, social sciences, hard sciences, and other areas that are important for some people to learn. This covers 57 tasks including elementary mathematics, US history, computer science, law, and more. To attain high accuracy on this test, models must possess extensive world knowledge and problem solving ability.  
            
            We have grouped some of the taks into borader categories for easier targeted execution. These categories are: STEM, Medical, Business, Social Sciences, Humanities, and Other.
            
            [Paper](https://arxiv.org/pdf/2009.03300) | [HuggingFace Dataset](https://huggingface.co/datasets/cais/mmlu)
            
            **Truthful QA**
            
            TruthfulQA is a benchmark to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance and politics. Questions are crafted so that some humans would answer falsely due to a false belief or misconception. To perform well, models must avoid generating false answers learned from imitating human texts.  
            
            [Paper](https://arxiv.org/pdf/2109.07958) | [HuggingFace Dataset](https://huggingface.co/datasets/truthfulqa/truthful_qa) | [GitHub](https://github.com/sylinrl/TruthfulQA)
                
            **PubMedQA**
            
            The task of PubMedQA is to answer research questions with yes/no/maybe _(e.g.: Do preoperative statins reduce atrial fibrillation after coronary artery bypass grafting?)_ using the corresponding abstracts. PubMedQA has 1k expert labeled instances. 
            
            [Paper](https://arxiv.org/pdf/1909.06146`) | [HuggingFace Dataset](https://huggingface.co/datasets/qiaojin/PubMedQA) | [Website](https://pubmedqa.github.io/) | [GitHub](https://github.com/pubmedqa/pubmedqa)

        """
    )

with st.expander("Learn More About Metrics for Custom Eval 🧮", expanded=False):
    st.markdown(
        """
        **Accuracy**: Number of correct predictions divided by the total number of predictions. Model outputs must be exact matches to ground truth.
        **Answer Similarity**: The similarity between the generated answer and the ground truth answer. This metric is calculated using the Sentence Transformers library, which provides a pre-trained model for computing sentence embeddings and calculating the cosine similarity.
        **Context Similarity**: The similarity between the generated answer and the context. This metric is calculated using the Sentence Transformers library, which provides a pre-trained model for computing sentence embeddings and calculating the cosine similarity.
        """
    )

with st.expander("What Else Can I do in this App? 🤔", expanded=False):
    st.markdown(
        """
        Dive into the capabilities of our application:

        - **Multi-Region Latency Benchmark**: Test the response time of various models across different regions. This feature helps you identify the fastest model for your needs, ensuring efficient performance no matter where you are.
        - **Throughput Test by Model**: Evaluate how many requests a model can handle over a set period. This is crucial for understanding a model's capacity and ensuring it can handle your workload without slowing down.
        - **Quality Benchmarks**: Run quality tests on your models to assess their performance in a way that YOU control.

        Our tool is designed to give you a comprehensive understanding of model performance, helping you make informed decisions. To begin, simply select an option from the sidebar. Let's optimize your AI model selection together! 👍
        """
    )

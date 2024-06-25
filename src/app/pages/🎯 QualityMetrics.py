import asyncio
import os
import pandas as pd
import plotly.express as px

import dotenv
import streamlit as st

from src.aoai.azure_openai import AzureOpenAIManager
from src.quality.evals import MMLU, TruthfulQA, PubMedQA, CustomEval
from utils.ml_logging import get_logger


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
    st.session_state.update(
        {
            "azure_openai_manager": None,
            "document_intelligence_manager": None,
            "blob_data_extractor_manager": None,
            "client_non_streaming": None,  # Initialize the key for the non-streaming client
        }
    )

    env_vars = {
        "AZURE_OPENAI_KEY": os.getenv("AZURE_OPENAI_KEY"),
        "AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID": os.getenv(
            "AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID"
        ),
        "AZURE_OPENAI_API_ENDPOINT": os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
    }

    st.session_state.update(env_vars)

    # Initialize the AzureOpenAIManager
    st.session_state["azure_openai_manager"] = AzureOpenAIManager(
        api_key=st.session_state["AZURE_OPENAI_KEY"],
        azure_endpoint=st.session_state["AZURE_OPENAI_API_ENDPOINT"],
        api_version=st.session_state["AZURE_OPENAI_API_VERSION"],
        chat_model_name=st.session_state["AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID"],
    )

# Main layout for initial submission

top_bar = st.empty()
top_bar.markdown("""
                 **Get Started with Quality Metrics AI Assistant**: 

                 Configure variables in the sidebar and run the benchmark tests to evaluate the quality of your LLMs.🎯

                 """)

with st.expander("Learn More About Quality Benchmarks 📖", expanded=False):
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

# Sidebar layout for initial submission
with st.sidebar:
    st.markdown("### Configure Benchmark Variables:")
    deployment_names = st.text_area(
        "Enter Deployment Names",
        help="Enter the deployment names you want to benchmark as a comma seperated list.",
    )

    st.write("Select the benchmark(s) you'd like to run:")
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
            df_validation_c = st.empty()
            if 'prompt' in custom_df.columns and 'ground_truth' in custom_df.columns:
                df_validation_c.success(f"Custom data uploaded successfully!")
            else:
                df_validation_c.error("Error. Could not find 'prompt' and 'ground_truth' columns in the uploaded file.")
            
            if 'context' not in custom_df.columns:
                metrics_list = ['Accuracy']
            else:
                metrics_list = ['Accuracy', 'Faithfulness']
            
            custom_subsample = st.slider(f'Select Custom benchmark subsample %. {custom_df.shape[0]} rows found', min_value=0, max_value=100)
            custom_metrics = st.multiselect(label="Select metrics:",options = metrics_list, help = "Select metrics for your custom evaluation.")

    run_benchmark = st.button("Run Benchmark 🚀")
    with st.expander("Benchmark Guide 📊", expanded=False):
        st.markdown(
            """
            Ready to test the  quality of your LLMs? Our benchmarking tool makes it easy! 📊✨

            Here's how it works:
            1. **Select your model settings**: Choose the tests to run, the models to evaluate, and other input parameters.
            2. **Run the benchmark**: Hit the 'Run Benchmark' to run your selected evaluations.
            3. **Review the results**: Once the benchmark is complete, view detailed results and performance metrics.

            Let's get started and optimize your LLM experience!
            """
        )


# Function to get the task list for the selected benchmark
def get_task_list(deployment_list:list):
    objects = []

    for deployment in deployment_list:
        deployment_config = {
            "key": st.session_state["AZURE_OPENAI_KEY"],
            "endpoint": st.session_state["AZURE_OPENAI_API_ENDPOINT"],
            "model": deployment,
            "version": st.session_state["AZURE_OPENAI_API_VERSION"],
        }
        if mmlu_select:
            obj = MMLU(
                deployment_config=deployment_config,
                sample_size=mmlu_subsample/100,
                log_level="ERROR",
                categories=mmlu_categories
            )
        if medpub_select:
            obj = PubMedQA(
                deployment_config=deployment_config,
                sample_size=medpub_subsample/100,
                log_level="ERROR"
            )
        if truthful_select:
            obj = TruthfulQA(
                deployment_config=deployment_config,
                sample_size=truthful_subsample/100,
                log_level="ERROR"
            )

        if custom_select:
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
        deployment_names_list = [name.strip() for name in deployment_names.split(",")]
        results = []
        c=st.container()

        if mmlu_select:
            
            mmlu_tasks = get_task_list(deployment_names_list)
            mmlu_stats = await asyncio.gather(*mmlu_tasks)
            mmlu_results = pd.concat(mmlu_stats)

            st.markdown("### MMLU Results")
            st.write("Subsample: ", f"{mmlu_subsample}% of each category")
            st.write("Categories: ", str(mmlu_categories))
            st.dataframe(mmlu_results.drop('test', axis = 1), hide_index=True)
            results.append(mmlu_results)
        
        if medpub_select:
            logger.info("Running MedPub QA benchmark")
            medpub_tasks = get_task_list(deployment_names_list)
            medpub_stats = await asyncio.gather(*medpub_tasks)
            medpub_results = pd.concat(medpub_stats)

            st.markdown("### MedPub QA Results")
            st.write("Sample Size: ", f"{int((medpub_subsample/100)*1000)} ({medpub_subsample}% of 1,000 samples)")
            st.dataframe(medpub_results.drop('test', axis = 1), hide_index=True)
            results.append(medpub_results)
        
        if truthful_select:
            logger.info("Running Truthful QA benchmark")
            truthful_tasks = get_task_list(deployment_names_list)
            truthful_stats = await asyncio.gather(*truthful_tasks)
            truthful_results = pd.concat(truthful_stats)

            st.markdown("### Truthful QA Results")
            st.write("Sample Size: ", f"{int((truthful_subsample/100)*814)} ({truthful_subsample}% of 814 samples)")
            st.dataframe(truthful_results.drop('test', axis = 1), hide_index=True)
            results.append(truthful_results)

        if custom_select:
            logger.info("Running Custom Evaluation")
            custom_tasks = get_task_list(deployment_names_list)
            custom_stats = await asyncio.gather(*custom_tasks)
            custom_results = pd.concat(custom_stats)

            st.markdown("### Custom Evaluation Results")
            st.write("Sample Size: ", f"{int((custom_subsample/100)*custom_df.shape[0])} ({custom_subsample}% of {custom_df.shape[0]} samples)")
            st.dataframe(custom_results.drop('test', axis = 1), hide_index=True)
            results.append(custom_results)

        results_df = pd.concat(results)

        c.markdown("## Benchmark Results")
        fig = px.bar(results_df, x='overall_score', y='test', color='deployment', barmode='group', orientation='h')
        c.plotly_chart(fig)
        top_bar.success("Benchmark tests completed successfully! 🎉")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


# Button to start the benchmark tests
if run_benchmark:
    with st.spinner("Running benchmark tests. Outputs will appeats as benchmarks complete. This may take a while..."):
        top_bar.warning("Warning: Editing sidebar while benchmark is running will kill the job.")
        asyncio.run(run_benchmark_tests())
    

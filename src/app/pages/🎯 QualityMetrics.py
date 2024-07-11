import os
from typing import Any, Dict, List, Tuple
import pandas as pd
import plotly.express as px
import streamlit as st
from multiprocessing import Pool, cpu_count
from functools import partial
import dotenv

# Custom module imports
from src.quality.evals import MMLU, CustomEval, PubMedQA, TruthfulQA
from utils.ml_logging import get_logger
from src.app.Home import create_benchmark_center, display_deployments, load_default_deployment

# Load environment variables
dotenv.load_dotenv(".env")

# Set up logger
logger = get_logger()

# Streamlit page configuration
st.set_page_config(
    page_title="Quality Metrics AI Assistant",
    page_icon="🎯",
)

def initialize_session_state(vars: List[str], initial_values: Dict[str, Any]) -> None:
    """
    Initialize Streamlit session state with default values if not already set.

    Args:
        vars (List[str]): List of session state variable names.
        initial_values (Dict[str, Any]): Dictionary of initial values for the session state variables.
    """
    for var in vars:
        if var not in st.session_state:
            st.session_state[var] = initial_values.get(var, None)

# Initialize session state variables and values
session_vars = ["results_quality", "benchmark_quality_settings"]
initial_values = {"results": {}, "benchmark_quality_settings": {}}

initialize_session_state(session_vars, initial_values)

# Initialize metrics list
METRICS_LIST = ["Accuracy", "Answer Similarity"]
CONTEXT_METRICS_LIST = ["Context Similarity"]

def run_test_for_deployment(deployment_config: Dict, test: str = None) -> pd.DataFrame:
    """
    Run the specified test for a given deployment configuration.

    Args:
        deployment_config (Dict): Deployment configuration dictionary.
        test (str, optional): The type of test to run. Defaults to None.

    Returns:
        pd.DataFrame: The results of the test.
    """
    test_obj = None
    if test == "mmlu":
        test_obj = MMLU(
            deployment_config=deployment_config,
            sample_size=st.session_state["benchmark_quality_settings"]["mmlu_subsample"] / 100,
            log_level="INFO",
            categories=st.session_state["benchmark_quality_settings"]["mmlu_categories"],
        )
        data = test_obj.load_data(dataset="cais/mmlu", subset="all", split="test")
    elif test == "medpub":
        test_obj = PubMedQA(
            deployment_config=deployment_config,
            sample_size=st.session_state["benchmark_quality_settings"]["medpub_subsample"] / 100,
            log_level="ERROR",
        )
        data = test_obj.load_data(
            dataset="qiaojin/PubMedQA",
            subset="pqa_labeled",
            split="train",
            flatten=True,
        )
    elif test == "truthfulqa":
        test_obj = TruthfulQA(
            deployment_config=deployment_config,
            sample_size=st.session_state["benchmark_quality_settings"]["truthful_subsample"] / 100,
            log_level="ERROR",
        )
        data = test_obj.load_data(
            dataset="truthful_qa", subset="multiple_choice", split="validation"
        )
    elif test == "custom":
        test_obj = CustomEval(
            deployment_config=deployment_config,
            metrics_list=st.session_state["benchmark_quality_settings"]["custom_metrics"],
            sample_size= st.session_state["benchmark_quality_settings"]["custom_subsample"] / 100,
            log_level="ERROR",
        )
        data = st.session_state["benchmark_quality_settings"]["custom_df"]

    if test_obj is not None:
        data = test_obj.transform_data(df=data)
        return test_obj.test(data)

    return pd.DataFrame()

def run_benchmark_tests(batch_c: st.container,
                        results_c: st.container, 
                        top_bar: st.empty,
                        mmlu_select: bool,
                        medpub_select: bool,
                        truthful_select: bool,
                        custom_select: bool) -> None:
    """
    Run benchmark tests and log progress.

    Args:
        batch_c (st.container): Streamlit container for batch results.
        results_c (st.container): Streamlit container for final results.
        top_bar (st.empty): Streamlit element for displaying top bar messages.
        mmlu_select (bool): Indicates if MMLU Benchmark is selected.
        medpub_select (bool): Indicates if MedPub QA Benchmark is selected.
        truthful_select (bool): Indicates if Truthful QA Benchmark is selected.
        custom_select (bool): Indicates if Custom Evaluation is selected.
    """
    deployment_configs = [
        {
            "key": deployment.get("key"),
            "endpoint": deployment.get("endpoint"),
            "model": deployment_name,
            "version": deployment.get("version"),
        }
        for deployment_name, deployment in st.session_state.deployments.items()
    ]

    results = []

    if mmlu_select and st.session_state["benchmark_quality_settings"].get("mmlu_categories"):
        pool = Pool(processes=cpu_count() - 1)
        mmlu_stats = pool.map(partial(run_test_for_deployment, test="mmlu"), deployment_configs)
        mmlu_results = pd.concat(mmlu_stats)
        logger.debug(f"MMLU results: {mmlu_results}")

        batch_c.markdown("#### MMLU Results")
        batch_c.write(f"Subsample: {st.session_state['benchmark_quality_settings']['mmlu_subsample']}% of each category")
        batch_c.write(f"Categories: {str(st.session_state['benchmark_quality_settings']['mmlu_categories'])}")
        batch_c.dataframe(mmlu_results.drop("test", axis=1), hide_index=True)
        results.append(mmlu_results)

    if medpub_select and st.session_state["benchmark_quality_settings"].get("medpub_subsample") is not None:
        pool = Pool(processes=cpu_count() - 1)
        medpub_stats = pool.map(partial(run_test_for_deployment, test="medpub"), deployment_configs)
        medpub_results = pd.concat(medpub_stats)
        logger.debug(f"MedPub results: {medpub_results}")

        batch_c.markdown("#### MedPub QA Results")
        batch_c.write(f"Subsample: {st.session_state['benchmark_quality_settings']['medpub_subsample']}%")
        batch_c.dataframe(medpub_results.drop("test", axis=1), hide_index=True)
        results.append(medpub_results)

    if truthful_select and st.session_state["benchmark_quality_settings"].get("truthful_subsample") is not None:
        pool = Pool(processes=cpu_count() - 1)
        truthful_stats = pool.map(partial(run_test_for_deployment, test="truthfulqa"), deployment_configs)
        truthful_results = pd.concat(truthful_stats)
        logger.debug(f"Truthful QA results: {truthful_results}")

        batch_c.markdown("#### Truthful QA Results")
        batch_c.write(f"Subsample: {st.session_state['benchmark_quality_settings']['truthful_subsample']}%")
        batch_c.dataframe(truthful_results.drop("test", axis=1), hide_index=True)
        results.append(truthful_results)
        
    if custom_select and st.session_state["benchmark_quality_settings"].get("custom_df") is not None:
        pool = Pool(processes=cpu_count() - 1)
        custom_stats = pool.map(partial(run_test_for_deployment, test="custom"), deployment_configs)
        custom_results = pd.concat(custom_stats)

        batch_c.markdown("#### Custom Evaluation Results")
        batch_c.write(
            f"Sample Size: {int((st.session_state['benchmark_quality_settings']['custom_subsample'] / 100) * st.session_state['benchmark_quality_settings']['custom_df'].shape[0])} ({st.session_state['benchmark_quality_settings']['custom_subsample']}% of {st.session_state['benchmark_quality_settings']['custom_df'].shape[0]} samples)"
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

def configure_benchmarking_settings(mmlu_select: bool, medpub_select: bool, truthful_select: bool, custom_select: bool) -> None:
    """
    Configure the benchmark settings based on user selections.

    Args:
        mmlu_select (bool): Indicates if MMLU Benchmark is selected.
        medpub_select (bool): Indicates if MedPub QA Benchmark is selected.
        truthful_select (bool): Indicates if Truthful QA Benchmark is selected.
        custom_select (bool): Indicates if Custom Evaluation is selected.
    """
    if mmlu_select:
        st.write("**MMLU Benchmark Settings**")

        # Sample to categories
        st.session_state["benchmark_quality_settings"]["mmlu_categories"] = st.multiselect(
            "Select MMLU subcategories to run",
            ["STEM", "Medical", "Business", "Social Sciences", "Humanities", "Other"],
            help="Select subcategories of the MMLU benchmark you'd like to run.",
        )

        # Subsample
        st.session_state["benchmark_quality_settings"]["mmlu_subsample"] = st.slider(
            "Select MMLU benchmark subsample for each selected category %. (14,402 total samples)",
            min_value=0,
            max_value=100,
        )

    if medpub_select:
        st.write("**MedPub QA Benchmark Settings**")
        st.session_state["benchmark_quality_settings"]["medpub_subsample"] = st.slider(
            "Select MedPub QA benchmark subsample %. (1,000 total samples)",
            min_value=0,
            max_value=100,
        )

    if truthful_select:
        st.write("**Truthful QA Benchmark Settings**")
        st.session_state["benchmark_quality_settings"]["truthful_subsample"] = st.slider(
            "Select Truthful QA benchmark subsample %. (814 total samples)",
            min_value=0,
            max_value=100,
        )

    if custom_select:
        st.write("**Custom Benchmark Settings**")
        uploaded_file = st.file_uploader(
            "Upload CSV data",
            type=["csv"],
            help="Upload a CSV file with custom data for evaluation. CSV columns should be 'prompt', 'ground_truth', and 'context'. Context is optional.",
        )
        if uploaded_file is not None:
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
                help="Select the context column if available. Otherwise leave as 'None'.",
            )

            custom_df.rename(
                columns={prompt_col: "prompt", ground_truth_col: "ground_truth"},
                inplace=True,
            )

            if context_col != "None":
                custom_df.rename(columns={context_col: "context"}, inplace=True)
                metrics_list = METRICS_LIST + CONTEXT_METRICS_LIST

            st.session_state["benchmark_quality_settings"]["custom_df"] = custom_df

            st.session_state["benchmark_quality_settings"]["custom_subsample"] = st.slider(
                f"Select Custom benchmark subsample %. {custom_df.shape[0]} rows found",
                min_value=0,
                max_value=100,
            )
            st.session_state["benchmark_quality_settings"]["custom_metrics"] = st.multiselect(
                label="Select metrics:",
                options=metrics_list,
                help="Select metrics for your custom evaluation.",
            )

def configure_sidebar() -> Tuple[bool, bool, bool, bool]:
    """
    Configure the sidebar with Benchmark Center and deployment forms.
    Returns a tuple of booleans indicating the selection state of each benchmark.
    """
    with st.sidebar:
        st.markdown("## 🤖 Deployment Center ")
        if st.session_state.get('deployments', {}) == {}:
            load_default_deployment()
        create_benchmark_center()
        display_deployments()

        st.divider()

        st.markdown("## 🎯 Benchmark Configuration")
        st.markdown("Select the benchmark(s) you'd like to run:")
        
        mmlu_select = st.checkbox("MMLU")
        medpub_select = st.checkbox("MedPub QA")
        truthful_select = st.checkbox("Truthful QA")
        custom_select = st.checkbox("Custom Evaluation")

        configure_benchmarking_settings(mmlu_select, medpub_select, truthful_select, custom_select)

    return mmlu_select, medpub_select, truthful_select, custom_select

def main():
    """
    Main function to run the Streamlit application.
    """
    top_bar = st.empty()
    results_c = st.container()
    batch_c = st.container()
    
    mmlu_select, medpub_select, truthful_select, custom_select = configure_sidebar()
   
    run_benchmark = st.sidebar.button("Run Benchmark 🚀")

    if run_benchmark:
        with st.spinner(
            "Running benchmark tests. Outputs will appear as benchmarks complete. This may take a while..."
        ):
            top_bar.warning(
                "Warning: Editing sidebar while benchmark is running will kill the job."
            )
            run_benchmark_tests(results_c, 
                                batch_c, 
                                top_bar,
                                mmlu_select,
                                medpub_select,
                                truthful_select,
                                custom_select)
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


if __name__ == "__main__":
    main()




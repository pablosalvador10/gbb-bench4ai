
import streamlit as st
from src.quality.evals import MMLU, CustomEval, PubMedQA, TruthfulQA
import pandas as pd
import asyncio
import plotly.express as px
from utils.ml_logging import get_logger
from src.app.quality.results import BenchmarkQualityResult

# Set up logger
logger = get_logger()

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
            st.session_state["results_quality"][results_quality.id] = results_quality.results

    except Exception as e:
        top_bar.error(f"An error occurred: {str(e)}")

def display_quality_results(results_container: st.container, id:str):
    """
    Display results for a selected run ID from session state.

    :param results_c: Streamlit container to display the results.
    """
  
    results_df = st.session_state["results_quality"][id]["all_results"]
    
    results_container.markdown("## Benchmark Results")
    
    if not results_df.empty:
        fig = px.bar(
            results_df,
            x="overall_score",
            y="test",
            color="deployment",
            barmode="group",
            orientation="h",
            title="Benchmark Results Overview"
        )
        fig.update_layout(
            xaxis_title="Overall Score",
            yaxis_title="Test",
            legend_title="Deployment",
            barmode='group'
        )
        results_container.plotly_chart(fig, use_container_width=True)
    else:
        results_container.info(
            "👈 Hey - you haven't fired any benchmarks yet. Please configure the benchmark settings and click 'Start Benchmark' to begin."
        )
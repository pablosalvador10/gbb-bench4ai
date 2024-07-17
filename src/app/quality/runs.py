import os
import streamlit as st
import pandas as pd
from datetime import datetime
from src.quality.evals import MMLU, CustomEval, PubMedQA, TruthfulQA
from my_utils.ml_logging import get_logger
import asyncio
import concurrent.futures
import copy
from src.app.quality.results import BenchmarkQualityResult
from src.app.managers import (create_benchmark_non_streaming_client,
                              create_benchmark_streaming_client)
import copy
from src.app.performance.results import BenchmarkPerformanceResult
# Set up logger
logger = get_logger()


# Set up logger
logger = get_logger()

top_bar = st.empty()

# Function to get the task list for the selected benchmark
def get_task_list(test: str = None):
    """
    Get the list of tasks to run for the selected benchmark.

    :param test: The name of the benchmark test to run.
    :return: A list of tasks to run.
    """
    objects = []
    settings = st.session_state.get("settings_quality", {})
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

def run_retrieval_quality_for_client(client):
    """
    Run retrieval quality evaluation for a single client.

    :param client: The evaluation client to run the retrieval quality evaluation for.
    :return: A tuple containing the deployment name and the evaluation results.
    """
    try:
        results_retrieval = client.run_retrieval_quality(data_input=st.session_state["evaluation_clients_retrieval_df"])
        deployment_name = client.model_config.azure_deployment
        return (deployment_name, results_retrieval)
    except Exception as e:
        print(f"An error occurred: {e}")
        return (client.model_config.azure_deployment, None)

def run_retrieval_quality_in_parallel():
    """
    Run retrieval quality evaluations in parallel for all clients.

    :return: A dictionary containing the evaluation results for each deployment.
    """
    clients = st.session_state.get("evaluation_clients_retrieval", [])
    logger.info(f"Running retrieval quality evaluations for {len(clients)} clients.")
    results_gpt_evals = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(run_retrieval_quality_for_client, clients))
    
    for deployment_name, result in results:
        if result is not None:
            results_gpt_evals[deployment_name] = result["metrics"]
    
    return results_gpt_evals

async def run_benchmark_tests():
    """
    Run benchmark tests and log progress. This function runs selected benchmark tests asynchronously.

    :return: None
    """
    try:
        results = []
        results_retrievals = {}
        results_rai = {}
        if "results_quality" not in st.session_state:
            st.session_state["results_quality"] = {}
        
        results_df = pd.DataFrame()
        truthful_results = pd.DataFrame()
        mmlu_results = pd.DataFrame()
        medpub_results = pd.DataFrame()
        retrieval_results = pd.DataFrame()
        rai_results = pd.DataFrame()

        settings = st.session_state.get("settings_quality", {})
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

            if "Retrieval" in settings.get("benchmark_selection", []):
                clients = st.session_state.get("evaluation_clients_retrieval", [])
                for client in clients:
                    try:
                        metrics, studio_url = client.run_chat_quality(data_input=st.session_state["settings_quality"]["evaluation_clients_retrieval_df"])
                        deployment_name = client.model_config.azure_deployment
                        # Add studio_url to the metrics dictionary
                        metrics['studio_url'] = studio_url
                        results_retrievals[deployment_name] = metrics
                        logger.info(f"Retrieval quality evaluation for {deployment_name} completed.")
                    except Exception as e:
                        logger.error(f"Failed to run retrieval quality evaluation for client {client}: {e}")
                        st.error(f"Failed to run retrieval quality evaluation for client {client}. Check logs for details.")

                data_for_df = []

                for model_name, metrics in results_retrievals.items():
                    row_data = metrics.copy()  
                    row_data['model'] = model_name 
                    data_for_df.append(row_data)

                retrieval_results = pd.DataFrame(data_for_df).set_index('model')

            if "RAI" in settings.get("benchmark_selection", []):
                clients = st.session_state.get("evaluation_clients_rai", [])
                for client in clients:
                    try:
                        metrics, studio_url = client.run_chat_content_safety(data_input=st.session_state["settings_quality"]["evaluation_clients_rai_df"])
                        deployment_name = client.model_config.azure_deployment
                        metrics['studio_url'] = studio_url
                        results_rai[deployment_name] = metrics
                        logger.info(f"RAI quality evaluation for {deployment_name} completed.")
                    except Exception as e:
                        logger.error(f"Failed to run RAI quality evaluation for client {client}: {e}")
                        st.error(f"Failed to run RAI quality evaluation for client {client}. Check logs for details.")

                data_for_df_rai = []
                
                for model_name, metrics in results_rai.items():
                    row_data = metrics.copy()  
                    row_data['model'] = model_name 
                    data_for_df_rai.append(row_data)
                
                rai_results = pd.DataFrame(data_for_df_rai).set_index('model')

            results_df = pd.concat(results)
            results_df = results_df if isinstance(results_df, pd.DataFrame) else pd.DataFrame()
            truthful_results = truthful_results if isinstance(truthful_results, pd.DataFrame) else pd.DataFrame()
            mmlu_results = mmlu_results if isinstance(mmlu_results, pd.DataFrame) else pd.DataFrame()
            medpub_results = medpub_results if isinstance(medpub_results, pd.DataFrame) else pd.DataFrame()
            results_retrievals = retrieval_results if isinstance(retrieval_results, pd.DataFrame) else pd.DataFrame()
            results_rai = rai_results if isinstance(rai_results, pd.DataFrame) else pd.DataFrame()
            results = {
                "all_results": results_df,
                "truthful_results": truthful_results,
                "mmlu_results": mmlu_results,
                "medpub_results": medpub_results,
                "retrieval_results": results_retrievals,
                "rai_results": results_rai
            }
            settings_snapshot = copy.deepcopy(settings)
            results_quality = BenchmarkQualityResult(result=results, settings=settings_snapshot)
            st.session_state["results_quality"][results_quality.id] = results_quality.to_dict()

    except Exception as e:
        top_bar.error(f"An error occurred: {str(e)}")


async def run_benchmark_quality(test_status_placeholder: st.container) -> None:
    """
    Run the benchmark tests asynchronously, with detailed configuration for each test.

    :param test_status_placeholder: Streamlit placeholder for the test status.
    """
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

    async def safe_run(client, deployment_name):
        try:
            await client.run_latency_benchmark_bulk(
                deployment_names=[deployment_name],
                max_tokens_list=st.session_state["settings"]["max_tokens_list"],
                iterations=st.session_state["settings"]["num_iterations"],
                context_tokens=st.session_state["settings"]["context_tokens"],
                temperature=st.session_state["settings"]["temperature"],
                byop=st.session_state["settings"]["prompts"],
                context=st.session_state["settings"]["context"],
                prevent_server_caching=st.session_state["settings"][
                    "prevent_server_caching"
                ],
                timeout=st.session_state["settings"]["timeout"],
                top_p=st.session_state["settings"]["top_p"],
                n=1,
                presence_penalty=st.session_state["settings"]["presence_penalty"],
                frequency_penalty=st.session_state["settings"]["frequency_penalty"],
            )
        except Exception as e:
            logger.error(
                f"An error occurred with deployment '{deployment_name}': {str(e)}",
                exc_info=True,
            )
            st.error(f"An error occurred with deployment '{deployment_name}': {str(e)}")

    logger.info(f"Total number of deployment clients: {len(deployment_clients)}")

    for client, deployment_name in deployment_clients:
        await safe_run(client, deployment_name)

    try:
        stats = [
            client.calculate_and_show_statistics() for client, _ in deployment_clients
        ]
        stats_raw = [client.results for client, _ in deployment_clients]
        st.session_state["benchmark_results"] = stats
        st.session_state["benchmark_results_raw"] = stats_raw
        settings_snapshot = copy.deepcopy(st.session_state["settings"])
        results = BenchmarkPerformanceResult(
            result=stats, settings=settings_snapshot
        )
        st.session_state["results"][results.id] = results.to_dict()
        test_status_placeholder.markdown(
            f"Benchmark <span style='color: grey;'>{results.id}</span> Completed",
            unsafe_allow_html=True,
        )
    except Exception as e:
        logger.error(
            f"An error occurred while processing the results: {str(e)}", exc_info=True
        )
        st.error(f"An error occurred while processing the results: {str(e)}")
import os
import streamlit as st
import pandas as pd
from datetime import datetime
from requests.exceptions import ConnectionError, RequestException
from src.quality.evals import MMLU, CustomEval, PubMedQA, TruthfulQA
from my_utils.ml_logging import get_logger
import asyncio
import traceback
from typing import List, Dict, Any, Tuple
from src.app.quality.results import BenchmarkQualityResult
from src.app.managers import (
    create_benchmark_non_streaming_client,
    create_benchmark_streaming_client
)
import copy

# Set up logger
logger = get_logger()

top_bar = st.empty()

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
    "settings_quality",
    "benchmark_selection_multiselect",
    "benchmark_selection"
]
initial_values = {
    "settings_quality": {},
    "benchmark_selection_multiselect": [],
    "benchmark_selection": []
}

initialize_session_state(session_vars, initial_values)

def get_task_list(test: str = None) -> List[asyncio.Task]:
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

def run_retrieval_quality_for_client(client: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Run retrieval quality evaluation for a single client.

    :param client: The evaluation client to run the retrieval quality evaluation for.
    :return: A tuple containing the deployment name and the evaluation results.
    """
    try:
        results_retrieval = client.run_retrieval_quality(data_input=st.session_state["evaluation_clients_retrieval_df"])
        deployment_name = client.model_config.azure_deployment
        return deployment_name, results_retrieval
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return client.model_config.azure_deployment, None

async def run_benchmark_tests() -> None:
    """
    Run benchmark tests and log progress. This function runs selected benchmark tests asynchronously.

    :return: None
    """
    try:
        results = []
        if "results_quality" not in st.session_state:
            st.session_state["results_quality"] = {}

        # Initialize dataframes
        results_df = pd.DataFrame()
        truthful_results = pd.DataFrame()
        mmlu_results = pd.DataFrame()
        medpub_results = pd.DataFrame()
        retrieval_results = pd.DataFrame()
        rai_results = pd.DataFrame()

        settings = st.session_state.get("settings_quality", {})

        # Run benchmarks based on settings
        if "benchmark_selection_multiselect" in settings:
            if "MMLU" in settings["benchmark_selection_multiselect"]:
                logger.info("Running MMLU benchmark")
                mmlu_tasks = get_task_list(test="mmlu")
                mmlu_stats = await asyncio.gather(*mmlu_tasks)
                if mmlu_stats:
                    mmlu_results = pd.concat(mmlu_stats, ignore_index=True)
                    results.append(mmlu_results)

            if "MedPub QA" in settings["benchmark_selection_multiselect"]:
                logger.info("Running MedPub QA benchmark")
                medpub_tasks = get_task_list(test="medpub")
                medpub_stats = await asyncio.gather(*medpub_tasks)
                if medpub_stats:
                    medpub_results = pd.concat(medpub_stats, ignore_index=True)
                    results.append(medpub_results)

            if "Truthful QA" in settings["benchmark_selection_multiselect"]:
                logger.info("Running Truthful QA benchmark")
                truthful_tasks = get_task_list(test="truthfulqa")
                truthful_stats = await asyncio.gather(*truthful_tasks)
                if truthful_stats:
                    truthful_results = pd.concat(truthful_stats, ignore_index=True)
                    results.append(truthful_results)

        if "benchmark_selection" in settings:
            # Retrieval benchmark
            if "retrieval" in settings.get("benchmark_selection", []):
                logger.info("Running retrieval quality evaluations")
                clients = st.session_state.get("evaluation_clients_retrieval", [])
                data_for_df = []
                for deployment_name, deployment in st.session_state.deployments.items():
                    results_retrievals = {}
                    try:
                        df_input = st.session_state["settings_quality"]["retrieval_df"]
                        df_filtered = df_input[df_input['deployment'] == deployment_name]
                        if not df_filtered.empty:
                            metrics, studio_url = clients[0].run_chat_quality(data_input=df_filtered)
                            metrics['studio_url'] = studio_url
                            results_retrievals[deployment_name] = metrics
                            logger.info(f"Retrieval quality evaluation for {deployment_name} completed.")

                            for model_name, metrics in results_retrievals.items():
                                row_data = metrics.copy()
                                row_data['deployment'] = model_name
                                data_for_df.append(row_data)
                        else:
                            logger.error(f"Failed to run retrieval quality evaluation for client {deployment_name} due to empty filtered data")
                    except Exception as e:
                        logger.error(f"Failed to run retrieval quality evaluation for client {deployment_name}: {e}")
                        st.error(f"Failed to run retrieval quality evaluation for client {deployment_name}. Check logs for details.")
                
                retrieval_results = pd.DataFrame(data_for_df).set_index('deployment')

            # RAI benchmark
            if "rai" in settings.get("benchmark_selection", []):
                logger.info("Running RAI quality evaluations")
                clients = st.session_state.get("evaluation_clients_rai", [])
                data_for_df_rai = []
                for deployment_name, deployment in st.session_state.deployments.items():
                    results_rai = {}
                    for client in clients:
                        try:
                            df_input = st.session_state["settings_quality"]["rai_df"]
                            df_filtered = df_input[df_input['deployment'] == client.model_config.azure_deployment]
                            metrics, studio_url = client.run_chat_content_safety(data_input=df_filtered)
                            metrics['studio_url'] = studio_url
                            results_rai[deployment_name] = metrics
                            logger.info(f"RAI quality evaluation for {deployment_name} completed.")
                            for model_name, metrics in results_rai.items():
                                row_data = metrics.copy()
                                row_data['deployment'] = model_name
                                data_for_df_rai.append(row_data)
                        except RequestException as e:
                            if isinstance(e.__cause__, ConnectionError):
                                logger.warning(f"Connection was forcibly closed by the remote host for client {client.model_config.azure_deployment}. Proceeding with an empty DataFrame.")
                                st.warning(f"The system failed to retrieve data for client {client.model_config.azure_deployment} due to a connection issue. Proceeding with an empty DataFrame.")
                                rai_results = pd.DataFrame()
                            else:
                                logger.error(f"Failed to run RAI quality evaluation for client {client.model_config.azure_deployment}. Exception: {e}")
                                st.error(f"Failed to run RAI quality evaluation for client {client.model_config.azure_deployment}. Check logs for details.")
                                rai_results = pd.DataFrame()
                        except Exception as e:
                            logger.error(f"Failed to run RAI quality evaluation for client {client.model_config.azure_deployment}. Exception: {e}")
                            st.error(f"Failed to run RAI quality evaluation for client {client.model_config.azure_deployment}. Check logs for details.")
                            rai_results = pd.DataFrame()
                
                rai_results = pd.DataFrame(data_for_df_rai).set_index('deployment')

            # Combine results into a dictionary
            results_df = pd.concat(results) if results else pd.DataFrame()
            truthful_results = truthful_results if isinstance(truthful_results, pd.DataFrame) else pd.DataFrame()
            mmlu_results = mmlu_results if isinstance(mmlu_results, pd.DataFrame) else pd.DataFrame()
            medpub_results = medpub_results if isinstance(medpub_results, pd.DataFrame) else pd.DataFrame()
            retrieval_results = retrieval_results if isinstance(retrieval_results, pd.DataFrame) else pd.DataFrame()
            rai_results = rai_results if isinstance(rai_results, pd.DataFrame) else pd.DataFrame()

            results_dict = {
                "understanding_results": results_df,
                "truthful_results": truthful_results,
                "mmlu_results": mmlu_results,
                "medpub_results": medpub_results,
                "retrieval_results": retrieval_results,
                "rai_results": rai_results
            }

            settings_snapshot = copy.deepcopy(settings)
            results_quality = BenchmarkQualityResult(result=results_dict, settings=settings_snapshot)
            st.session_state["results_quality"][results_quality.id] = results_quality.to_dict()
            logger.info("Benchmark results successfully stored in session state")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error(traceback.format_exc())
        top_bar.error(f"An error occurred: {str(e)}")

async def run_evaluation_benchmark(session_key: str) -> pd.DataFrame:
    """
    Run the evaluation benchmark asynchronously.

    :param session_key: The session key to identify the settings and data.
    :return: DataFrame with the results of the benchmark tests.
    """
    try:
        df = await run_benchmark_quality(df=st.session_state["settings_quality"][f"{session_key}_df"],
                                         max_tokens=st.session_state["settings_quality"][f"max_tokens_{session_key}"])
        if df.empty:
            st.info("🚫 No data returned. Switching to default dataset. 🔄")
            default_eval_path = os.path.join("my_utils", "data", "evaluations", "dataframe", "golden_eval_dataset.csv")
            df = pd.read_csv(default_eval_path)
        return df
    except Exception as e:
        logger.error(f"Error in run_evaluation_benchmark: {e}")
        raise e

async def run_benchmark_quality(df: pd.DataFrame, max_tokens: int) -> pd.DataFrame:
    """
    Run the benchmark tests asynchronously, with detailed configuration for each test.

    :param df: DataFrame containing the data for the benchmark tests.
    :param max_tokens: Maximum number of tokens to use in the benchmark tests.
    :return: DataFrame with the results of the benchmark tests.
    """
    try:
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

        prompts = df["question"].tolist()
        context = df["context"].tolist()
        ground_truth = df["ground_truth"].tolist()

        all_results = []

        async def safe_run(client: Any, deployment_name: str) -> List[List[Any]]:
            try:
                return await client.run_latency_benchmark_bulk(
                    deployment_names=[deployment_name], byop=prompts, context=context, ground_truth=ground_truth, max_tokens_list=[max_tokens]
                )
            except Exception as e:
                logger.error(
                    f"An error occurred with deployment '{deployment_name}': {str(e)}",
                    exc_info=True,
                )
                return []

        logger.info(f"Total number of deployment clients: {len(deployment_clients)}")

        for client, deployment_name in deployment_clients:
            results = await safe_run(client, deployment_name)
            modified_results = [result + [deployment_name] for result in results]
            all_results.extend(modified_results)
        
        final_df = pd.DataFrame()
        if all_results:
            final_df = pd.DataFrame(all_results, columns=['question', 'context', 'answer', 'ground_truth', 'deployment'])
        return final_df
    except Exception as e:
        logger.error(
            f"An error occurred while processing the results: {str(e)}", exc_info=True
        )
        raise e
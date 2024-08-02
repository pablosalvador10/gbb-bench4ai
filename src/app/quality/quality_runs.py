import streamlit as st
import asyncio
import pandas as pd
import copy
from my_utils.ml_logging import get_logger
from promptflow.core import AzureOpenAIModelConfiguration
from promptflow.evals.evaluators import (
    RelevanceEvaluator,
    F1ScoreEvaluator,
    GroundednessEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
    SimilarityEvaluator,
)
from src.quality.public_evals import MMLU, PubMedQA, TruthfulQA
from src.app.quality.results import BenchmarkQualityResult
from promptflow.evals.evaluate import evaluate
from datasets import load_dataset
from typing import List, Dict


logger = get_logger()


def _get_deplyoments_to_test() -> List:
    # Get deployments to test
    deployments_to_test = st.session_state["quality_deployments"]
    test_deployments = []
    for deployment_name, deployment in st.session_state.deployments.items():
        if deployment_name in deployments_to_test:
            deployment_config = {
                "key": deployment.get("key"),
                "endpoint": deployment.get("endpoint"),
                "model": deployment_name,
                "version": deployment.get("version"),
            }
            test_deployments.append(deployment_config)
    return test_deployments

def _get_evaluator_deployment() -> Dict:
    for deployment_name, deployment in st.session_state.deployments.items():
        if deployment_name == st.session_state["evaluator_deployment_name"]:
            return deployment
    
    st.error("Evaluator deployment not found.")
    return None

def load_public_data(dataset: str, subset: str, split: str, flatten: bool = False) -> pd.DataFrame:
        # Download dataset
        logger.info(f"Loading {dataset} data")
        hf_data = load_dataset(dataset, subset, split=split)
        if flatten:
            hf_data = hf_data.flatten()
        df = hf_data.to_pandas()
        logger.info(f"Load Complete. {df.shape[0]} rows.")
        return df

async def run_quality_tests() -> None:
    
    logger.info(f"DEPLOYMENTS TO TEST: {_get_deplyoments_to_test()}")
    if _get_deplyoments_to_test() == []:
        st.error("No deployments selected for evaluation.")
        return
    
    # Initialize dataframes
    results_df = pd.DataFrame()
    truthful_results = pd.DataFrame()
    mmlu_results = pd.DataFrame()
    medpub_results = pd.DataFrame()

    # Run custom Benchmark
    if st.session_state["quality_settings"]["type"] == "custom":
        evals_to_run = st.session_state["quality_settings"]["evals_to_run"]
        df = st.session_state["quality_settings"]["custom_dataset"]["custom_df"]
        sample_size = st.session_state["quality_settings"]["custom_dataset"]["custom_subsample"]
        if df == None or sample_size == None:
            st.errror("Please upload a dataset and select the columns to run the benchmark.")
            return

        # generate responses for each deployment

        # instanstiate evaluator class based on evaluator model
        eval_deployment_config = _get_evaluator_deployment()

        # Initialize the evaluators based on evals_to_run
        logger.info("Running custom benchmark...")



    #### Run public Benchmark ####
    elif st.session_state["quality_settings"]["type"] == "public":
        logger.info("Running public benchmark(s)...")
        evals_to_run = st.session_state["quality_settings"]["evals_to_run"]
        
        results = []
        if "mmlu" in evals_to_run:
            logger.info("Running MMLU benchmark")
            settings =  st.session_state["quality_settings"]["mmlu"]
            mmlu_categories = settings.get("mmlu_categories", [])
            mmlu_subsample = settings.get("mmlu_subsample", 100)
            data = load_public_data(dataset="cais/mmlu", subset="all", split="test")
            test_objects = []
            for deployment_config in _get_deplyoments_to_test():
                obj = MMLU(
                    deployment_config=deployment_config,
                    sample_size=mmlu_subsample / 100,
                    log_level="ERROR",
                    categories=mmlu_categories,
                )
                data = obj.transform_data(df=data)
                test_objects.append(obj)
            
            mmlu_tasks = [asyncio.create_task(obj.test(data=data)) for obj in test_objects]
            mmlu_stats = await asyncio.gather(*mmlu_tasks)
            if mmlu_stats:
                mmlu_results = pd.concat(mmlu_stats, ignore_index=True)
                results.append(mmlu_results)

        if "medpub" in evals_to_run:
            logger.info("Running MedPub benchmark")
            settings =  st.session_state["quality_settings"]["medpub"]
            medpub_subsample = settings.get("medpub_subsample", 100)
            data = load_public_data(dataset="qiaojin/PubMedQA", subset="pqa_labeled", split="train", flatten=True)
            test_objects = []
            for deployment_config in _get_deplyoments_to_test():
                obj = PubMedQA(
                    deployment_config=deployment_config,
                    sample_size=medpub_subsample / 100,
                    log_level="ERROR",
                )
                data = obj.transform_data(df=data)
                test_objects.append(obj)
            
            medpub_tasks = [asyncio.create_task(obj.test(data=data)) for obj in test_objects]
            medpub_stats = await asyncio.gather(*medpub_tasks)
            if medpub_stats:
                medpub_results = pd.concat(medpub_stats, ignore_index=True)
                results.append(medpub_results)

        if "truthful" in evals_to_run:
            logger.info("Running TruthfulQA benchmark")
            settings =  st.session_state["quality_settings"]["truthful"]
            truthful_subsample = settings.get("truthful_subsample", 100)
            data = load_public_data(dataset="truthful_qa", subset="multiple_choice", split="validation")
            test_objects = []
            for deployment_config in _get_deplyoments_to_test():
                obj = TruthfulQA(
                    deployment_config=deployment_config,
                    sample_size=truthful_subsample / 100,
                    log_level="ERROR",
                )
                data = obj.transform_data(df=data)
                test_objects.append(obj)
            
            truthful_tasks = [asyncio.create_task(obj.test(data=data)) for obj in test_objects]
            truthful_stats = await asyncio.gather(*truthful_tasks)
            if truthful_stats:
                truthful_results = pd.concat(truthful_stats, ignore_index=True)
                results.append(truthful_results)

        results_df = pd.concat(results) if results else pd.DataFrame()
        # truthful_results = truthful_results if isinstance(truthful_results, pd.DataFrame) else pd.DataFrame()
        mmlu_results = mmlu_results if isinstance(mmlu_results, pd.DataFrame) else pd.DataFrame()
        # medpub_results = medpub_results if isinstance(medpub_results, pd.DataFrame) else pd.DataFrame()

        results_dict = {
                "understanding_results": results_df,
                "truthful_results": truthful_results,
                "mmlu_results": mmlu_results,
                "medpub_results": medpub_results
            }

        settings_snapshot = copy.deepcopy(st.session_state.get("quality_settings", {}))
        results_quality = BenchmarkQualityResult(result=results_dict, settings=settings_snapshot)
        st.session_state["results_quality"][results_quality.id] = results_quality.to_dict()
        logger.info(" Public Benchmark results successfully stored in session state")



    else:
        st.warning("Invalid Benchmark type. Must be either custom or public.")
    
    return
from promptflow.core import AzureOpenAIModelConfiguration
from promptflow.evals.evaluators import (
    RelevanceEvaluator,
    F1ScoreEvaluator,
    GroundednessEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
    SimilarityEvaluator,
    ViolenceEvaluator,
    SexualEvaluator,
    SelfHarmEvaluator,
    HateUnfairnessEvaluator,
)
from promptflow.evals.evaluate import evaluate


import pandas as pd
import streamlit as st
import tempfile
import json
from my_utils.ml_logging import get_logger
from typing import Dict, List

class AzureAIEval:

    def __init__(self, evaluator_deployment: Dict, eval_metrics: List):
        self.logger = get_logger(__name__)
        self.model_config = AzureOpenAIModelConfiguration(
                        azure_endpoint=evaluator_deployment.get("endpoint"),
                        api_key=evaluator_deployment.get("key"),
                        azure_deployment=st.session_state["evaluator_deployment_name"]
                        )

        self.eval_metrics = eval_metrics

    
    def _get_evaluators(self) -> Dict:
        evaluators = {}
        if "coherence" in self.eval_metrics:
            coherence = CoherenceEvaluator(model_config=self.model_config)
            evaluators["coherence"] = coherence

        if "fluency" in self.eval_metrics:
            fluency = FluencyEvaluator(model_config=self.model_config)
            evaluators["fluency"] = fluency

        if "relevance" in self.eval_metrics:
            relevance = RelevanceEvaluator(model_config=self.model_config)
            evaluators["relevance"] = relevance

        if "groundedness" in self.eval_metrics:
            groundedness = GroundednessEvaluator(model_config=self.model_config)
            evaluators["groundedness"] = groundedness

        if "similarity" in self.eval_metrics:
            similarity = SimilarityEvaluator(model_config=self.model_config)
            evaluators["similarity"] = similarity

        if "f1score" in self.eval_metrics:
            f1score = F1ScoreEvaluator()
            evaluators["f1_score"] = f1score

        if "violenceevaluator" in self.eval_metrics:
            violenceevaluator = ViolenceEvaluator(model_config=self.model)
            evaluators["violenceevaluator"] = violenceevaluator

        if "sexualevlauator" in self.eval_metrics:
            sexualevaluator = SexualEvaluator(model_config=self.model_config)
            evaluators["sexualevaluator"] = sexualevaluator

        if "selfharmevaluator" in self.eval_metrics:
            selfharmevaluator = SelfHarmEvaluator(model_config=self.model_config)
            evaluators["selfharmevaluator"] = selfharmevaluator

        if "hateunfairnessevaluator" in self.eval_metrics:
            hateunfairnessevaluator = HateUnfairnessEvaluator(model_config=self.model_config)
            evaluators["hateunfairnessevaluator"] = hateunfairnessevaluator

        return evaluators
    
    def _get_evaluator_config(self) -> Dict:
        evaluator_config = {}
        if "coherence" in self.eval_metrics:
            evaluator_config["coherence"] = {
                "answer": "${data.answer}",
                "question": "${data.prompt}"
            }
        
        if "fluency" in self.eval_metrics:
            evaluator_config["fluency"] = {
                "answer": "${data.answer}",
                "question": "${data.prompt}"
            }

        if "relevance" in self.eval_metrics:
            evaluator_config["relevance"] = {
                "answer": "${data.answer}",
                "context": "${data.context}",
                "question": "${data.prompt}"
            }

        if "groundedness" in self.eval_metrics:
            evaluator_config["groundedness"] = {
                "answer": "${data.answer}",
                "context": "${data.context}"
            }

        if "similarity" in self.eval_metrics:
            evaluator_config["similarity"] = {
                "answer": "${data.answer}",
                "ground_truth": "${data.ground_truth}",
                "question": "${data.prompt}"
            }

        if "f1score" in self.eval_metrics:
            evaluator_config["f1_score"] = {
                "answer": "${data.answer}",
                "ground_truth": "${data.ground_truth}"
            }

        if "violenceevaluator" in self.eval_metrics:
            evaluator_config["violenceevaluator"] = {
                "answer": "${data.answer}",
                "question": "${data.prompt}"
            }
        
        if "sexualevlauator" in self.eval_metrics:
            evaluator_config["sexualevaluator"] = {
                "answer": "${data.answer}",
                "question": "${data.prompt}"
            }
        
        if "selfharmevaluator" in self.eval_metrics:
            evaluator_config["selfharmevaluator"] = {
                "answer": "${data.answer}",
                "question": "${data.prompt}"
            }

        if "hateunfairnessevaluator" in self.eval_metrics:
            evaluator_config["hateunfairnessevaluator"] = {
                "answer": "${data.answer}",
                "question": "${data.prompt}"
            }


        return evaluator_config
   
    def _convert_to_jsonl(self, data: pd.DataFrame, temp_file: tempfile.NamedTemporaryFile) -> None:
        """
        Convert DataFrame to JSONL format and write to a temporary file.

        :param data: DataFrame containing the data.
        :param temp_file: Temporary file to write the JSONL data.
        """
        try:
            for record in data.to_dict(orient='records'):
                json_record = json.dumps(record)
                temp_file.write(json_record + '\n')
            temp_file.flush()
            self.logger.info("Data successfully converted to JSONL format.")
        except Exception as e:
            self.logger.error(f"Error converting DataFrame to JSONL: {e}")
            raise
    
    def run_tests(self, data: pd.DataFrame) -> None:

        evaluators = self._get_evaluators()
        evaluator_configs = self._get_evaluator_config()

        # get path to data. Convert data to jsonl
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl', mode='w')
        self._convert_to_jsonl(data, temp_file)

        result = evaluate(
            data=temp_file.name,
            evaluators=evaluators,
            evaluator_config=evaluator_configs
        )

        return result
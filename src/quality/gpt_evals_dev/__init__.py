import os
import pandas as pd
from typing import List, Optional, Dict, Any
import logging
from promptflow.core import AzureOpenAIModelConfiguration
from promptflow.evals.evaluators import (
    RelevanceEvaluator,
    F1ScoreEvaluator,
    GroundednessEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
    SimilarityEvaluator,
)
from promptflow.evals.evaluate import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceAndQualityEvaluators:
    def __init__(self, model_config: AzureOpenAIModelConfiguration):
        """
        Initialize performance and quality evaluators.

        :param model_config: Configuration for Azure OpenAI model.
        :type model_config: AzureOpenAIModelConfiguration
        """
        self.model_config = model_config
        self.evaluators = {
            "RelevanceEvaluator": RelevanceEvaluator(self.model_config),
            "GroundednessEvaluator": GroundednessEvaluator(self.model_config),
            "CoherenceEvaluator": CoherenceEvaluator(self.model_config),
            "FluencyEvaluator": FluencyEvaluator(self.model_config),
            "SimilarityEvaluator": SimilarityEvaluator(self.model_config),
            "F1ScoreEvaluator": F1ScoreEvaluator()
        }

    def get_evaluators(self) -> Dict[str, Any]:
        """
        Get the performance and quality evaluators.

        :return: A dictionary of evaluator instances.
        :rtype: dict
        """
        return self.evaluators

    def execute_evaluators(self, data_input: Any, evaluator_names: List[str], output_jsonl_path: Optional[str] = None, **kwargs) -> None:
        """
        Execute specified evaluators on the loaded data and save results in JSONL format.

        :param data_input: A pandas DataFrame or a path to a CSV file containing the data.
        :param evaluator_names: A list of evaluator names to be executed.
        :param output_jsonl_path: Path to save the JSONL file with the results.
        :param kwargs: Additional keyword arguments for evaluators.
        :raises ValueError: If an evaluator is not found or required columns are missing.
        """
        try:
            if isinstance(data_input, pd.DataFrame):
                data = data_input
            elif isinstance(data_input, str):
                data = pd.read_csv(data_input)
            else:
                raise ValueError("data_input must be a pandas DataFrame or a path to a CSV file.")

            with tempfile.TemporaryDirectory() as temp_dir:
                jsonl_file_path = os.path.join(temp_dir, "data.jsonl")
                data.to_json(jsonl_file_path, orient='records', lines=True)
                logger.info(f"Saved DataFrame to JSONL format at: {jsonl_file_path}")

                evaluators = {name: self.evaluators[name] for name in evaluator_names if name in self.evaluators}
                evaluator_config = {name: {"data": f"${{data}}"} for name in evaluator_names}

                evaluate_config = {
                    "data": jsonl_file_path,
                    "evaluators": evaluators,
                    "evaluator_config": evaluator_config,
                    "kwargs": kwargs
                }

                logger.info(f'''Evaluator Configuration: {evaluate_config}. This configuration
                             maps evaluators to their required data columns for processing.''')
                
                result = evaluate(**evaluate_config)
                logger.info("Evaluation completed successfully.")
                
                if output_jsonl_path:
                    with open(output_jsonl_path, 'w') as f:
                        json.dump(result, f)
                    logger.info(f"Results saved to {output_jsonl_path}")

                return result

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise

    def get_available_evaluators(self) -> List[str]:
        """
        Get a list of available evaluators.

        :return: A list of evaluator names.
        """
        return list(self.evaluators.keys())


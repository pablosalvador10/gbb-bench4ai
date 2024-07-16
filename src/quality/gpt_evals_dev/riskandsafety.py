import os
import pandas as pd
from typing import List, Optional, Dict, Any
import tempfile
from promptflow.evals.evaluators import (
    ViolenceEvaluator,
    SexualEvaluator,
    SelfHarmEvaluator,
    HateUnfairnessEvaluator,
)
from promptflow.evals.evaluate import evaluate

from my_utils.ml_logging import get_logger
logger = get_logger()

class RiskAndSafetyEvaluators:
    def __init__(self, project_scope: dict, credential=None):
        """
        Initialize risk and safety evaluators.

        :param project_scope: The scope of the Azure AI project.
        :type project_scope: dict
        :param credential: The credential for connecting to Azure AI project.
        :type credential: TokenCredential, optional
        """
        self.project_scope = project_scope
        self.credential = credential
        self.evaluators = {
            "ViolenceEvaluator": ViolenceEvaluator(project_scope=self.project_scope, credential=self.credential),
            "SexualEvaluator": SexualEvaluator(project_scope=self.project_scope, credential=self.credential),
            "SelfHarmEvaluator": SelfHarmEvaluator(project_scope=self.project_scope, credential=self.credential),
            "HateUnfairnessEvaluator": HateUnfairnessEvaluator(project_scope=self.project_scope, credential=self.credential)
        }

    def get_evaluators(self) -> Dict[str, Any]:
        """
        Get the risk and safety evaluators.

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


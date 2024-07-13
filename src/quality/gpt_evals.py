import os
import pandas as pd
from typing import List, Optional, Dict, Any
from promptflow.core import AzureOpenAIModelConfiguration
from promptflow.evals.evaluators import (
    RelevanceEvaluator,
    F1ScoreEvaluator,
    GroundednessEvaluator,
    ViolenceEvaluator,
    SexualEvaluator,
    SelfHarmEvaluator,
    HateUnfairnessEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
    SimilarityEvaluator,
    QAEvaluator,
    ChatEvaluator,
    ContentSafetyEvaluator,
    ContentSafetyChatEvaluator
)

class GPTEvals:
    def __init__(self, azure_endpoint: Optional[str] = None, 
                 api_key: Optional[str] = None,
                 azure_deployment: Optional[str] = None, 
                 api_version: Optional[str] = None):
        """
        Initializes a comprehensive evaluation framework for Azure OpenAI model responses. 
        This class configures the model and sets up various evaluators to assess different aspects of AI-generated text, 
        including relevance, coherence, fluency, content safety, and more. Evaluators can be customized or extended based on specific needs.

        Parameters can be passed directly or sourced from environment variables.
        """
        self.model_config = AzureOpenAIModelConfiguration(
            azure_endpoint=azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            azure_deployment=azure_deployment or os.environ.get("AZURE_AOAI_COMPLETION_MODEL_DEPLOYMENT_ID"),
            api_version=api_version or os.environ.get("DEPLOYMENT_VERSION"),
        )
        self.evaluators = {
            "RelevanceEvaluator": (RelevanceEvaluator(self.model_config), ["question", "answer", "context"]),
            "F1ScoreEvaluator": (F1ScoreEvaluator(), ["answer", "ground_truth"]),
            "GroundednessEvaluator": (GroundednessEvaluator(self.model_config), ["answer", "context"]),
            "ViolenceEvaluator": (ViolenceEvaluator(self.model_config), ["question", "answer"]),
            "SexualEvaluator": (SexualEvaluator(self.model_config), ["question", "answer"]),
            "SelfHarmEvaluator": (SelfHarmEvaluator(self.model_config), ["question", "answer"]),
            "HateUnfairnessEvaluator": (HateUnfairnessEvaluator(self.model_config), ["question", "answer"]),
            "CoherenceEvaluator": (CoherenceEvaluator(self.model_config), ["question", "answer"]),
            "FluencyEvaluator": (FluencyEvaluator(self.model_config), ["question", "answer"]),
            "SimilarityEvaluator": (SimilarityEvaluator(self.model_config), ["question", "answer", "ground_truth"]),
            "QAEvaluator": (QAEvaluator(self.model_config), ["question", "answer", "context", "ground_truth"]),
            "ChatEvaluator": (ChatEvaluator(self.model_config), ["question", "answer", "context", "ground_truth"]),
            "ContentSafetyEvaluator": (ContentSafetyEvaluator(self.model_config), ["question", "answer", "context", "ground_truth"]),
            "ContentSafetyChatEvaluator": (ContentSafetyChatEvaluator(self.model_config), ["question", "answer", "context", "ground_truth"]),
        }
        self.data: List[Dict[str, Any]] = []

    def load_csv(self, file_path: str) -> None:
        """
        Load data from a CSV file and store it as a list of dictionaries.
        
        :param file_path: Path to the CSV file.
        """
        df = pd.read_csv(file_path)
        self.data = df.to_dict(orient='records')

    def check_required_columns(self, row: Dict[str, Any], required_columns: List[str]) -> bool:
        """
        Check if the required columns are present in the row.

        :param row: A dictionary representing a row of data.
        :param required_columns: A list of required columns.
        :return: True if all required columns are present, False otherwise.
        """
        return all(col in row for col in required_columns)

    def execute_evaluator(self, evaluator_name: str) -> List[Dict[str, Any]]:
        """
        Execute the specified evaluator on the loaded data.

        :param evaluator_name: The name of the evaluator to be executed.
        :return: A list of results from the evaluator.
        :raises ValueError: If the evaluator is not found or required columns are missing.
        """
        if evaluator_name not in self.evaluators:
            raise ValueError(f"Evaluator {evaluator_name} not found.")

        evaluator, required_columns = self.evaluators[evaluator_name]
        results = []
        for row in self.data:
            if not self.check_required_columns(row, required_columns):
                raise ValueError(f"Missing required columns for {evaluator_name}: {required_columns}")
            result = evaluator(**{col: row[col] for col in required_columns})
            results.append(result)
        return results

    def get_available_evaluators(self) -> List[str]:
        """
        Get a list of available evaluators.

        :return: A list of evaluator names.
        """
        return list(self.evaluators.keys())


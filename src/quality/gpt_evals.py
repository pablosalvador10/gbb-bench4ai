import pandas as pd
import json
import tempfile
from typing import Any, Optional, Dict
import os
import logging
from promptflow.evals.evaluate import evaluate
from promptflow.core import AzureOpenAIModelConfiguration
from promptflow.evals.evaluators import QAEvaluator, ContentSafetyEvaluator

# Set up logging
from utils.ml_logging import get_logger

logger = get_logger()

class AzureAIQualityEvaluator:
    def __init__(self,
                 azure_endpoint: Optional[str] = None,
                 api_key: Optional[str] = None,
                 azure_deployment: Optional[str] = None,
                 api_version: Optional[str] = None,
                 subscription_id: Optional[str] = None,
                 resource_group_name: Optional[str] = None,
                 project_name: Optional[str] = None):
        """
        Initialize the AzureAIQualityEvaluator with model configuration and evaluators.

        :param azure_endpoint: Azure OpenAI endpoint.
        :param api_key: API key for Azure OpenAI.
        :param azure_deployment: Azure OpenAI deployment ID.
        :param api_version: API version for Azure OpenAI.
        :param subscription_id: Azure subscription ID.
        :param resource_group_name: Azure resource group name.
        :param project_name: Azure project name.
        """
        try:
            self.model_config = AzureOpenAIModelConfiguration(
                azure_endpoint=azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                azure_deployment=azure_deployment or os.environ.get("AZURE_AOAI_COMPLETION_MODEL_DEPLOYMENT_ID"),
                api_version=api_version or os.environ.get("DEPLOYMENT_VERSION"),
            )

            self.qa_evaluator = QAEvaluator(model_config=self.model_config, parallel=True)

            self.azure_ai_project = None
            if subscription_id and resource_group_name and project_name:
                self.azure_ai_project = {
                    "subscription_id": subscription_id,
                    "resource_group_name": resource_group_name,
                    "project_name": project_name
                }
            
            self.content_safety_evaluator = ContentSafetyEvaluator(
                project_scope=self.azure_ai_project, parallel=True
            )

            logger.info("AzureAIQualityEvaluator initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize AzureAIQualityEvaluator: {e}")
            raise

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
            logger.info("Data successfully converted to JSONL format.")
        except Exception as e:
            logger.error(f"Error converting DataFrame to JSONL: {e}")
            raise

    def run_chat_quality(self, data_input: Any, 
                         azure_ai_project: Optional[Dict]=None) -> Any:
        """
        Evaluate the quality of chat responses using the QA evaluator.

        :param data_input: A pandas DataFrame or a path to a CSV file containing the data.
        :return: The result of the evaluation.
        """
        temp_file = None
        try:
            if isinstance(data_input, pd.DataFrame):
                data = data_input
            elif isinstance(data_input, str):
                data = pd.read_csv(data_input)
            else:
                raise ValueError("data_input must be a pandas DataFrame or a path to a CSV file.")

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl', mode='w')
            self._convert_to_jsonl(data, temp_file)

            result = evaluate(
                data=temp_file.name,
                evaluators={
                    "qa_evaluator": self.qa_evaluator,
                },
                evaluator_config={
                    "qa_evaluator": {
                        "question": "${data.question}",
                        "answer": "${data.answer}",
                        "context": "${data.context}",
                        "ground_truth": "${data.ground_truth}",
                    },
                },
                azure_ai_project=azure_ai_project or self.azure_ai_project
            )
            logger.info("Quality evaluation completed successfully.")
            return result

        except Exception as e:
            logger.error(f"Error during quality evaluation: {e}")
            raise
        finally:
            try:
                if temp_file:
                    os.remove(temp_file.name)
                    logger.info(f"Temporary file {temp_file.name} removed.")
            except Exception as e:
                logger.error(f"Error removing temporary file: {e}")

    def run_chat_content_safety(self, data_input: Any, 
                                azure_ai_project: Optional[Dict]=None) -> Any:
        """
        Evaluate the content safety of chat responses using the Content Safety evaluator.

        :param data_input: A pandas DataFrame or a path to a CSV file containing the data.
        :return: The result of the evaluation.
        """
        temp_file = None
        try:
            if isinstance(data_input, pd.DataFrame):
                data = data_input
            elif isinstance(data_input, str):
                data = pd.read_csv(data_input)
            else:
                raise ValueError("data_input must be a pandas DataFrame or a path to a CSV file.")

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl', mode='w')
            self._convert_to_jsonl(data, temp_file)

            result = evaluate(
                data=temp_file.name,
                evaluators={
                    "content_safety_evaluator": self.content_safety_evaluator,
                },
                evaluator_config={
                    "content_safety_evaluator": {
                        "question": "${data.question}",
                        "answer": "${data.answer}",
                    },
                },
                azure_ai_project=azure_ai_project or self.azure_ai_project
            )
            logger.info("Content safety evaluation completed successfully.")
            return result

        except Exception as e:
            logger.error(f"Error during content safety evaluation: {e}")
            raise
        finally:
            try:
                if temp_file:
                    os.remove(temp_file.name)
                    logger.info(f"Temporary file {temp_file.name} removed.")
            except Exception as e:
                logger.error(f"Error removing temporary file: {e}")
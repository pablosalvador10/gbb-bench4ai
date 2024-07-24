from src.aoai.azure_openai import AzureOpenAIManager
from src.quality.gpt_evals import AzureAIQualityEvaluator
from src.performance.latencytest import (AzureOpenAIBenchmarkNonStreaming,
                                         AzureOpenAIBenchmarkStreaming)

from typing import Optional

def create_azure_openai_manager(
    api_key: str, azure_endpoint: str, api_version: str, deployment_id: str
) -> AzureOpenAIManager:
    """
    Create a new Azure OpenAI Manager instance.

    :param api_key: API key for Azure OpenAI.
    :param azure_endpoint: API endpoint for Azure OpenAI.
    :param api_version: API version for Azure OpenAI.
    :param deployment_id: Deployment ID for Azure OpenAI.
    :return: AzureOpenAIManager instance.
    """
    return AzureOpenAIManager(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        chat_model_name=deployment_id,
    )


def create_benchmark_non_streaming_client(
    api_key: str, endpoint: str, api_version: str
) -> AzureOpenAIBenchmarkNonStreaming:
    """
    Create a new benchmark client instance for non-streaming.

    :param api_key: API key for Azure OpenAI.
    :param endpoint: API endpoint for Azure OpenAI.
    :param api_version: API version for Azure OpenAI.
    :return: AzureOpenAIBenchmarkNonStreaming instance.
    """
    return AzureOpenAIBenchmarkNonStreaming(
        api_key=api_key, azure_endpoint=endpoint, api_version=api_version
    )


def create_benchmark_streaming_client(
    api_key: str, endpoint: str, api_version: str
) -> AzureOpenAIBenchmarkStreaming:
    """
    Create a new benchmark client instance for streaming.

    :param api_key: API key for Azure OpenAI.
    :param endpoint: API endpoint for Azure OpenAI.
    :param api_version: API version for Azure OpenAI.
    :return: AzureOpenAIBenchmarkStreaming instance.
    """
    return AzureOpenAIBenchmarkStreaming(
        api_key=api_key, azure_endpoint=endpoint, api_version=api_version
    )


def create_eval_client(
    azure_endpoint: str,
    api_key: str,
    azure_deployment: str,
    api_version: str,
    subscription_id: Optional[str] = None,
    resource_group_name: Optional[str] = None,
    project_name: Optional[str] = None,
) -> AzureAIQualityEvaluator:
    """
    Create a new Azure AI Quality Evaluator instance.
    :return: AzureAIQualityEvaluator instance.
    """
    return AzureAIQualityEvaluator(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            azure_deployment=azure_deployment,
            api_version=api_version,
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
            project_name=project_name
        )

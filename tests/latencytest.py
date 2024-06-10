import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from src.performance.aoaihelpers.utils import calculate_statistics
from src.performance.latencytest import AzureOpenAIBenchmarkNonStreaming
from unittest.mock import patch, MagicMock
from math import inf
import os


API_KEY = os.getenv("OPENAI_API_KEY_EAST2")
API_VERSION = (os.getenv("AZURE_OPENAI_API_VERSION") or "2023-05-15")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT_EAST2")
DEPLOYMENT_NAME = "gpt4-turbo"
MAX_TOKENS = 100

az = AzureOpenAIBenchmarkNonStreaming(api_key=API_KEY,
                                      api_version=API_VERSION,
                                      azure_endpoint=AZURE_ENDPOINT)

@pytest.mark.parametrize("data, expected", [
    ([], (None, None, None, None, None)),
    ([2]*5, (2.0, 0.0, 2.0, 2.0, 0.0)),
    ([2,4,6,8,10,12,56,34],(9.0, 12.0, 48.29999999999999, 54.459999999999994, 1.066649448896851)),
    ([3,4.5,6.789,1000,456.678,8769.98,8967.90,1000], (728.3389999999999, 2936.27825, 8898.627999999999, 8954.0456, 1.4578329585969763)),
    ([2,"none","none",8,10,12,56,34], (None, None, None, None, None)),
])
def test_calculate_statistics_core(data, expected):
    """
    Test the calculate_statistics function with various types of data.
    Each test case is designed to verify the function's capability to handle different data distributions,
    including empty lists, repeated values, negative numbers, mixed data types, very small or very large numbers,
    and lists with outliers. This ensures robust performance across a wide range of real-world scenarios.
    """
    assert calculate_statistics(data) == expected


def test_validate_api_configuration():
    with patch.object(AzureOpenAIBenchmarkNonStreaming, '__init__', return_value=None) as mock_init:

        AzureOpenAIBenchmarkNonStreaming.azure_endpoint = 'mocked_endpoint' 
        AzureOpenAIBenchmarkNonStreaming.api_key = "mocked_key"
        AzureOpenAIBenchmarkNonStreaming.api_version = "mocked_version"

        az = AzureOpenAIBenchmarkNonStreaming()
        az._validate_api_configurations()

@pytest.mark.parametrize("data, expected", [
    ({
    "times_succesful": [2,None,5,7,None],
    "times_unsucessfull": [None,3,5,8,10],
    "regions": ["eastus"],
    "number_of_iterations": 6, 
    "completion_tokens": [45,67,89,97,56,78],
    "prompt_tokens": [3,6,7,8,7,8,34,3456],
    "errors": {"count": 3, "codes": [429,500,502,429,429]},
    "best_run": {
        "time": inf,
        "completion_tokens": 0,
        "prompt_tokens": 0,
    },
    "worst_run": {
        "time": -inf,
        "completion_tokens": 0,
        "prompt_tokens": 0,
    }
}, {'median_time': 5.0,
'regions': ['eastus'],
'iqr_time': 2.5,
'percentile_95_time': 6.8,
'percentile_99_time': 6.96,
'cv_time': 0.44031528592635544,
'median_completion_tokens': 72.5,
'iqr_completion_tokens': 27.5,
'percentile_95_completion_tokens': 95.0,
'percentile_99_completion_tokens': 96.6,
'cv_completion_tokens': 0.25102669836529556,
'median_prompt_tokens': 7.5,
'iqr_prompt_tokens': 7.75,
'percentile_95_prompt_tokens': 2258.2999999999984,
'percentile_99_prompt_tokens': 3216.459999999999,
'cv_prompt_tokens': 2.5832862634989806,
'error_rate': 0.5,
'number_of_iterations': 6,
'throttle_count': 3,
'throttle_rate': 0.5,
'errors_types': [429, 500, 502, 429, 429],
'successful_runs': 5,
'unsuccessful_runs': 5,
'best_run': {'time': inf, 'completion_tokens': 0, 'prompt_tokens': 0},
'worst_run': {'time': -inf, 'completion_tokens': 0, 'prompt_tokens': 0}}),
])
def test_calculate_statistics_method(data, expected):
    """
    Test the calculate_statistics function with various types of data.
    Each test case is designed to verify the function's capability to handle different data distributions,
    including empty lists, repeated values, negative numbers, mixed data types, very small or very large numbers,
    and lists with outliers. This ensures robust performance across a wide range of real-world scenarios.
    """
    assert az._calculate_statistics(data) == expected


def test_make_call_succesfull_nostreaming():
    """test make call no streaming succesful"""
    
    az.make_call(deployment_name=DEPLOYMENT_NAME,
                 max_tokens=MAX_TOKENS,
                 temperature=0,
                 prevent_server_caching=True,
                 )
    
    key = f"{DEPLOYMENT_NAME}_{MAX_TOKENS}"
    assert len(az.results[key]["times_succesful"]) != 0 
    assert len(az.results[key]["completion_tokens"]) != 0 
    assert len(az.results[key]["prompt_tokens"]) != 0
    assert az.results[key]["best_run"]["completion_tokens"] > 0
    assert az.results[key]["best_run"]["prompt_tokens"] > 0
    assert az.results[key]["best_run"]["time"] != inf

    az.make_call(deployment_name=DEPLOYMENT_NAME,
                max_tokens=100,
                temperature=0,
                prevent_server_caching=True,
                )
    
    assert len(az.results[key]["times_succesful"]) == 2
    assert len(az.results[key]["completion_tokens"]) == 2 
    assert len(az.results[key]["prompt_tokens"]) == 2


def test_make_call_unsuccesfull_nostreaming():
    """test make call no streaming unsucesful"""
    
    az.make_call(deployment_name=DEPLOYMENT_NAME,
                 max_tokens=MAX_TOKENS,
                 temperature=0,
                 prevent_server_caching=True,
                 )
    
    key = f"{DEPLOYMENT_NAME}_{MAX_TOKENS}"
    assert az.results[key]["errors"]["count"] == 1
    assert az.results[key]["errors"]["codes"][0] == 500
    assert len(az.results[key]["times_unsucessfull"]) != 0 

    az.make_call(deployment_name=DEPLOYMENT_NAME,
                 max_tokens=MAX_TOKENS,
                 temperature=0,
                 prevent_server_caching=True,
                 )
    
    assert az.results[key]["errors"]["count"] == 2
    assert az.results[key]["errors"]["codes"][1] == 500
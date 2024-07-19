"""
This Python script is designed to perform latency testing for the Azure OpenAI service. It includes the necessary imports and setup for asynchronous API calls, backoff strategies for error handling, and utilities for message generation, token counting, and logging. The script leverages environment variables for configuration, sets up a logger for output, and defines constants for use in HTTP headers and user agent identification during API requests. Its purpose is to benchmark the response time and efficiency of various models hosted on Azure OpenAI, aiding in performance analysis and optimization.
"""
import asyncio
import json
import os
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
from dotenv import load_dotenv
from tabulate import tabulate
from termcolor import colored

from src.performance.aoaihelpers.utils import (
    calculate_statistics, extract_rate_limit_and_usage_info_async,
    get_local_time_in_azure_region, log_system_info)
from src.performance.messagegeneration import (BYOPMessageGenerator,
                                               RandomMessagesGenerator)
from src.performance.utils import (count_tokens, detect_model_encoding,
                                   get_encoding_from_model_name,
                                   split_list_into_variable_parts)
from my_utils.ml_logging import get_logger

load_dotenv()

# Set up logger
logger = get_logger()

# Constants for headers, user agent and assistant
TELEMETRY_USER_AGENT_HEADER = "x-ms-useragent"
USER_AGENT = "latency-benchmark"
MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Give me history of Seattle"},
]
MAX_RETRY_ATTEMPTS = 3
MAX_TIMEOUT_SECONDS = 60
RETRY_AFTER_MS_HEADER = "retry-after-ms"


class AzureOpenAIBenchmarkLatency(ABC):
    def __init__(
        self,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        """
        Initializes an instance of AzureOpenAIBenchmarkLatency.

        This base class is designed for benchmarking latency on Azure OpenAI services. It requires API key, API version, and endpoint for Azure OpenAI services. If not provided, these values are attempted to be retrieved from environment variables.

        :param api_key: The API key for authenticating requests to Azure OpenAI. Defaults to the environment variable 'AZURE_OPENAI_KEY' if not provided.
        :param azure_endpoint: The endpoint URL for Azure OpenAI services. Defaults to the environment variable 'AZURE_OPENAI_API_ENDPOINT' if not provided.
        :param api_version: The version of the Azure OpenAI API to use. Defaults to the environment variable 'AZURE_OPENAI_API_VERSION' or '2023-05-15' if neither is provided.
        """
        self.api_key: Optional[str] = api_key or os.getenv("AZURE_OPENAI_KEY")
        self.api_version: str = (
            api_version or os.getenv("AZURE_OPENAI_API_VERSION") or "2023-05-15"
        )
        self.azure_endpoint: Optional[str] = azure_endpoint or os.getenv(
            "AZURE_OPENAI_API_ENDPOINT"
        )
        self._validate_api_configurations()
        self.results: dict = {}
        self.is_streaming: str = "N/A"

    def _validate_api_configurations(self) -> None:
        """
        Validates if all necessary configurations are set for the Azure OpenAI API connection.

        This method checks if the API key, Azure endpoint, and API version are all provided. If any of these are missing, it raises a ValueError indicating that one or more necessary configurations are not set.

        :raises ValueError: If the API key, Azure endpoint, or API version is not provided.
        """
        if not all(
            [
                self.api_key,
                self.azure_endpoint,
                self.api_version,
            ]
        ):
            raise ValueError(
                "One or more OpenAI API setup variables are empty. Please review your environment variables and `SETTINGS.md`"
            )

    @abstractmethod
    async def make_call(
        self,
        deployment_name: str,
        max_tokens: int,
        temperature: Optional[int] = 0,
        timeout: int = 120,
        prompt: Optional[str] = None,
        context: Optional[str] = None,
        context_tokens: Optional[int] = None,
        prevent_server_caching: Optional[bool] = True,
        top_p: int = 1,
        n: int = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
    ):
        """
        Asynchronously makes a chat completion call to the specified deployment and logs the latency.

        This abstract method should be implemented to make an API call to the Azure OpenAI service or any other specified deployment, capturing and logging the time taken for the call to complete.

        :param deployment_name: The name of the deployment to which the call is made.
        :param max_tokens: The maximum number of tokens to generate in the completion.
        :param temperature: Controls randomness in generation. Setting to 0 results in deterministic output. Defaults to 0.
        :param timeout: The maximum time in seconds to wait for the call to complete. Defaults to 120.
        :param prompt: The input text to generate completions for. Optional.
        :param context_tokens: The number of tokens to consider from the prompt for generating completions. Optional.
        :param prevent_server_caching: If True, prevents the server from caching the request. Defaults to True.
        :param top_p: Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered. Defaults to 1.
        :param n: The number of completions to generate. Defaults to 1.
        :param presence_penalty: Adds a penalty for repeated usage of the same token in the completion. Defaults to 0.
        :param frequency_penalty: Adds a penalty for the frequency of token usage in the completion. Defaults to 0.
        """
        pass

    async def run_latency_benchmark(
        self,
        deployment_names: List[str],
        max_tokens_list: List[int],
        iterations: int = 1,
        same_model_interval: int = 1,
        different_model_interval: int = 5,
        temperature: Optional[int] = 0,
        byop: Optional[List] = None,
        context: Optional[List] = None,
        ground_truth: Optional[List] = None,
        context_tokens: Optional[int] = None,
        prevent_server_caching: Optional[bool] = True,
        timeout: int = 60,
        top_p: int = 1,
        n: int = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
    ):
        """
        Runs asynchronous latency benchmarks across different deployments and configurations.

        This method is designed to test the latency of various deployments with different configurations. It iterates over a list of deployment names and max token counts, performing a specified number of iterations for each combination. It supports testing with different intervals between calls to the same model and calls to different models to simulate real-world usage patterns.

        :param deployment_names: A list of deployment names to test.
        :param max_tokens_list: A list of maximum token counts to use for each test.
        :param iterations: The number of times to test each deployment and token count combination. Defaults to 1.
        :param same_model_interval: The wait time in seconds between calls to the same model. Defaults to 1 second.
        :param different_model_interval: The wait time in seconds between calls to different models. Defaults to 5 seconds.
        :param temperature: Controls randomness in generation. Setting to 0 results in deterministic output. Optional.
        :param byop: A list of BYOP (Bring Your Own Prompt) configurations to test. Optional.
        :param context_tokens: The number of tokens to consider from the prompt for generating completions. Optional.
        :param prevent_server_caching: If True, prevents the server from caching the request. Defaults to True.
        :param timeout: The maximum time in seconds to wait for the call to complete. Defaults to 60 seconds.
        :param top_p: Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered. Defaults to 1.
        :param n: The number of completions to generate. Defaults to 1.
        :param presence_penalty: Adds a penalty for repeated usage of the same token in the completion. Defaults to 0.
        :param frequency_penalty: Adds a penalty for the frequency of token usage in the completion. Defaults to 0.
        """
        result = [] 
        if byop:
            iterations = len(byop)
            for deployment_name in deployment_names:
                for max_tokens in max_tokens_list:
                    for i in range(iterations):
                        current_context = context[i] if context is not None and i < len(context) else None
                        current_ground_truth = ground_truth[i] if ground_truth is not None and i < len(ground_truth) else None
                        log_system_info()
                        response = await self.make_call(
                            deployment_name=deployment_name,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            context_tokens=context_tokens,
                            prevent_server_caching=prevent_server_caching,
                            timeout=timeout,
                            prompt=byop[i],
                            context=current_context,
                            top_p=top_p,
                            n=n,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty,
                        )
                        current_result = [byop[i], current_context, response, current_ground_truth]
                        result.append(current_result)
                        await asyncio.sleep(same_model_interval)
        else:
            for deployment_name in deployment_names:
                for max_tokens in max_tokens_list:
                    for _ in range(iterations):
                        log_system_info()
                        response = await self.make_call(
                            deployment_name=deployment_name,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            context_tokens=context_tokens,
                            prevent_server_caching=prevent_server_caching,
                            context=None,
                            prompt=None,
                            timeout=timeout,
                            top_p=top_p,
                            n=n,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty,
                        )
                        result.append(response)
                        await asyncio.sleep(same_model_interval)
        await asyncio.sleep(different_model_interval)
        return result

    async def run_latency_benchmark_bulk(
        self,
        deployment_names: List[str],
        max_tokens_list: List[int],
        same_model_interval: int = 1,
        different_model_interval: int = 5,
        iterations: int = 1,
        temperature: Optional[int] = 0,
        context_tokens: Optional[int] = None,
        byop: Optional[List] = None,
        context: Optional[List] = None,
        ground_truth: Optional[List] = None,
        prevent_server_caching: Optional[bool] = True,
        timeout: int = 60,
        top_p: int = 1,
        n: int = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
    ) -> Optional[List[Any]]:
        """
        Runs latency benchmarks for multiple deployments and token counts concurrently.

        This method is designed to test the latency of various deployments with different configurations in a bulk, concurrent manner. It iterates over a list of deployment names and max token counts, performing a specified number of iterations for each combination. It supports testing with different intervals between calls to the same model and calls to different models to simulate real-world usage patterns.

        :param deployment_names: A list of deployment names to test.
        :param max_tokens_list: A list of maximum token counts to use for each test.
        :param same_model_interval: The wait time in seconds between calls to the same model. Defaults to 1 second.
        :param different_model_interval: The wait time in seconds between calls to different models. Defaults to 5 seconds.
        :param iterations: The number of times to test each deployment and token count combination. Defaults to 1.
        :param temperature: Controls randomness in generation. Setting to 0 results in deterministic output. Optional.
        :param context_tokens: The number of tokens to consider from the prompt for generating completions. Optional.
        :param byop: A list of BYOP (Bring Your Own Prompt) configurations to test. Optional.
        :param prevent_server_caching: If True, prevents the server from caching the request. Defaults to True.
        :param timeout: The maximum time in seconds to wait for the call to complete. Defaults to 60 seconds.
        :param top_p: Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered. Defaults to 1.
        :param n: The number of completions to generate. Defaults to 1.
        :param presence_penalty: Adds a penalty for repeated usage of the same token in the completion. Defaults to 0.
        :param frequency_penalty: Adds a penalty for the frequency of token usage in the completion. Defaults to 0.
        :return: An optional list of results from the benchmark tests. Each element in the list corresponds to a test result for a specific deployment and token count combination.
        """
        tasks = []
        results = []

        async def run_and_collect(*args, **kwargs):
            result = await self.run_latency_benchmark(*args, **kwargs)
            results.extend(result)  # Collect the result

        if byop:
            byop_chunks = split_list_into_variable_parts(byop)
            iterations = len(byop_chunks)
            if context is not None: 
                if ground_truth is not None:
                    if not (len(context) == len(byop) == len(ground_truth)):
                        raise ValueError("context, byop, and ground_truth must have the same number of elements.")
                    else: 
                        context_chunks = split_list_into_variable_parts(context)
                else:
                    raise ValueError("context, byop, and ground_truth must have the same number of elements.")
            if ground_truth:
                ground_truth_chunks = split_list_into_variable_parts(ground_truth)
          
            for deployment_name in deployment_names:
                for max_tokens in max_tokens_list:
                    for i in range(iterations):
                        tasks.append(
                            run_and_collect(
                                deployment_names=[deployment_name],
                                max_tokens_list=[max_tokens],
                                iterations=iterations,
                                same_model_interval=same_model_interval,
                                different_model_interval=different_model_interval,
                                temperature=temperature,
                                byop=byop_chunks[i],
                                context=context_chunks[i] if context else None,
                                ground_truth=ground_truth_chunks[i] if ground_truth else None,
                                context_tokens=context_tokens,
                                prevent_server_caching=prevent_server_caching,
                                timeout=timeout,
                                top_p=top_p,
                                n=n,
                                presence_penalty=presence_penalty,
                                frequency_penalty=frequency_penalty,
                            )
                        )
        else:
            tasks = [
                run_and_collect(
                    deployment_names=[deployment_name],
                    max_tokens_list=[max_tokens],
                    iterations=iterations,
                    same_model_interval=same_model_interval,
                    different_model_interval=different_model_interval,
                    temperature=temperature,
                    byop=None,
                    context=None,
                    context_tokens=context_tokens,
                    prevent_server_caching=prevent_server_caching,
                    timeout=timeout,
                    top_p=top_p,
                    n=n,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                )
                for deployment_name in deployment_names
                for max_tokens in max_tokens_list
            ]

        await asyncio.gather(*tasks)
        return results

    def calculate_and_show_statistics(self, show_descriptions: bool = False):
        """
        Calculates and displays statistics for all tests conducted.

        This method processes the results of latency tests, calculating various statistics such as average latency, minimum latency, maximum latency, and standard deviation. It can optionally display descriptions for each statistic to provide more context to the user.

        :param show_descriptions: If True, includes descriptions for each statistic alongside their values. Defaults to False.
        """
        stats = {
            key: self._calculate_statistics(data) for key, data in self.results.items()
        }

        headers = [
            "Model_MaxTokens",
            "is_Streaming",
            "Iterations",
            "Regions",
            "Average TTLT",
            "Median TTLT",
            "IQR TTLT",
            "95th Percentile TTLT",
            "99th Percentile TTLT",
            "CV TTLT",
            "Median Prompt Tokens",
            "IQR Prompt Tokens",
            "Median Completion Tokens",
            "IQR Completion Tokens",
            "95th Percentile Completion Tokens",
            "99th Percentile Completion Tokens",
            "CV Completion Tokens",
            "Average TBT",
            "Median TBT",
            "IQR TBT",
            "95th Percentile TBT",
            "99th Percentile TBT",
            "Average TTFT",
            "Median TTFT",
            "IQR TTFT",
            "95th Percentile TTFT",
            "99th Percentile TTFT",
            "Error Rate",
            "Error Types",
            "Successful Runs",
            "Unsuccessful Runs",
            "Throttle Count",
            "Throttle Rate",
            "Best Run",
            "Worst Run",
        ]

        descriptions = [
            "The maximum number of tokens that the model can handle.",
            "Whether the test was conducted in streaming mode.",
            "The number of times the test was run.",
            "The geographical regions where the tests were conducted.",
            "The average value of time taken for all successful runs.",
            "The middle value of time taken for all runs. This is a measure of central tendency.",
            "The interquartile range (IQR) of time taken. This is a measure of statistical dispersion, being equal to the difference between the 75th and 25th percentiles. IQR is a measure of variability and can help identify outliers.",
            "95% of the times taken are less than this value. This is another way to understand the distribution of values.",
            "99% of the times taken are less than this value. This is used to understand the distribution of values, particularly for identifying and handling outliers.",
            "The coefficient of variation of time taken. This is a normalized measure of the dispersion of the distribution. It's useful when comparing the degree of variation from one data series to another, even if the means are drastically different from each other.",
            "The middle value of the number of prompt tokens in all successful runs.",
            "The interquartile range of the number of prompt tokens. This can help identify if the number of prompt tokens varies significantly in the tests.",
            "The middle value of the number of completion tokens in all successful runs.",
            "The interquartile range of the number of completion tokens. This can help identify if the number of completion tokens varies significantly in the tests.",
            "95% of the completion tokens counts are less than this value.",
            "99% of the completion tokens counts are less than this value.",
            "The coefficient of variation of the number of completion tokens. This can help identify if the number of completion tokens varies significantly in the tests.",
            "Average Time Between Tokens (ms)",
            "Median Time Between Tokens (ms)",
            "Interquartile Range for Time Between Tokens",
            "95th Percentile of Time Between Tokens",
            "99th Percentile of Time Between Tokens",
            "Average Time To First Token (ms)",
            "Median Time To First Token (ms)",
            "Interquartile Range for Time To First Token",
            "95th Percentile of Time To First Token",
            "99th Percentile of Time To First Token",
            "The proportion of tests that resulted in an error.",
            "The types of errors that occurred during the tests.",
            "The number of tests that were successful.",
            "The number of tests that were not successful.",
            "The number of times the test was throttled or limited.",
            "The rate at which the test was throttled or limited.",
            "Details of the run with the best (lowest) time.",
            "Details of the run with the worst (highest) time.",
        ]

        table = []
        for key, data in stats.items():
            regions = data.get("regions", [])
            regions = [r for r in regions if r is not None]
            region_string = ", ".join(set(regions)) if regions else "N/A"
            row = [
                key,
                data.get("is_Streaming", "N/A"),
                data.get("number_of_iterations", "N/A"),
                region_string,
                data.get("average_ttlt", "N/A"),
                data.get("median_ttlt", "N/A"),
                data.get("iqr_ttlt", "N/A"),
                data.get("percentile_95_ttlt", "N/A"),
                data.get("percentile_99_ttlt", "N/A"),
                data.get("cv_ttlt", "N/A"),
                data.get("median_prompt_tokens", "N/A"),
                data.get("iqr_prompt_tokens", "N/A"),
                data.get("median_completion_tokens", "N/A"),
                data.get("iqr_completion_tokens", "N/A"),
                data.get("percentile_95_completion_tokens", "N/A"),
                data.get("percentile_99_completion_tokens", "N/A"),
                data.get("cv_completion_tokens", "N/A"),
                data.get("average_tbt", "N/A"),
                data.get("median_tbt", "N/A"),
                data.get("iqr_tbt", "N/A"),
                data.get("percentile_95_tbt", "N/A"),
                data.get("percentile_99_tbt", "N/A"),
                data.get("average_ttft", "N/A"),
                data.get("median_ttft", "N/A"),
                data.get("iqr_ttft", "N/A"),
                data.get("percentile_95_ttft", "N/A"),
                data.get("percentile_99_ttft", "N/A"),
                data.get("error_rate", "N/A"),
                data.get("errors_types", "N/A"),
                data.get("successful_runs", "N/A"),
                data.get("unsuccessful_runs", "N/A"),
                data.get("throttle_count", "N/A"),
                data.get("throttle_rate", "N/A"),
                json.dumps(data.get("best_run", {})) if data.get("best_run") else "N/A",
                json.dumps(data.get("worst_run", {}))
                if data.get("worst_run")
                else "N/A",
            ]
            table.append(row)

        try:
            table.sort(key=lambda x: x[3])
        except Exception as e:
            logger.warning(f"An error occurred while sorting the table: {e}")

        if show_descriptions:
            for header, description in zip(headers, descriptions):
                print(colored(header, "blue"))
                print(description)

        print(tabulate(table, headers, tablefmt="pretty"))
        return stats

    @staticmethod
    def save_statistics_to_file(stats: Dict, location: str):
        """
        Saves the collected statistics to a specified file in JSON format.

        This method takes a dictionary of statistics and writes it to a file at the specified location in JSON format. It is useful for persisting the results of latency tests or other statistical analyses for later review or processing.

        :param stats: A dictionary containing the statistics to be saved. The keys should be the names of the statistics, and the values should be their corresponding values.
        :param location: The file path where the statistics should be saved. If the file does not exist, it will be created. If it does exist, it will be overwritten.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(location), exist_ok=True)

        with open(location, "w") as f:
            json.dump(stats, f, indent=2)

    def _store_results(
        self, deployment_name: str, max_tokens: int, headers: Dict, time_taken=None
    ):
        """
        Stores the results of API calls for latency testing.

        This method is responsible for recording the outcomes of individual API calls made during latency testing. It captures various details such as the deployment name, the maximum number of tokens requested, the headers sent with the request, and the time taken for the call to complete. This information is crucial for analyzing the performance and reliability of the API under test.

        :param deployment_name: The name of the deployment to which the API call was made.
        :param max_tokens: The maximum number of tokens that were requested in the API call.
        :param headers: A dictionary containing the headers that were sent with the API call.
        :param time_taken: The duration, in seconds, that the API call took to complete. This parameter is optional and should be provided if the call was successful.
        """
        key = f"{deployment_name}_{max_tokens}"

        if key not in self.results:
            self.results[key] = {
                "ttlt_successfull": [],
                "ttlt_unsucessfull": [],
                "tbt": [],
                "ttft": [],
                "regions": [],
                "number_of_iterations": 0,
                "completion_tokens": [],
                "prompt_tokens": [],
                "errors": {"count": 0, "codes": []},
                "best_run": {
                    "ttlt": float("inf"),
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                },
                "worst_run": {
                    "ttlt": float("-inf"),
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                },
            }

        if time_taken is not None:
            self.results[key]["number_of_iterations"] += 1
            self.results[key]["ttlt_successfull"].append(time_taken)
            self.results[key]["completion_tokens"].append(headers["completion_tokens"])
            self.results[key]["prompt_tokens"].append(headers["prompt_tokens"])
            self.results[key]["regions"].append(headers["region"])
            self.results[key]["tbt"].append(headers.get("tbt"))
            self.results[key]["ttft"].append(headers.get("ttft"))

            current_run = {
                "ttlt": round(time_taken, 2),
                "completion_tokens": headers.get("completion_tokens"),
                "prompt_tokens": headers.get("prompt_tokens"),
                "region": headers["region"],
                "utilization": headers["utilization"],
                "local_time": get_local_time_in_azure_region(headers["region"]),
            }

            # Update best and worst runs
            if time_taken < self.results[key]["best_run"]["ttlt"]:
                self.results[key]["best_run"] = current_run
            if time_taken > self.results[key]["worst_run"]["ttlt"]:
                self.results[key]["worst_run"] = current_run
        else:
            self._handle_error(deployment_name, max_tokens, None, "-99")

    def _handle_error(
        self, deployment_name: str, max_tokens: int, time_taken: int, code: str
    ):
        """
        Handles errors encountered during API calls and logs the details.

        This method is invoked when an API call results in an error. It records the error details, including the deployment name, the maximum number of tokens requested, the time taken for the call, and the response received from the API. This information is crucial for diagnosing issues and improving the reliability and performance of the API.

        :param deployment_name: The name of the deployment to which the API call was made.
        :param max_tokens: The maximum number of tokens that were requested in the API call.
        :param time_taken: The duration, in seconds, that the API call took to complete.
        :param response: The response object received from the API call. This includes any error messages or codes returned by the API.
        """
        key = f"{deployment_name}_{max_tokens}"
        if key not in self.results:
            self.results[key] = {
                "ttlt_successfull": [],
                "ttlt_unsucessfull": [],
                "tbt": [],
                "ttft": [],
                "regions": [],
                "number_of_iterations": 0,
                "completion_tokens": [],
                "prompt_tokens": [],
                "errors": {"count": 0, "codes": []},
                "best_run": {
                    "ttlt": float("inf"),
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                },
                "worst_run": {
                    "ttlt": float("-inf"),
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                },
            }
        self.results[key]["errors"]["count"] += 1
        self.results[key]["number_of_iterations"] += 1
        self.results[key]["ttlt_unsucessfull"].append(time_taken)
        if code == "-99":
            logger.error("Error during API call: No captured time, test client error")
            self.results[key]["errors"]["codes"].append("-99")
        elif code == "-100":
            logger.error(
                "Timeout error during API call: The request took too long to complete."
            )
            self.results[key]["errors"]["codes"].append("-100")
        else:
            self.results[key]["errors"]["codes"].append(code)

    def _calculate_statistics(self, data: Dict) -> Dict:
        """
        Calculates statistical metrics based on the provided test data.

        This method processes the test data collected during API performance tests to calculate various statistical metrics. These metrics include average response time, minimum and maximum response times, standard deviation, and potentially other relevant statistics such as percentiles. The calculated statistics are returned in a dictionary, providing a comprehensive overview of the API's performance characteristics.

        :param data: A dictionary containing the test data, typically including response times and possibly error rates or other relevant metrics.
        :return: A dictionary containing the calculated statistical metrics, offering insights into the performance and reliability of the tested API.
        """
        total_requests = data["number_of_iterations"]
        ttlts = list(filter(None, data.get("ttlt_successfull", [])))
        completion_tokens = list(filter(None, data.get("completion_tokens", [])))
        prompt_tokens = list(filter(None, data.get("prompt_tokens", [])))
        tbt = data.get("tbt")
        if isinstance(tbt, list):
            if all(isinstance(sublist, list) for sublist in tbt):
                tbt = [item for sublist in tbt for item in sublist]
        else:
            tbt = list(filter(None, tbt))
        ttft = data.get("ttft")
        if ttft is not None:
            ttft = list(filter(None, ttft))
        error_count = data["errors"]["count"]
        error_codes = data["errors"]["codes"]
        error_distribution = {
            str(code): error_codes.count(code) for code in set(error_codes)
        }
        count_throttle = error_distribution.get("429", 0)
        successful_runs = len(data["ttlt_successfull"])
        unsuccessful_runs = len(data["ttlt_unsucessfull"])

        stats = {
            "median_ttlt": None,
            "is_Streaming": self.is_streaming,
            "regions": list(set(data.get("regions", []))),
            "iqr_ttlt": None,
            "percentile_95_ttlt": None,
            "percentile_99_ttlt": None,
            "cv_ttlt": None,
            "median_completion_tokens": None,
            "iqr_completion_tokens": None,
            "percentile_95_completion_tokens": None,
            "percentile_99_completion_tokens": None,
            "cv_completion_tokens": None,
            "median_prompt_tokens": None,
            "iqr_prompt_tokens": None,
            "percentile_95_prompt_tokens": None,
            "percentile_99_prompt_tokens": None,
            "cv_prompt_tokens": None,
            "average_ttlt": round(sum(ttlts) / len(ttlts), 2) if ttlts else None,
            "error_rate": error_count / total_requests if total_requests > 0 else 0,
            "number_of_iterations": total_requests,
            "throttle_count": count_throttle,
            "throttle_rate": count_throttle / total_requests
            if total_requests > 0
            else 0,
            "errors_types": data.get("errors", {}).get("codes", []),
            "successful_runs": successful_runs,
            "unsuccessful_runs": unsuccessful_runs,
            "median_tbt": None,
            "iqr_tbt": None,
            "percentile_95_tbt": None,
            "percentile_99_tbt": None,
            "cv_tbt": None,
            "average_tbt": round(sum(tbt) / len(tbt), 2) if tbt else None,
            "median_ttft": None,
            "iqr_ttft": None,
            "percentile_95_ttft": None,
            "percentile_99_ttft": None,
            "cv_ttft": None,
            "average_ttft": round(sum(ttft) / len(ttft), 2) if ttft else None,
        }

        if ttlts:
            stats.update(
                zip(
                    [
                        "median_ttlt",
                        "iqr_ttlt",
                        "percentile_95_ttlt",
                        "percentile_99_ttlt",
                        "cv_ttlt",
                    ],
                    calculate_statistics(ttlts),
                )
            )

        if completion_tokens:
            stats.update(
                zip(
                    [
                        "median_completion_tokens",
                        "iqr_completion_tokens",
                        "percentile_95_completion_tokens",
                        "percentile_99_completion_tokens",
                        "cv_completion_tokens",
                    ],
                    calculate_statistics(completion_tokens),
                )
            )

        if prompt_tokens:
            stats.update(
                zip(
                    [
                        "median_prompt_tokens",
                        "iqr_prompt_tokens",
                        "percentile_95_prompt_tokens",
                        "percentile_99_prompt_tokens",
                        "cv_prompt_tokens",
                    ],
                    calculate_statistics(prompt_tokens),
                )
            )

        if tbt:
            stats.update(
                zip(
                    [
                        "median_tbt",
                        "iqr_tbt",
                        "percentile_95_tbt",
                        "percentile_99_tbt",
                        "cv_tbt",
                    ],
                    calculate_statistics(tbt),
                )
            )

        if ttft:
            stats.update(
                zip(
                    [
                        "median_ttft",
                        "iqr_ttft",
                        "percentile_95_ttft",
                        "percentile_99_ttft",
                        "cv_ttft",
                    ],
                    calculate_statistics(ttft),
                )
            )

        stats["best_run"] = data.get("best_run", {})
        stats["worst_run"] = data.get("worst_run", {})

        return stats

    @staticmethod
    def generate_test_messages(
        model_name: str,
        prompt: str = None,
        context: str = None,
        context_tokens: int = None,
        prevent_server_caching: bool = False,
        max_tokens: int = None,
    ) -> List[Dict[str, str]]:
        """
        This method is designed to prepare a set of messages for latency testing of a language model API. It can either use a specific prompt provided by the user or generate random messages if no prompt is given. The method allows for the simulation of context by specifying a number of tokens, and it can also prevent server-side caching of requests by adding random prefixes to the messages. The maximum number of tokens to be generated can be specified as well.

        :param model_name: The name of the model for which messages are being generated.
        :param prompt: An optional parameter that allows the user to specify a prompt for message generation. If not provided, messages will be generated randomly.
        :param context_tokens: An optional parameter to specify the number of tokens to simulate as context for the message. If not provided, a default value is used.
        :param prevent_server_caching: A boolean parameter that, when set to True, adds random prefixes to messages to prevent server-side caching of the requests.
        :param max_tokens: An optional parameter that specifies the maximum number of tokens to generate for each message.
        :return: A list of dictionaries, each containing a generated message. The messages are ready to be used for latency testing.
        """
        # Set default context tokens if not provided
        if context_tokens is None:
            logger.info(
                "As no context was provided, 1000 tokens were added as average workloads."
            )
            context_tokens = 1000

        if prompt is None:
            random = RandomMessagesGenerator(
                model=model_name,
                prevent_server_caching=prevent_server_caching,
                tokens=context_tokens,
                max_tokens=max_tokens,
            )
            messages, messages_tokens_count = random.generate_messages()
        else:
            generator = BYOPMessageGenerator(
                model=model_name,
                prevent_server_caching=prevent_server_caching,
                max_tokens=max_tokens,
            )
            if context:
                messages, messages_tokens_count = generator.generate_messages(
                    prompt, context
                )
            else: 
                messages, messages_tokens_count = generator.generate_messages(prompt)

        return messages, messages_tokens_count


class AzureOpenAIBenchmarkNonStreaming(AzureOpenAIBenchmarkLatency):
    def __init__(self, api_key, azure_endpoint, api_version="2024-02-15-preview"):
        """
        This class is designed for conducting latency benchmarks on the Azure OpenAI service without using streaming capabilities. It extends a base class that handles general latency benchmarking tasks. The constructor initializes the class with necessary details such as the API key, endpoint, and API version. It also initializes a dictionary to store results and sets a flag indicating that streaming is not used in this benchmark.

        :param api_key: The API key required to authenticate requests to the Azure OpenAI service.
        :param azure_endpoint: The endpoint URL of the Azure OpenAI service.
        :param api_version: The version of the API to use for requests. Defaults to "2024-02-15-preview".
        """
        super().__init__(api_key, azure_endpoint, api_version)
        self.results = {}
        self.is_streaming = False

    async def make_call(
        self,
        deployment_name: str,
        max_tokens: int,
        temperature: Optional[int] = 0,
        timeout: int = 120,
        prompt: Optional[str] = None,
        context: Optional[str] = None,
        context_tokens: Optional[int] = None,
        prevent_server_caching: Optional[bool] = True,
        top_p: int = 1,
        n: int = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
    ):
        """
        Asynchronously makes a chat completion call to the Azure OpenAI API and logs the time taken for the call.

        This method is designed to handle asynchronous API calls to the Azure OpenAI service, utilizing exponential backoff with full jitter for retry logic in case of client errors. It supports a wide range of parameters to customize the chat completion request, including the ability to prevent server caching, control diversity via nucleus sampling, and adjust the likelihood of new words based on their presence or frequency in the text so far.

        :param deployment_name: Name of the model deployment to use for generating completions.
        :param max_tokens: Maximum number of tokens to generate in the completion.
        :param temperature: Controls randomness in generation, with 0 being deterministic. Defaults to 0.0.
        :param timeout: Timeout for the API call in seconds. Defaults to 120 seconds to accommodate longer processing times.
        :param prompt: Initial text to generate completions for. If None, an empty string is assumed.
        :param context_tokens: Number of context tokens to consider for generating completions. If not provided, a default value is used.
        :param prevent_server_caching: If True, modifies the prompt in a way to prevent server-side caching of the request. Defaults to True.
        :param top_p: Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered. Defaults to 1.0.
        :param n: Number of completions to generate for each prompt. Defaults to 1.
        :param presence_penalty: Adjusts the likelihood of new words based on their presence in the text so far. Defaults to 0.0.
        :param frequency_penalty: Adjusts the likelihood of new words based on their frequency in the text so far. Defaults to 0.0.
        :param stop_sequences: A list of strings where the API should stop generating further tokens. Useful for defining natural endpoints in generated text.
        :param return_prompt: If True, the response includes the prompt along with the generated completion. Defaults to False.

        The method handles client errors by retrying the request using an exponential backoff strategy with full jitter to mitigate the impact of retry storms. It gives up on retrying if the error code is terminal (400, 401, 403, 404, 429, 500), indicating that further attempts are unlikely to succeed.
        """

        url = f"{self.azure_endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={self.api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
            TELEMETRY_USER_AGENT_HEADER: USER_AGENT,
        }
        model_name, _ = detect_model_encoding(deployment_name)
        messages, _ = self.generate_test_messages(
            model_name, prompt, context, context_tokens, prevent_server_caching, max_tokens
        )

        body = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "messages": messages,
        }

        logger.info(
            f"Initiating call for Model: {deployment_name}, Max Tokens: {max_tokens}"
        )
        async with aiohttp.ClientSession() as session:
            start_time = time.perf_counter()
            try:
                response = await session.post(
                    url, headers=headers, json=body, timeout=timeout
                )
                end_time = time.perf_counter()
                time_taken = end_time - start_time
                if response.status != 200:
                    if response.status == 429:
                        if RETRY_AFTER_MS_HEADER in response.headers:
                            retry_after_str = response.headers[RETRY_AFTER_MS_HEADER]
                            retry_after_ms = float(retry_after_str)
                            logger.info(f"retry-after sleeping for {retry_after_ms}ms")
                            await asyncio.sleep(retry_after_ms / 1000.0)
                    logger.error(
                        f"Unsuccessful Run - Error {response.status}: {response.text} - Time taken: {time_taken:.2f} seconds. Traceback: {traceback.format_exc()}"
                    )
                    self._handle_error(
                        deployment_name,
                        max_tokens,
                        round(time_taken, 2),
                        response.status,
                    )
                    return None
                else:
                    response_headers = response.headers
                    response_body = await response.json()
                    headers = extract_rate_limit_and_usage_info_async(
                        response_headers, response_body
                    )
                    headers["tbt"] = round(
                        time_taken / headers.get("completion_tokens", 1), 2
                    )
                    headers["ttft"] = round(time_taken, 2)
                    self._store_results(
                        deployment_name, max_tokens, headers, time_taken
                    )
                    logger.info(
                        f"Succesful Run - Time taken: {time_taken:.2f} seconds."
                    )
                    return response_body["choices"][0]["message"]["content"]
            except aiohttp.ClientError as e:
                end_time = time.perf_counter()
                time_taken = end_time - start_time
                logger.error(f"Error during API call: {str(e)}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                # FIXME: This is a temporary fix to handle the case where the status code is not available in the exception object
                status_code = getattr(e, "status", -99)
                self._handle_error(
                    deployment_name,
                    max_tokens,
                    round(time_taken, 2),
                    type(e).__name__,
                    status_code,
                )
                logger.error(
                    f"Unsuccessful Run - Error {type(e).__name__}: {str(e)} - Time taken: {time_taken:.2f} seconds. Traceback: {traceback.format_exc()}"
                )
                return None
            except asyncio.TimeoutError:
                end_time = time.perf_counter()
                time_taken = end_time - start_time
                logger.error(f"Timeout error after {time_taken:.2f} seconds.")
                self._handle_error(deployment_name, max_tokens, time_taken, "-100")
                return None
            except Exception as e:
                end_time = time.perf_counter()
                time_taken = end_time - start_time
                logger.error(f"Unexpected error: {str(e)}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                self._handle_error(
                    deployment_name, max_tokens, round(time_taken, 2), type(e).__name__
                )
                logger.error(
                    f"Unsuccessful Run - Error {type(e).__name__}: {str(e)} - Time taken: {time_taken:.2f} seconds. Traceback: {traceback.format_exc()}"
                )
                return None


class AzureOpenAIBenchmarkStreaming(AzureOpenAIBenchmarkLatency):
    def __init__(self, api_key, azure_endpoint, api_version="2024-02-15-preview"):
        """
        Initializes the AzureOpenAIBenchmarkStreaming class, which is designed to conduct latency benchmarks on the Azure OpenAI service using streaming capabilities. This class extends AzureOpenAIBenchmarkLatency, inheriting its setup and functionality but with modifications to support streaming.

        Streaming is particularly useful for scenarios where real-time interaction with the model is required, such as chat applications or where the response from the model needs to be processed as it arrives rather than after the entire response is received.

        :param api_key: The API key required to authenticate requests to the Azure OpenAI service. This key should have the necessary permissions to access the API and perform requests.
        :param azure_endpoint: The endpoint URL of the Azure OpenAI service. This URL is specific to the Azure resource you have created and is used to direct requests to the correct service instance.
        :param api_version: The version of the API to use for requests. This parameter allows for specifying the version of the API, enabling the use of newer or older versions as needed. Defaults to "2024-02-15-preview", which is a placeholder for the latest version available at the time of writing.

        The constructor initializes the base class with the provided parameters and sets an additional property, `is_streaming`, to True. This property is used to differentiate between streaming and non-streaming benchmark tests and to enable specific logic required for handling streaming responses.
        """
        super().__init__(api_key, azure_endpoint, api_version)
        self.is_streaming = True

    async def make_call(
        self,
        deployment_name: str,
        max_tokens: int,
        temperature: Optional[int] = 0,
        timeout: int = 60,
        context_tokens: Optional[int] = None,
        prompt: Optional[str] = None,
        context: Optional[str] = None,
        prevent_server_caching: Optional[bool] = True,
        top_p: int = 1,
        n: int = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
    ) -> Optional[str]:
        url = f"{self.azure_endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={self.api_version}"

        if context_tokens is None:
            logger.info(
                "As no context was provided, 1000 tokens were added as average workloads."
            )
            context_tokens = 1000

        model_name, _ = detect_model_encoding(deployment_name)
        messages, context_num_tokens = self.generate_test_messages(
            model_name, prompt, context, context_tokens, prevent_server_caching, max_tokens
        )
        logger.info(f"Messages: {messages} and Context Tokens: {context_num_tokens}")

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
            "x-ms-useragent": "latency-benchmark",
        }

        body = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "messages": messages,
            "stream": True,
        }

        encoding = get_encoding_from_model_name(deployment_name)
        final_text_response = ""
        region = None

        logger.info(
            f"Starting call to model {deployment_name} with max tokens {max_tokens} at (Local time): {datetime.now()}, (GMT): {datetime.now(timezone.utc)}"
        )

        async with aiohttp.ClientSession() as session:
            start_time = time.perf_counter()
            try:
                async with session.post(
                    url, headers=headers, json=body, timeout=timeout
                ) as response:
                    if region is None:
                        region = response.headers.get("x-ms-region", "N/A")
                    if response.status != 200:
                        await self.handle_error_response(
                            response, deployment_name, max_tokens, start_time
                        )
                        return None

                    prev_end_time = start_time
                    first_token_time = None
                    token_times = []

                    async for line in response.content:
                        chunk = line.decode("utf-8").rstrip()
                        if chunk and chunk.startswith("data: "):
                            json_string = chunk[6:]
                            if json_string:
                                try:
                                    data = json.loads(json_string)
                                    if "choices" in data and data["choices"]:
                                        event_text = (
                                            data["choices"][0]
                                            .get("delta", {})
                                            .get("content", "")
                                        )
                                        if event_text:
                                            end_time = time.perf_counter()
                                            time_taken = end_time - prev_end_time
                                            prev_end_time = end_time
                                            if first_token_time is None:
                                                first_token_time = time_taken
                                            token_times.append(time_taken)
                                            final_text_response += event_text
                                            utilization = response.headers.get(
                                                "azure-openai-deployment-utilization",
                                                "N/A",
                                            )
                                except json.JSONDecodeError as e:
                                    logger.error(f"Error decoding JSON: {e}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                await self.handle_exception(e, deployment_name, max_tokens, start_time)
                return None

        end_time = time.perf_counter()
        await self.process_response(
            end_time,
            deployment_name,
            max_tokens,
            final_text_response,
            token_times,
            context_num_tokens,
            start_time,
            region,
            encoding,
            utilization,
        )
        return final_text_response

    async def handle_error_response(
        self,
        response: aiohttp.ClientResponse,
        deployment_name: str,
        max_tokens: int,
        start_time: float,
    ):
        """
        Handles errors that occur during the API call.

        :param response: The aiohttp ClientResponse object.
        :param deployment_name: The name of the deployment.
        :param max_tokens: The maximum number of tokens.
        :param start_time: The start time of the API call.
        """
        end_time = time.perf_counter()
        time_taken = end_time - start_time
        if response.status == 429:
            retry_after_str = response.headers.get(RETRY_AFTER_MS_HEADER, "6000")
            retry_after_ms = float(retry_after_str)
            logger.debug(
                f"429 Too Many Requests: retry-after sleeping for {retry_after_ms}ms"
            )
            await asyncio.sleep(retry_after_ms / 1000.0)
        logger.error(f"Error during API call: {await response.text()}")
        logger.error(f"Exception type: {response.status}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.error(
            f"APIM Request ID: {response.headers.get('apim-request-id', 'N/A')}"
        )
        self._handle_error(deployment_name, max_tokens, time_taken, response)

    async def handle_exception(
        self,
        exception: Exception,
        deployment_name: str,
        max_tokens: int,
        start_time: float,
    ):
        """
        Handles exceptions that occur during the API call.

        :param exception: The exception that occurred.
        :param deployment_name: The name of the deployment.
        :param max_tokens: The maximum number of tokens.
        :param start_time: The start time of the API call.
        """
        end_time = time.perf_counter()
        time_taken = end_time - start_time
        if isinstance(exception, asyncio.TimeoutError):
            logger.error(f"Timeout error after {time_taken:.2f} seconds.")
            self._handle_error(deployment_name, max_tokens, time_taken, "-100")
        elif isinstance(exception, aiohttp.ClientError):
            logger.error(f"Client error: {str(exception)}")
            logger.error(f"Exception type: {type(exception).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        
            http_status_code = "-99"
            retry_after = None
        
            # Check if the exception has the status and headers attributes
            if hasattr(exception, 'status'):
                http_status_code = exception.status
            if hasattr(exception, 'headers') and 'Retry-After' in exception.headers:
                retry_after = exception.headers['Retry-After']
                
            logger.info(f"Retry-After: {retry_after}")
    
            # Pass the extracted information to _handle_error
            self._handle_error(
                deployment_name, max_tokens, time_taken, http_status_code
            )
        else:
            logger.error(f"Unexpected error: {str(exception)}")
            logger.error(f"Exception type: {type(exception).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._handle_error(
                deployment_name, max_tokens, time_taken, type(exception).__name__
            )

    async def process_response(
        self,
        end_time: float,
        deployment_name: str,
        max_tokens: int,
        final_text_response: str,
        token_times: List[float],
        context_num_tokens: int,
        start_time: float,
        region: Optional[str],
        encoding: Any,
        utilization: Optional[str],
    ):
        """
        Processes the API response and logs the results.

        :param deployment_name: The name of the deployment.
        :param max_tokens: The maximum number of tokens.
        :param final_text_response: The final text response from the API.
        :param token_times: The list of token generation times.
        :param context_num_tokens: The number of context tokens.
        :param start_time: The start time of the API call.
        :param region: The region of the API call.
        :param encoding: The encoding used for the model.
        """
        total_time_taken = end_time - start_time
        token_times = [
            max(round(time_taken * 1000, 2), 1) / 1000 for time_taken in token_times
        ]  # Convert to seconds
        token_generation_count = count_tokens(encoding, final_text_response)
        logger.info(
            f"Finished call to model {deployment_name}. Time taken for chat: {round(total_time_taken, 2)} seconds or {round(total_time_taken * 1000, 2)} milliseconds."
        )

        if token_times:
            TBT = token_times[1:]
            TTFT = token_times[0] if token_times else None

            headers_dict = {
                "completion_tokens": token_generation_count,
                "prompt_tokens": context_num_tokens,
                "region": region,
                "utilization": utilization,
                "tbt": TBT,
                "ttft": round(TTFT, 2) if TTFT else None,
            }
            self._store_results(
                deployment_name, max_tokens, headers_dict, round(total_time_taken, 2)
            )
        else:
            self._handle_error(deployment_name, max_tokens, None, "-99")

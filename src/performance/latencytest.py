import asyncio
import json
import os
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import backoff
import tiktoken


import aiohttp
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AzureOpenAI
from tabulate import tabulate
from termcolor import colored
import regex as re

from src.performance.aoaihelpers.utils import (
    calculate_statistics, extract_rate_limit_and_usage_info_async,
    get_local_time_in_azure_region, log_system_info)
from src.performance.messagegeneration import RandomMessagesGenerator
from src.performance.utils import terminal_http_code, count_tokens, get_encoding_from_model_name
from utils.ml_logging import get_logger

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
        Base class for AzureOpenAIBenchmark with the API key, API version, and endpoint.
        """
        self.api_key = api_key or os.getenv("AZURE_OPENAI_KEY")
        self.api_version = (
            api_version or os.getenv("AZURE_OPENAI_API_VERSION") or "2023-05-15"
        )
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_API_ENDPOINT")
        self._validate_api_configurations()
        self.results = {}
        self.is_streaming = "N/A"

    def _validate_api_configurations(self):
        """
        Validates if all necessary configurations are set.
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
        timeout: int = 60,
        context_tokens: Optional[int] = None,
        prevent_server_caching: Optional[bool] = True,
        top_p: int = 1,
        n: int = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
    ):
        """
        Make an asynchronous chat completion call and log the time taken for the call.
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
        context_tokens: Optional[int] = None,
        multiregion: bool = False,
        prevent_server_caching: Optional[bool] = True,
        timeout: int = 60,
        top_p: int = 1,
        n: int = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
    ):
        """
        Run asynchronous tests across different deployments and token counts.
        """
        self.by_region = multiregion
        for deployment_name in deployment_names:
            for max_tokens in max_tokens_list:
                for _ in range(iterations):
                    log_system_info()
                    await self.make_call(
                        deployment_name,
                        max_tokens,
                        temperature,
                        context_tokens,
                        prevent_server_caching,
                        timeout=timeout,
                        top_p=top_p,
                        n=n,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                    )
                    await asyncio.sleep(same_model_interval)
            await asyncio.sleep(different_model_interval)

    async def run_latency_benchmark_bulk(
        self,
        deployment_names: List[str],
        max_tokens_list: List[int],
        same_model_interval: int = 1,
        different_model_interval: int = 5,
        iterations: int = 1,
        temperature: Optional[int] = 0,
        context_tokens: Optional[int] = None,
        multiregion: bool = False,
        prevent_server_caching: Optional[bool] = True,
        timeout: int = 60,
        top_p: int = 1,
        n: int = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
    ) -> Optional[List[Any]]:
        """
        Run latency benchmarks for multiple deployments and token counts concurrently.
        """
        tasks = [
            self.run_latency_benchmark(
                [deployment_name],
                [max_tokens],
                iterations,
                same_model_interval,
                different_model_interval,
                temperature,
                context_tokens,
                multiregion,
                prevent_server_caching,
                timeout,
                top_p,
                n,
                presence_penalty,
                frequency_penalty,
            )
            for deployment_name in deployment_names
            for max_tokens in max_tokens_list
        ]

        await asyncio.gather(*tasks)

    def calculate_and_show_statistics(self, show_descriptions: bool = False):
        """
        Calculate and display statistics for all tests conducted.

        Args:
            show_descriptions (bool): Whether to show descriptions for each statistic header.
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
                json.dumps(data.get("worst_run", {})) if data.get("worst_run") else "N/A",
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
        Save the statistics to a JSON file.

        :param stats: Statistics data.
        :param location: File path to save the data.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(location), exist_ok=True)

        with open(location, "w") as f:
            json.dump(stats, f, indent=2)

    def _store_results(
        self, deployment_name: str, max_tokens: int, headers: Dict, time_taken=None
    ):
        """
        Store the results from each API call for later analysis.
        Includes handling cases where the response might be None due to failed API calls.
        """
        key = f"{deployment_name}_{max_tokens}"

        if key not in self.results:
            self.results[key] = {
                "ttlt_successful": [],
                "ttlt_unsuccessful": [],
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
            self.results[key]["ttlt_successful"].append(time_taken)
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
        self, deployment_name: str, max_tokens: int, time_taken: int, response
    ):
        """
        Handle exceptions during API calls and store error details.

        :param deployment_name: Model deployment name.
        :param max_tokens: Maximum tokens parameter for the call.
        :param response: Response from the API call.
        """
        key = f"{deployment_name}_{max_tokens}"
        if key not in self.results:
            self.results[key] = {
                "ttlt_successful": [],
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
        if response is not None:
            if response == "-99":
                logger.error(
                    "Error during API call: No captured time, test client error"
                )
                self.results[key]["errors"]["codes"].append("-99")
            self.results[key]["errors"]["codes"].append(response.status)
            logger.error(f"Error during API call: {response.text}")
        else:
            logger.error("Error during API call: Unknown error")

    def _calculate_statistics(self, data: Dict) -> Dict:
        """
        Calculate and return the statistical metrics for test results.

        :param data: Test data collected.
        :return: Dictionary of calculated statistical metrics.
        """
        total_requests = data["number_of_iterations"]
        ttlts = list(filter(None, data.get("ttlt_successful", [])))
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
        successful_runs = len(data["ttlt_successful"])
        unsuccessful_runs = len(data["ttlt_unsuccessful"])

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
            "throttle_rate": count_throttle / total_requests if total_requests > 0 else 0,
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

    

    

class AzureOpenAIBenchmarkNonStreaming(AzureOpenAIBenchmarkLatency):
    def __init__(self, api_key, azure_endpoint, api_version="2024-02-15-preview"):
        """
        Initialize the AzureOpenAIBenchmarkNonStreaming with the API key, API version, and endpoint.
        """
        super().__init__(api_key, azure_endpoint, api_version)
        self.results = {}
        self.is_streaming = False
    
    @backoff.on_exception(
        backoff.expo,
        aiohttp.ClientError,
        jitter=backoff.full_jitter,
        max_tries=MAX_RETRY_ATTEMPTS,
        giveup=terminal_http_code,
    )
    async def make_call(
        self,
        deployment_name: str,
        max_tokens: int,
        temperature: Optional[int] = 0,
        timeout: int = 120,
        context_tokens: Optional[int] = None,
        prevent_server_caching: Optional[bool] = True,
        top_p: int = 1,
        n: int = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
    ):
        """
        Make an asynchronous chat completion call to the Azure OpenAI API and log the time taken for the call.

        :param deployment_name: Name of the model deployment to use.
        :param max_tokens: Maximum number of tokens to generate.
        :param temperature: The temperature to use for the chat completion. Defaults to 0.
        :param timeout: Timeout for the API call in seconds. Updated to 120 seconds.
        :param context_tokens: Number of context tokens to use. If not provided, 1000 tokens are used as default.
        :param prevent_server_caching: Flag to indicate if server caching should be prevented. Default is True.
        :param top_p: Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered. Default is 1.
        :param n: Number of completions to generate. Default is 1.
        :param presence_penalty: Adjusts the likelihood of new words based on their presence in the text so far. Default is 0.
        :param frequency_penalty: Adjusts the likelihood of new words based on their frequency in the text so far. Default is 0.
        """
        
        url = f"{self.azure_endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={self.api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
            TELEMETRY_USER_AGENT_HEADER: USER_AGENT,
        }

        # Set default context tokens if not provided
        if context_tokens is None:
            logger.info(
                "As no context was provided, 1000 tokens were added as average workloads."
            )
            context_tokens = 1000

        random = RandomMessagesGenerator(
            model="gpt-4",
            prevent_server_caching=prevent_server_caching,
            tokens=context_tokens,
            max_tokens=max_tokens,
        )
        messages, _ = random.generate_messages()

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
            response = await session.post(url, headers=headers, json=body, timeout=timeout)
            end_time = time.perf_counter()
            time_taken = end_time - start_time
            if response.status != 200:
                if response.status == 429:
                    if RETRY_AFTER_MS_HEADER in response.headers:
                        retry_after_str = response.headers[RETRY_AFTER_MS_HEADER]
                        retry_after_ms = float(retry_after_str)
                        logger.debug(f"retry-after sleeping for {retry_after_ms}ms")
                        await asyncio.sleep(retry_after_ms / 1000.0)
                logger.error(f"Error during API call: {response.text}")
                logger.error(f"Exception type: {response.status}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                self._handle_error(deployment_name, max_tokens, round(time_taken, 2), response)
                logger.info(f"Unsuccesful Run - Time taken: {time_taken:.2f} seconds.")
            else: 
                response_headers = response.headers
                response_body = await response.json()
                headers = extract_rate_limit_and_usage_info_async(
                    response_headers, response_body
                )
                headers['tbt'] = round(time_taken / headers.get("completion_tokens", 1), 2)
                headers['ttft'] = round(time_taken, 2)
                self._store_results(deployment_name, max_tokens, headers, time_taken)
                logger.info(f"Succesful Run - Time taken: {time_taken:.2f} seconds.")
                return response_body['choices'][0]['message']['content']

                
class AzureOpenAIBenchmarkStreaming(AzureOpenAIBenchmarkLatency):
    def __init__(self, api_key, azure_endpoint, api_version="2024-02-15-preview"):
        """
        Initialize the AzureOpenAILatencyBenchmark with the API key, API version, and endpoint.
        """
        super().__init__(api_key, azure_endpoint, api_version)
        self.is_streaming = True
   
    @backoff.on_exception(
        backoff.expo,
        aiohttp.ClientError,
        jitter=backoff.full_jitter,
        max_tries=MAX_RETRY_ATTEMPTS,
        giveup=terminal_http_code,
    )
    async def make_call(
        self,
        deployment_name: str,
        max_tokens: int,
        temperature: Optional[int] = 0,
        timeout: int = 60,
        context_tokens: Optional[int] = None,
        prevent_server_caching: Optional[bool] = True,
        top_p: int = 1,
        n: int = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
    ) -> Optional[str]:
        """
        Make a chat completion call and print the time taken for the call.
        """
        url = f"{self.azure_endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={self.api_version}"
        if context_tokens is None:
            logger.info("As no context was provided, 1000 tokens were added as average workloads.")
            context_tokens = 1000

        random = RandomMessagesGenerator(
            model="gpt-4",
            prevent_server_caching=prevent_server_caching,
            tokens=context_tokens,
            max_tokens=max_tokens,
        )
        messages, context_num_tokens = random.generate_messages()
        logger.info(f"Messages: {messages} and Context Tokens: {context_num_tokens}")

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
            TELEMETRY_USER_AGENT_HEADER: USER_AGENT,
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

        logger.info(f"Starting call to model {deployment_name} with max tokens {max_tokens} at (Local time): {datetime.now()}, (GMT): {datetime.now(timezone.utc)}")
        async with aiohttp.ClientSession() as session:
            start_time = time.perf_counter()
            try:
                async with session.post(url, headers=headers, json=body, timeout=timeout) as response:
                    prev_end_time = start_time
                    token_generation_count = 0
                    first_token_time = None
                    token_times = []
                    async for line in response.content:
                        chunk = line.decode('utf-8').rstrip()
                        if chunk:
                            if chunk.startswith("data: "):
                                json_string = chunk[6:]
                                if json_string:
                                    try:
                                        data = json.loads(json_string)
                                        if 'choices' in data and data['choices']:
                                            event_text = data['choices'][0].get('delta', {}).get('content', '')
                                            if event_text:
                                                end_time = time.perf_counter()
                                                time_taken = end_time - prev_end_time
                                                prev_end_time = end_time  
                                                if first_token_time is None:
                                                    first_token_time = time_taken 
                                                token_times.append(time_taken) 
                                                final_text_response += event_text
                                    except json.JSONDecodeError as e:
                                        logger.error(f"Error decoding JSON: {e}")
            except aiohttp.ClientError as e:
                logger.error(f"Error during API call: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                self._handle_error(deployment_name, max_tokens, None, "-99")
                return None

        end_time = time.perf_counter()
        total_time_taken = end_time - start_time
        token_times = [max(round(time_taken * 1000, 2), 1) for time_taken in token_times]
        token_generation_count = token_generation_count + count_tokens(encoding, final_text_response)
        logger.info(f"Finished call to model {deployment_name}. Time taken for chat: {round(total_time_taken, 2)} seconds or {round(total_time_taken * 1000, 2)} milliseconds.")

        if token_times:
            TBT = token_times[1:]
            TTFT = first_token_time

            headers_dict = {
                "completion_tokens": token_generation_count,
                "prompt_tokens": context_num_tokens,
                "region": response.headers.get("region", "N/A"),
                "utilization": response.headers.get("azure-openai-deployment-utilization", "N/A"),
                "tbt": TBT,
                "ttft": round(TTFT, 2),
            }
            self._store_results(deployment_name, max_tokens, headers_dict, round(total_time_taken, 2))

            return final_text_response
        else:
            self._handle_error(deployment_name, max_tokens, None, "-99")
  
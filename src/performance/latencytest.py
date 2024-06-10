import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from termcolor import colored
import traceback

import aiohttp
from openai import AsyncAzureOpenAI, AzureOpenAI
from tabulate import tabulate

from src.performance.aoaihelpers.utils import (
    calculate_statistics, extract_rate_limit_and_usage_info_async,
    get_local_time_in_azure_region, log_system_info)
from src.performance.messagegeneration import (RandomMessagesGenerator)
from utils.ml_logging import get_logger
from dotenv import load_dotenv

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

        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )

        self._validate_api_configurations()

        self.results = {}

    def _validate_api_configurations(self):
        """
        Validates if all necessary configurations are set.

        This method checks if the API key and Azure endpoint are set in the OpenAI client.
        These configurations are necessary for making requests to the OpenAI API.
        If any of these configurations are not set, the method raises a ValueError.

        :raises ValueError: If the API key or Azure endpoint is not set.
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
        self, deployment_name: str, max_tokens: int, temperature: Optional[int] = 0
    ):
        """
        Make an asynchronous chat completion call and log the time taken for the call.

        :param deployment_name: Name of the model deployment to use.
        :param max_tokens: Maximum number of tokens to generate.
        :param temperature: The temperature to use for the chat completion. Defaults to 0.
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
        prevent_server_caching: Optional[bool]=True,

    ):
        """
        Run asynchronous tests across different deployments and token counts.

        :param deployment_names: List of deployment names to test.
        :param max_tokens_list: List of max tokens values to test.
        :param same_model_interval: Interval in seconds to wait between requests to the same model. Default is 1.
        :param different_model_interval: Interval in seconds to wait between requests to different models. Default is 5.
        :param iterations: Number of iterations to run for each combination. Default is 1.
        :param temperature: The temperature to use for the model. Default is 0.
        :param context_tokens: The number of context tokens to use. If None, the model's maximum is used. Default is None.
        :param multiregion: Flag to indicate if tests should be run across multiple regions. Default is False.
        :param prevent_server_caching: Flag to indicate if server caching should be prevented. Default is True.
        :return: None. The results are stored in the instance attribute `self.results`.
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
                        prevent_server_caching
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
            prevent_server_caching: Optional[bool]=True,
        ) -> Optional[List[Any]]:
        """
        Run latency benchmarks for multiple deployments and token counts concurrently.

        :param deployment_names: List of deployment names to test.
        :param max_tokens_list: List of max tokens values to test.
        :param same_model_interval: Interval in seconds to wait between requests to the same model. Default is 1.
        :param different_model_interval: Interval in seconds to wait between requests to different models. Default is 5.
        :param iterations: Number of iterations to run for each combination. Default is 1.
        :param temperature: The temperature to use for the model. Default is 0.
        :param context_tokens: The number of context tokens to use. If None, the model's maximum is used. Default is None.
        :param multiregion: Flag to indicate if tests should be run across multiple regions. Default is False.
        :param prevent_server_caching: Flag to indicate if server caching should be prevented. Default is True.
        :return: None. The results are stored in the instance attribute `self.results`.
        """
        # Prepare tasks for each deployment
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
                prevent_server_caching
            )
            for deployment_name in deployment_names
            for max_tokens in max_tokens_list
        ]

        # Run tasks concurrently
        await asyncio.gather(*tasks)

    def calculate_and_show_statistics(self, show_descriptions: bool = False):
        """
        Calculate and display statistics for all tests conducted.
        Ensure each piece of expected data has a fallback.
        """
        stats = {
            key: self._calculate_statistics(data) for key, data in self.results.items()
        }
        headers = [
            "Model_MaxTokens",
            "Iterations",
            "Regions",
            "Average Time",
            "Median Time",
            "IQR Time",
            "95th Percentile Time",
            "99th Percentile Time",
            "CV Time",
            "Median Prompt Tokens",
            "IQR Prompt Tokens",
            "Median Completion Tokens",
            "IQR Completion Tokens",
            "95th Percentile Completion Tokens",
            "99th Percentile Completion Tokens",
            "CV Completion Tokens",
            "Error Rate",
            "Error Types",
            "Successful Runs",
            "Unsuccessful Runs",
            "Throttle Count",
            "Throttle Rate",
            "Best Run",
            "Worst Run"
        ]
        descriptions = [
            "The maximum number of tokens that the model can handle.",
            "The number of times the test was run.",
            "The geographical regions where the tests were conducted.",
            "The average value of time taken for all succesfull runs.",
            "The middle value of time taken for all runs. This is a measure of central tendency.",
            "The interquartile range (IQR) of time taken. This is a measure of statistical dispersion, being equal to the difference between 75th and 25th percentiles. IQR is a measure of variability and can help identify outliers.",
            "95% of the times taken are less than this value. This is another way to understand the distribution of values.",
            "99% of the times taken are less than this value. This is used to understand the distribution of values, particularly for identifying and handling outliers.",
            "The coefficient of variation of time taken. This is a normalized measure of the dispersion of the distribution. It's useful when comparing the degree of variation from one data series to another, even if the means are drastically different from each other.",
            "The middle value of the number of prompt tokens in all succesful runs.",
            "The interquartile range of the number of prompt tokens. This can help identify if the number of prompt tokens varies significantly in the tests.",
            "The middle value of the number of completion tokens in all succesful runs.",
            "The interquartile range of the number of completion tokens. This can help identify if the number of completion tokens varies significantly in the tests.",
            "95% of the completion tokens counts are less than this value.",
            "99% of the completion tokens counts are less than this value.",
            "The coefficient of variation of the number of completion tokens. This can help identify if the number of completion tokens varies significantly in the tests.",
            "The proportion of tests that resulted in an error.",
            "The types of errors that occurred during the tests.",
            "The number of tests that were successful.",
            "The number of tests that were not successful.",
            "The number of times the test was throttled or limited.",
            "The rate at which the test was throttled or limited.",
            "Details of the run with the best (lowest) time.",
            "Details of the run with the worst (highest) time."
        ]
        table = []
        for key, data in stats.items():
            regions = data.get("regions", [])
            regions = [r for r in regions if r is not None]
            region_string = ", ".join(set(regions)) if regions else "N/A"
            row = [
                key,
                data.get("number_of_iterations", "N/A"),
                region_string,
                data.get("average_time", "N/A"),
                data.get("median_time", "N/A"),
                data.get("iqr_time", "N/A"),
                data.get("percentile_95_time", "N/A"),
                data.get("percentile_99_time", "N/A"),
                data.get("cv_time", "N/A"),
                data.get("median_prompt_tokens","N/A"),
                data.get("iqr_prompt_tokens","N/A"),
                data.get("median_completion_tokens", "N/A"),
                data.get("iqr_completion_tokens", "N/A"),
                data.get("percentile_95_completion_tokens", "N/A"),
                data.get("percentile_99_completion_tokens", "N/A"),
                data.get("cv_completion_tokens", "N/A"),
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
                print(colored(header, 'blue'))
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
                "times_succesful": [],
                "times_unsucessfull": [],
                "regions": [],
                "number_of_iterations": 0, 
                "completion_tokens": [],
                "prompt_tokens": [],
                "errors": {"count": 0, "codes": []},
                "best_run": {
                    "time": float("inf"),
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                },
                "worst_run": {
                    "time": float("-inf"),
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                },
            }

        if time_taken is not None:
            self.results[key]["number_of_iterations"] += 1
            self.results[key]["times_succesful"].append(time_taken)
            self.results[key]["completion_tokens"].append(
                headers["completion_tokens"]
            )
            self.results[key]["prompt_tokens"].append(headers["prompt_tokens"])
            self.results[key]["regions"].append(
                headers["region"]
            )

            current_run = {
                "time": time_taken,
                "completion_tokens": headers.get("completion_tokens"),
                "prompt_tokens": headers.get("prompt_tokens"),
                "region": headers["region"],
                "utilization": headers["utilization"],
                "local_time": get_local_time_in_azure_region(headers["region"]),
            }

            # Update best and worst runs
            if time_taken < self.results[key]["best_run"]["time"]:
                self.results[key]["best_run"] = current_run
            if time_taken > self.results[key]["worst_run"]["time"]:
                self.results[key]["worst_run"] = current_run
        else:
            self._handle_error(deployment_name, max_tokens, None, "-99")

    def _handle_error(
        self, deployment_name: str, max_tokens: int, time_taken:int, response
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
                "times_succesful": [],
                "times_unsucessfull": [],
                "regions": [],
                "number_of_iterations": 0, 
                "completion_tokens": [],
                "prompt_tokens": [],
                "errors": {"count": 0, "codes": []},
                "best_run": {
                    "time": float("inf"),
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                },
                "worst_run": {
                    "time": float("-inf"),
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                },
            }
        self.results[key]["errors"]["count"] += 1
        self.results[key]["number_of_iterations"] += 1
        self.results[key]["times_unsucessfull"].append(time_taken)
        if response is not None:
            if response == "-99":
                logger.error("Error during API call: No captured time, test client error")
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
            times = list(filter(None, data.get("times_succesful", [])))
            completion_tokens = list(filter(None, data.get("completion_tokens", [])))
            prompt_tokens = list(filter(None, data.get("prompt_tokens", [])))
            error_count = data["errors"]["count"]
            error_codes = data["errors"]["codes"]
            error_distribution = {str(code): error_codes.count(code) for code in set(error_codes)}
            count_throw = error_distribution.get('429', 0)
            successful_runs = len(data['times_succesful'])
            unsuccessful_runs = len(data['times_unsucessfull'])
        
            stats = {
                "median_time": None,
                "regions": list(set(data.get("regions", []))),
                "iqr_time": None,
                "percentile_95_time": None,
                "percentile_99_time": None,
                "cv_time": None,
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
                "error_rate": error_count / total_requests if total_requests > 0 else 0,
                "number_of_iterations": total_requests,
                "throttle_count": count_throw,
                "throttle_rate": count_throw / total_requests if total_requests > 0 else 0,
                "errors_types": data.get("errors", {}).get("codes", []),
                "average_time": sum(times) / len(times) if times else None,
                "successful_runs": successful_runs,
                "unsuccessful_runs": unsuccessful_runs
            }
        
            if times:
                stats.update(zip(
                    ["median_time", "iqr_time", "percentile_95_time", "percentile_99_time", "cv_time"],
                    calculate_statistics(times)
                ))
        
            if completion_tokens:
                stats.update(zip(
                    ["median_completion_tokens", "iqr_completion_tokens", "percentile_95_completion_tokens", "percentile_99_completion_tokens", "cv_completion_tokens"],
                    calculate_statistics(completion_tokens)
                ))
        
            if prompt_tokens:
                stats.update(zip(
                    ["median_prompt_tokens", "iqr_prompt_tokens", "percentile_95_prompt_tokens", "percentile_99_prompt_tokens", "cv_prompt_tokens"],
                    calculate_statistics(prompt_tokens)
                ))
        
            # Optional: Add best_run and worst_run if they're defined and valid
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

    async def make_call(
        self,
        deployment_name: str,
        max_tokens: int,
        temperature: Optional[int] = 0,
        context_tokens: Optional[int] = None,
        prevent_server_caching: Optional[bool]=True,
    ):
        """
        Make an asynchronous chat completion call to the Azure OpenAI API and log the time taken for the call.

        :param deployment_name: Name of the model deployment to use.
        :param max_tokens: Maximum number of tokens to generate.
        :param temperature: The temperature to use for the chat completion. Defaults to 0.
        :param context_tokens: Number of context tokens to use. If not provided, 1000 tokens are used as default.
        :param prevent_server_caching: Flag to indicate if server caching should be prevented. Default is True.
        """
        url = f"{self.azure_endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={self.api_version}"
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
            "top_p": 1,
            "n": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "messages": messages,
        }

        logger.info(
            f"Initiating call for Model: {deployment_name}, Max Tokens: {max_tokens}"
        )
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            response = await session.post(url, headers=headers, json=body)
            end_time = time.time()
            time_taken = end_time - start_time
            if response.status != 200:
                logger.error(f"Error during API call: {response.text}")
                logger.error(f"Exception type: {response.status}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                self._handle_error(deployment_name, max_tokens, time_taken, response)
                logger.info(f"Unsuccesful Run - Time taken: {time_taken:.2f} seconds.")
            else: 
                response_headers = response.headers
                response_body = await response.json()
                headers = extract_rate_limit_and_usage_info_async(
                response_headers, response_body)
                self._store_results(
                    deployment_name, max_tokens, headers, time_taken
                )
                logger.info(f"Succesful Run - Time taken: {time_taken:.2f} seconds.")

class AzureOpenAIBenchmarkStreaming(AzureOpenAIBenchmarkLatency):
    # TODO: calculate stats
    # https://medium.com/@averma9838/implement-asynchronous-programming-in-azure-openai-for-task-parallelization-c26430491d7c
    def __init__(self, api_key, azure_endpoint, api_version="2024-02-15-preview"):
        """
        Initialize the AzureOpenAILatencyBenchmark with the API key, API version, and endpoint.
        """
        super().__init__(api_key, azure_endpoint, api_version)
        self.client = AsyncAzureOpenAI(
            api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint
        )
        self.results = {}

    async def make_call(self, deployment_name, max_tokens):
        """
        Make a chat completion call and print the time taken for the call.
        """
        start_time = time.time()
        logger.info(
            f"Starting call to model {deployment_name} with max tokens {max_tokens} at (Local time): {datetime.now()}, (GMT): {datetime.now(timezone.utc)}"
        )

        response = await self.client.chat.completions.create(
            model=deployment_name,
            temperature=0.7,
            max_tokens=max_tokens,
            stream=True,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Give me history of New York"},
            ],
        )

        try:
            async for chunk in response:
                chunk_json = json.loads(chunk.model_dump_json(indent=2))
                if chunk_json["choices"]:
                    content = chunk_json["choices"][0]["delta"]["content"]
                    print(content, end="")

        except Exception as e:
            logger.error(f"Error during API call: {str(e)}")
            self._handle_error(deployment_name, max_tokens, e)

        end_time = time.time()
        time_taken = end_time - start_time

        logger.info(
            f"Finished call to model {deployment_name}. Time taken for chat: {round(time_taken)} seconds or {round(time_taken * 1000)} milliseconds."
        )

import requests
import os
import threading
import datetime
import time
import csv
import json
from queue import Queue
from typing import Optional, Dict, List, Tuple
from threading import Lock
from dotenv import load_dotenv

from utils.ml_logging import get_logger
import statistics

# Load environment variables from .env file
load_dotenv()

TELEMETRY_USER_AGENT_HEADER = "x-ms-useragent"
USER_AGENT = "aoai-benchmark"

# Set up logger
logger = get_logger()


class AzureOpenAITestClientMultiRegion:
    """
    A tool for benchmarking Azure OpenAI in MultiregionDeployment.

    This class provides a tool for running load generation tests on Azure OpenAI. It allows for specification of
    parameters such as model, region, and endpoint.
    """

    def __init__(
        self,
        api_key: str,
        chat_model_name: str,
        concurrent_users: int,
        loop_times: int,
        azure_endpoint: Optional[str] = None,
        api_version: str = "2023-11-01",
    ) -> None:
        self.azure_endpoint: str = azure_endpoint
        self.api_key: str = api_key
        self.chat_model_name: str = chat_model_name
        self.concurrent_users: int = concurrent_users
        self.loop_times: int = loop_times
        self.api_version: str = api_version
        self.results_queue: Queue = Queue()
        self.event_time_up: threading.Event = threading.Event()
        self.response_times: List[float] = []
        self.lock: Lock = Lock()
        self.call_counter: int = 0
        self.successful_requests: int = 0
        self.unsuccessful_requests: int = 0
        self.response_times_by_region: Dict[str, List[float]] = {}
        self.call_counter_by_region: Dict[str, int] = {}
        self.successful_requests_by_region: Dict[str, int] = {}
        self.unsuccessful_requests_by_region: Dict[str, int] = {}
        self.errors: Dict[str, List[str]] = {}
        self.errors_by_region: Dict[str, Dict[str, List[str]]] = {}

    def call_azure_openai_chat_completions_api(
        self, body: Dict, loop_iteration: int
    ) -> None:
        url = f"{self.azure_endpoint}openai/deployments/{self.chat_model_name}/chat/completions?api-version={self.api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
            TELEMETRY_USER_AGENT_HEADER: USER_AGENT,
        }
        logger.info(f"Sending request to {url} with headers {headers} and body {body}")
        with requests.Session() as session:
            session.headers.update(headers)
            region = None
            try:
                start_time: float = time.monotonic()
                logger.info(
                    f"Thread {threading.current_thread().name} is sending request {loop_iteration + 1} at {start_time}"
                )
                response = session.post(url, json=body)
                response.raise_for_status()  # Raises HTTPError for bad responses
                elapsed_time: float = time.monotonic() - start_time
                region = response.headers.get("x-ms-region")
                with self.lock:
                    self.call_counter += 1
                    self.successful_requests += 1
                    if region:
                        if region not in self.successful_requests_by_region:
                            self.successful_requests_by_region[region] = 0
                        if region not in self.call_counter_by_region:
                            self.call_counter_by_region[region] = 0
                        if region not in self.unsuccessful_requests_by_region:
                            self.unsuccessful_requests_by_region[region] = 0
                        self.successful_requests_by_region[region] += 1
                        self.call_counter_by_region[region] += 1
                        self.response_times.append(elapsed_time)
                        if region not in self.response_times_by_region:
                            self.response_times_by_region[region] = []
                        self.response_times_by_region[region].append(elapsed_time)
            except requests.RequestException as e:
                error_message = str(e)
                logger.error(f"Request failed: {e}")
                with self.lock:
                    self.unsuccessful_requests += 1
                    self.errors.setdefault("Overall", []).append(error_message)
                    if region:
                        if region not in self.unsuccessful_requests_by_region:
                            self.unsuccessful_requests_by_region[region] = 0
                        self.unsuccessful_requests_by_region[region] += 1
                        self.errors_by_region.setdefault(region, {}).setdefault(
                            "Errors", []
                        ).append(error_message)

    def loop_test(self, body: Dict, interval: float) -> None:
        for i in range(self.loop_times):
            if self.event_time_up.is_set():
                break
            self.call_azure_openai_chat_completions_api(body=body, loop_iteration=i)
            time.sleep(interval)

    def run_test(
        self,
        body: Dict,
        interval: float,
        test_duration: float = 180.0,
        export_format: Optional[str] = None,
        results_filename: Optional[str] = None,
    ) -> None:
        # Get the current date and time
        now = datetime.datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M%S")

        # Replace any slashes in the model name with underscores
        model_name = self.chat_model_name.replace("/", "_")

        # Reset counters and set test time
        self.call_counter = 0
        self.successful_requests = 0
        self.unsuccessful_requests = 0
        self.test_time = test_duration

        # Set the results filename if not provided
        if not results_filename:
            results_filename = f"speed_test_benchmark_{model_name}_{date_time}"
        self.results_filename = results_filename

        # Create a list of threads for concurrent users
        threads: List[threading.Thread] = [
            threading.Thread(target=self.loop_test, args=(body, interval), daemon=True)
            for _ in range(self.concurrent_users)
        ]

        # Start a timer for the test duration
        timer: threading.Timer = threading.Timer(test_duration, self.event_time_up.set)
        timer.start()

        # Start all threads
        for thread in threads:
            thread.start()
        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        # Cancel the timer
        timer.cancel()

        # Calculate and print statistics
        stats = self.calculate_stats()

        # Export results if an export format is provided
        if export_format:
            self.export_results(stats, results_filename, export_format)

    def calculate_stats(self) -> Dict[str, Dict[str, float]]:
        total_requests: int = self.call_counter
        total_time: float = sum(self.response_times)
        test_time_in_minutes: float = self.test_time / 60
        stats: Dict[str, Dict[str, float]] = {}

        if total_requests > 0:
            avg_tpr: float = total_time / total_requests
            rpm: float = total_requests / test_time_in_minutes
            min_tpr: float = min(self.response_times)
            max_tpr: float = max(self.response_times)
            median_tpr: float = statistics.median(self.response_times)
            stats["Overall"] = {
                "Total Time": total_time,
                "Total Requests": total_requests,
                "Total Successful Requests": self.successful_requests,
                "Total Unsuccessful Requests": self.unsuccessful_requests,
                "Errors": self.errors.get("Overall", []),
                "Requests per Minute": rpm,
                "Average Time per Request": avg_tpr,
                "Min Time per Request": min_tpr,
                "Max Time per Request": max_tpr,
                "Median Time per Request": median_tpr,
            }
            logger.info(f"Overall Statistics: {stats['Overall']}")

            for region, times in self.response_times_by_region.items():
                total_requests_region: int = self.call_counter_by_region.get(region, 0)
                total_successful_requests_region: int = (
                    self.successful_requests_by_region.get(region, 0)
                )
                total_unsuccessful_requests_region: int = (
                    self.unsuccessful_requests_by_region.get(region, 0)
                )
                total_time_region: float = sum(times)
                if total_requests_region > 0:
                    avg_tpr_region: float = total_time_region / total_requests_region
                    rpm_region: float = total_requests_region / test_time_in_minutes
                    min_tpr_region: float = min(times)
                    max_tpr_region: float = max(times)
                    median_tpr_region: float = statistics.median(times)
                    stats[region] = {
                        "Total Time": total_time_region,
                        "Total Requests": total_requests_region,
                        "Total Successful Requests": total_successful_requests_region,
                        "Total Unsuccessful Requests": total_unsuccessful_requests_region,
                        "Requests per Minute": rpm_region,
                        "Errors": self.errors_by_region.get(region, {}).get(
                            "Errors", []
                        ),
                        "Average Time per Request": avg_tpr_region,
                        "Min Time per Request": min_tpr_region,
                        "Max Time per Request": max_tpr_region,
                        "Median Time per Request": median_tpr_region,
                    }
                    logger.info(f"Statistics for {region}: {stats[region]}")
        else:
            logger.info("No successful requests were made.")

        return stats

    def export_results(
        self,
        stats: Dict[str, Dict[str, float]],
        results_filename: str,
        export_format: str,
    ) -> None:
        if not export_format:
            export_format = "json"
        filename = f"{results_filename}.{export_format}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        try:
            if export_format == "csv":
                with open(filename, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Region", "Statistic", "Value"])
                    for region, region_stats in stats.items():
                        for stat, value in region_stats.items():
                            writer.writerow([region, stat, value])
            elif export_format == "json":
                with open(filename, "w") as file:
                    json.dump(stats, file, indent=4)
            logger.info(f"Results exported to {filename}")
        except IOError as e:
            logger.error(f"Failed to write to file: {e}")

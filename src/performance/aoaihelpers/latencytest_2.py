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


class AzureOpenAITestClient:
    """
    A tool for benchmarking Azure OpenAI.

    This class provides a tool for running load generation tests on Azure OpenAI. It allows for specification of
    parameters such as model, region, and endpoint.
    """

    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        chat_model_name: str,
        concurrent_users: int,
        loop_times: int,
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
            try:
                start_time: float = time.monotonic()
                logger.info(
                    f"Thread {threading.current_thread().name} is sending request {loop_iteration + 1} at {start_time}"
                )
                response = session.post(url, json=body)
                response.raise_for_status()  # Raises HTTPError for bad responses
                elapsed_time: float = time.monotonic() - start_time
                logger.info(
                    f"Request finished at {time.monotonic()}, elapsed time: {elapsed_time}"
                )
                self.results_queue.put(elapsed_time)
                with self.lock:
                    self.call_counter += 1
                    self.successful_requests += 1
                    logger.info(f"Number of calls made so far: {self.call_counter}")
                    self.response_times.append(elapsed_time)
            except requests.RequestException as e:
                logger.error(f"Request failed: {e}")
                with self.lock:
                    self.unsuccessful_requests += 1

    def loop_test(self, body: Dict, interval: float) -> None:
        for i in range(self.loop_times):
            if self.event_time_up.is_set():
                break
            self.call_azure_openai_chat_completions_api(body, i)
            time.sleep(interval)

    def run_test(
        self,
        body: Dict,
        interval: float,
        test_duration: float = 180.0,
        export_format: Optional[str] = None,
        results_filename: Optional[str] = None,
    ) -> None:
        now = datetime.datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M%S")
        model_name = self.chat_model_name.replace("/", "_")
        self.call_counter = 0
        self.successful_requests = 0
        self.unsuccessful_requests = 0
        self.test_time = test_duration

        if not results_filename:
            results_filename = f"speed_test_benchmark_{model_name}_{date_time}"
        self.results_filename = results_filename

        threads: List[threading.Thread] = [
            threading.Thread(target=self.loop_test, args=(body, interval), daemon=True)
            for _ in range(self.concurrent_users)
        ]

        timer: threading.Timer = threading.Timer(test_duration, self.event_time_up.set)
        timer.start()

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        timer.cancel()
        stats = self.calculate_stats()

        if export_format:
            self.export_results(stats, results_filename, export_format)

    def calculate_stats(self) -> Dict[str, float]:
        total_requests: int = self.call_counter
        total_time: float = sum(self.response_times)
        test_time_in_minutes: float = self.test_time / 60
        stats: Dict[str, float] = {}

        if total_requests > 0:
            avg_tpr: float = total_time / total_requests
            rpm: float = total_requests / test_time_in_minutes
            min_tpr: float = min(self.response_times)
            max_tpr: float = max(self.response_times)
            median_tpr: float = statistics.median(self.response_times)
            stats = {
                "Total Time": total_time,
                "Total Requests": total_requests,
                "Total Successful Requests": self.successful_requests,
                "Total Unsuccessful Requests": self.unsuccessful_requests,
                "Requests per Minute": rpm,
                "Average Time per Request": avg_tpr,
                "Min Time per Request": min_tpr,
                "Max Time per Request": max_tpr,
                "Median Time per Request": median_tpr,
            }
            logger.info(f"Statistics: {stats}")
        else:
            logger.info("No successful requests were made.")

        return stats

    def export_results(
        self, stats: Dict[str, float], results_filename: str, export_format: str
    ) -> None:
        if not export_format:
            export_format = "json"
        filename = f"{results_filename}.{export_format}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        try:
            if export_format == "csv":
                with open(filename, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Statistic", "Value"])
                    for stat, value in stats.items():
                        writer.writerow([stat, value])
            elif export_format == "json":
                with open(filename, "w") as file:
                    json.dump(stats, file, indent=4)
            logger.info(f"Results exported to {filename}")
        except IOError as e:
            logger.error(f"Failed to write to file: {e}")

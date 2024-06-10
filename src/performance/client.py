import datetime
import subprocess
import os
from typing import Optional, Literal, Tuple
from utils.ml_logging import get_logger
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logger
logger = get_logger()


class LoadTestBenchmarking:
    """
    A tool for benchmarking Azure OpenAI.

    This class provides a tool for running load generation tests on Azure OpenAI. It allows for specification of
    parameters such as model, region, and endpoint.

    :param model: Specifies the name of the Azure OpenAI model to be used for benchmarking.
                  This should be a string representing the model name, such as 'gpt-3'.
    :param region: Specifies the region where the Azure OpenAI model is deployed.
                   This should be a string representing the region, such as 'westus'.
    :param endpoint: Specifies the base endpoint URL of the Azure OpenAI deployment.
                     This should be a string representing the URL, such as 'https://api.openai.com'.
    """

    def __init__(self, model: str, region: str, endpoint: str):
        self.model = model
        self.region = region
        self.endpoint = endpoint

    def set_region(self, region: str) -> None:
        """
        Set a new region for the Azure OpenAI model.

        :param region: A string representing the new region, such as 'eastus'.
        """
        self.region = region

    def set_model(self, model: str) -> None:
        """
        Set a new model for benchmarking.

        :param model: A string representing the new model name, such as 'gpt-4'.
        """
        self.model = model

    @staticmethod
    def get_time_and_run_id() -> Tuple[str, str, str]:
        """
        Get the current time and a four-digit run ID.

        :return: A tuple containing the run ID, date string, and time string.
        """
        now = datetime.datetime.now()
        run_id = str(uuid.uuid4())[:4]
        now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        date_str, time_str = now_str.split("_")
        return run_id, date_str, time_str

    def run_tests(
        self,
        deployment: str,
        api_base_endpoint: Optional[int] = None,
        api_version: str = "2023-05-15",
        api_key_env: str = "OPENAI_API_KEY",
        clients: int = 20,
        requests: Optional[int] = None,
        duration: Optional[int] = None,
        run_end_condition_mode: Literal["and", "or"] = "or",
        rate: Optional[float] = None,
        aggregation_window: float = 60,
        context_generation_method: Literal["generate", "replay"] = "generate",
        replay_path: Optional[str] = None,
        shape_profile: Literal[
            "balanced", "context", "generation", "custom"
        ] = "balanced",
        context_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        prevent_server_caching: bool = True,
        completions: int = 1,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        output_format: Literal["jsonl", "human"] = "human",
        log_file_path: Optional[str] = "logs/",
        log_save_dir:Optional[str] = "logs/",
        retry: Literal["none", "exponential"] = "none",
        save_results: bool = False,
        results_save_path: Optional[float] = None,
    ) -> None:
        """
        Run load generation tests using Azure OpenAI.

        Retrieve and process data from the specified data repository.

        This function fetches and processes data from a given source in the specified data repository project. It allows
        for optional specification of deployment parameters such as API version, API key environment, number of clients,
        requests, duration, run end condition mode, rate, aggregation window, context generation method, replay path,
        shape profile, context tokens, max tokens, prevent server caching, completions, frequency penalty, presence
        penalty, temperature, top p, output format, log save directory, and retry strategy.

        :param deployment: Azure OpenAI deployment name.
        :param api_base_endpoint: Azure OpenAI deployment base endpoint.
        :param api_version: Set OpenAI API version. Defaults to "2023-05-15".
        :param api_key_env: Environment variable that contains the API KEY. Defaults to "OPENAI_API_KEY".
        :param clients: Set number of parallel clients to use for load generation. Defaults to 20.
        :param requests: Number of requests for the load run (whether successful or not). Default to 'until killed'.
        :param duration: Duration of load in seconds. Defaults to 'until killed'.
        :param run_end_condition_mode: Determines whether both the `requests` and `duration` args must be reached before
                                        ending the run ('and'), or whether to end the run when either arg is reached ('or').
                                        If only one arg is set, the run will end when it is reached. Defaults to 'or'.
        :param rate: Rate of request generation in Requests Per Minute (RPM). Default to as fast as possible.
        :param aggregation_window: Statistics aggregation sliding window duration in seconds. Defaults to 60.
        :param context_generation_method: Source of context messages to be used during testing. Defaults to "generate".
        :param replay_path: Path to JSON file containing messages for replay when using --context-message-source=replay.
        :param shape_profile: Shape profile of requests. Defaults to "balanced".
        :param context_tokens: Number of context tokens to use when --shape-profile=custom.
        :param max_tokens: Number of requested max_tokens when --shape-profile=custom. Defaults to unset.
        :param prevent_server_caching: Adds a random prefixes to all requests in order to prevent server-side caching.
                                        Defaults to True.
        :param completions: Number of completion for each request. Defaults to 1.
        :param frequency_penalty: Request frequency_penalty.
        :param presence_penalty: Request presence_penalty.
        :param temperature: Request temperature.
        :param top_p: Request top_p.
        :param output_format: Output format. Defaults to "human".
        :param log_save_dir: If provided, will save stdout to this directory. Filename will include important run parameters.
        :param retry: Request retry strategy. See README for details. Defaults to "none".
        :raises: subprocess.CalledProcessError: If an error occurs while running the command.
        :raises: Exception: If an unexpected error occurs.
        """
        process = None
        try:
            if api_base_endpoint is None and self.endpoint is None:
                raise ValueError(
                    "Either 'api_base_endpoint' must be provided as an argument, or 'endpoint' must be set in the constructor."
                )

            run_id, date_str, time_str = self.get_time_and_run_id()
           
            command = [
                "python",
                "-m",
                "src.performance.bench",
                "load",
                "--api-version",
                api_version,
                "--api-key-env",
                api_key_env,
                "--clients",
                str(clients),
            ]

            if requests is not None:
                command.extend(["--requests", str(requests)])

            if duration is not None:
                command.extend(["--duration", str(duration)])

            command.extend(
                [
                    "--run-end-condition-mode",
                    run_end_condition_mode,
                ]
            )

            if rate is not None:
                command.extend(["--rate", str(rate)])

            command.extend(
                [
                    "--aggregation-window",
                    str(aggregation_window),
                    "--context-generation-method",
                    context_generation_method,
                ]
            )

            if replay_path is not None:
                command.extend(["--replay-path", replay_path])

            command.extend(
                [
                    "--shape-profile",
                    shape_profile,
                ]
            )

            if context_tokens is not None:
                command.extend(["--context-tokens", str(context_tokens)])

            if max_tokens is not None:
                command.extend(["--max-tokens", str(max_tokens)])

            command.extend(
                [
                    "--prevent-server-caching",
                    str(prevent_server_caching),
                    "--completions",
                    str(completions),
                ]
            )

            if frequency_penalty is not None:
                command.extend(["--frequency-penalty", str(frequency_penalty)])

            if presence_penalty is not None:
                command.extend(["--presence-penalty", str(presence_penalty)])

            if temperature is not None:
                command.extend(["--temperature", str(temperature)])

            if top_p is not None:
                command.extend(["--top-p", str(top_p)])

            command.extend(
                [
                    "--output-format",
                    output_format,
                ]
            )

            if log_save_dir is not None:
                command.extend(["--log-save-dir", log_save_dir])

            command.extend(
                [
                    "--retry",
                    retry,
                    "--deployment",
                    deployment,
                    api_base_endpoint or self.endpoint,
                ]
            )

            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            logger.info(
                f"Initiating load generation tests with ID {run_id}. Log output will be directed to: {log_file_path}"
            )
            logger.info(f"Executing command: {' '.join(command)}")

            with open(log_file_path, "w") as log_file:
                process = subprocess.Popen(
                    command, stdout=log_file, stderr=subprocess.STDOUT
                )

                # Wait for the process to terminate
                process.communicate()

            if save_results:
                if not results_save_path:
                    results_save_path = os.path.join(
                        log_save_dir,
                        f"{self.model}/{self.region}/results/{date_str}/{time_str}/{run_id}.csv",
                    )

                os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
                combine_logs_command = [
                    "python",
                    "-m",
                    "src.performancecontrib.combine_logs",
                    log_file_path,
                    results_save_path,
                ]

                logger.info(
                    f"Combining logs. Results will be saved to: {results_save_path}"
                )
                subprocess.check_call(combine_logs_command)

            logger.info(
                f"Load generation tests have completed. Please refer to {log_file_path} for the detailed logs."
            )

        except subprocess.CalledProcessError as e:
            logger.error(
                f"An error occurred while executing the command {command}: {str(e)}",
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
            raise
        finally:
            if process and process.poll() is None:
                process.kill()

    def run_tests_batch(
        self,
        deployment: str,
        token_rate_workload_list: str,
        api_base_endpoint: Optional[str] = None,
        aggregation_window: int = 120,
        duration: Optional[int] = None,
        requests: Optional[int] = None,
        clients: int = 20,
        start_ptum_runs_at_full_utilization: bool = True,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        prevent_server_caching: bool = True,
        api_key_env: str = "OPENAI_API_KEY",
        api_version: str = "2023-05-15",
        context_generation_method: Literal["generate", "replay"] = "generate",
        replay_path: Optional[str] = None,
        shape_profile: Literal[
            "balanced", "context", "generation", "custom"
        ] = "balanced",
        output_format: Literal["jsonl", "human"] = "human",
        log_save_dir: Optional[str] = "logs/",
        retry: Literal["none", "exponential"] = "none",
        run_end_condition_mode: Literal["and", "or"] = "or",
        context_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        save_results: bool = True,
        results_save_path: Optional[float] = None,
    ) -> None:
        """
        Run load generation tests using Azure OpenAI.

        Retrieve and process data from the specified data repository.

        :param deployment: Azure OpenAI deployment name.
        :param api_base_endpoint: Azure OpenAI deployment base endpoint.
        :param token_rate_workload_list: Comma-separated list of all workload args to test.
        :param aggregation_window: Length of time to collect and aggregate statistics for each run. Defaults to 120.
        :param duration: Max duration to run each benchmark run.
        :param requests: Minimum number of requests to include in each benchmark run.
        :param run_end_condition_mode: Determines whether both the `requests` and `duration` args must be reached before ending the run ('and'), or whether to end the run when either arg is reached ('or'). Defaults to 'or'.
        :param clients: Number of clients to use for each run. Defaults to 20.
        :param start_ptum_runs_at_full_utilization: Starts all PTU-M runs at 100% utilization, preventing any burst capacity from inflating the results. Defaults to True.
        :param log_save_dir: If provided, will save stdout to this directory. Filename will include important run parameters.
        :param retry: Request retry strategy. Defaults to "none".
        :param frequency_penalty: Request frequency_penalty.
        :param presence_penalty: Request frequency_penalty.
        :param temperature: Request temperature.
        :param top_p: Request top_p.
        :param prevent_server_caching: Adds a random prefixes to all requests in order to prevent server-side caching. Defaults to True.
        :param api_key_env: Environment variable that contains the API KEY. Defaults to "OPENAI_API_KEY".
        :param api_version: Set OpenAI API version. Defaults to "2023-05-15".
        :param replay_path: Path to JSON file containing messages for replay when using --context-message-source=replay.
        :param context_tokens: Number of context tokens to use when --shape-profile=custom.
        :param max_tokens: Number of requested max_tokens when --shape-profile=custom.
        :raises: subprocess.CalledProcessError: If an error occurs while running the command.
        :raises: Exception: If an unexpected error occurs.
        """
        process = None
        try:
            run_id, date_str, time_str = self.get_time_and_run_id()
            log_file_path = os.path.join(
                log_save_dir,
                f"{self.model}/{self.region}/log_runs/{date_str}/{time_str}/{run_id}.log",
            )

            command = [
                "python",
                "-m",
                "src.performancecontrib.batch_runner",
                api_base_endpoint or self.endpoint,
                "--deployment",
                deployment,
                "--token-rate-workload-list",
                token_rate_workload_list,
                "--log-save-dir",
                log_save_dir,
                "--aggregation-window",
                str(aggregation_window),
                "--start-ptum-runs-at-full-utilization",
                str(start_ptum_runs_at_full_utilization),
                "--api-version",
                str(api_version),
                "--api-key-env",
                api_key_env,
                "--clients",
                str(clients),
            ]

            if requests is not None:
                command.extend(["--requests", str(requests)])

            if duration is not None:
                command.extend(["--duration", str(duration)])

            command.extend(
                [
                    "--run-end-condition-mode",
                    run_end_condition_mode,
                ]
            )

            if frequency_penalty is not None:
                command.extend(["--frequency-penalty", str(frequency_penalty)])

            if presence_penalty is not None:
                command.extend(["--presence-penalty", str(presence_penalty)])

            if temperature is not None:
                command.extend(["--temperature", str(temperature)])

            if top_p is not None:
                command.extend(["--top-p", str(top_p)])

            command.extend(
                [
                    "--prevent-server-caching",
                    str(prevent_server_caching),
                ]
            )

            if replay_path is not None:
                command.extend(["--replay-path", replay_path])

            if context_tokens is not None:
                command.extend(["--context-tokens", str(context_tokens)])

            if max_tokens is not None:
                command.extend(["--max-tokens", str(max_tokens)])

            if retry is not None:
                command.extend(["--retry", retry])

            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            logger.info(
                f"Initiating load generation tests. Log output will be directed to: {log_file_path}"
            )
            logger.info(f"Executing command: {' '.join(command)}")

            with open(log_file_path, "w") as log_file:
                process = subprocess.Popen(
                    command, stdout=log_file, stderr=subprocess.STDOUT
                )

                # Wait for the process to terminate
                process.communicate()

            if process:
                # Wait for the process to terminate
                process.communicate()

            if save_results:
                if not results_save_path:
                    results_save_path = os.path.join(
                        log_save_dir,
                        f"{self.model}/{self.region}/results/{date_str}/{time_str}/{run_id}.csv",
                    )

                os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
                combine_logs_command = [
                    "python",
                    "-m",
                    "src.performancecontrib.combine_logs",
                    log_file_path,
                    results_save_path,
                ]

                logger.info(
                    f"Combining logs. Results will be saved to: {results_save_path}"
                )
                subprocess.check_call(combine_logs_command)

            logger.info(
                f"Load generation tests have completed. Please refer to {log_file_path} for the detailed logs."
            )

        except subprocess.CalledProcessError as e:
            logger.error(
                f"An error occurred while executing the command {command}: {str(e)}",
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
            raise
        finally:
            if process and process.poll() is None:
                process.kill()

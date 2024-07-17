from src.app.managers import (create_benchmark_non_streaming_client,
                              create_benchmark_streaming_client)

from my_utils.ml_logging import get_logger
import copy
from src.app.performance.results import BenchmarkPerformanceResult
# Set up logger
logger = get_logger()

import streamlit as st

async def run_benchmark_tests(test_status_placeholder: st.container) -> None:
    """
    Run the benchmark tests asynchronously, with detailed configuration for each test.

    :param test_status_placeholder: Streamlit placeholder for the test status.
    """
    deployment_clients = [
        (
            create_benchmark_streaming_client(
                deployment["key"], deployment["endpoint"], deployment["version"]
            )
            if deployment["stream"]
            else create_benchmark_non_streaming_client(
                deployment["key"], deployment["endpoint"], deployment["version"]
            ),
            deployment_name,
        )
        for deployment_name, deployment in st.session_state.deployments.items()
    ]

    async def safe_run(client, deployment_name):
        try:
            await client.run_latency_benchmark_bulk(
                deployment_names=[deployment_name],
                max_tokens_list=st.session_state["settings"]["max_tokens_list"],
                iterations=st.session_state["settings"]["num_iterations"],
                context_tokens=st.session_state["settings"]["context_tokens"],
                temperature=st.session_state["settings"]["temperature"],
                byop=st.session_state["settings"]["prompts"],
                prevent_server_caching=st.session_state["settings"][
                    "prevent_server_caching"
                ],
                timeout=st.session_state["settings"]["timeout"],
                top_p=st.session_state["settings"]["top_p"],
                n=1,
                presence_penalty=st.session_state["settings"]["presence_penalty"],
                frequency_penalty=st.session_state["settings"]["frequency_penalty"],
            )
        except Exception as e:
            logger.error(
                f"An error occurred with deployment '{deployment_name}': {str(e)}",
                exc_info=True,
            )
            st.error(f"An error occurred with deployment '{deployment_name}': {str(e)}")

    logger.info(f"Total number of deployment clients: {len(deployment_clients)}")

    for client, deployment_name in deployment_clients:
        await safe_run(client, deployment_name)

    try:
        stats = [
            client.calculate_and_show_statistics() for client, _ in deployment_clients
        ]
        stats_raw = [client.results for client, _ in deployment_clients]
        st.session_state["benchmark_results"] = stats
        st.session_state["benchmark_results_raw"] = stats_raw
        settings_snapshot = copy.deepcopy(st.session_state["settings"])
        results = BenchmarkPerformanceResult(
            result=stats, settings=settings_snapshot
        )
        st.session_state["results"][results.id] = results.to_dict()
        test_status_placeholder.markdown(
            f"Benchmark <span style='color: grey;'>{results.id}</span> Completed",
            unsafe_allow_html=True,
        )
    except Exception as e:
        logger.error(
            f"An error occurred while processing the results: {str(e)}", exc_info=True
        )
        st.error(f"An error occurred while processing the results: {str(e)}")

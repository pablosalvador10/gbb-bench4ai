"""
This script contains a utility function for interacting with the Azure OpenAI API.
"""

from requests import Response
import numpy as np
from scipy import stats
import geocoder
from datetime import datetime
import pytz
import psutil

from typing import List, Tuple, Optional, Dict, Any
from utils.ml_logging import get_logger
from src.performance.aoaihelpers.constants import AZURE_REGION_TO_TIMEZONE


# Set up logger
logger = get_logger()
UTILIZATION_HEADER = "azure-openai-deployment-utilization"


def extract_rate_limit_and_usage_info(response: Response) -> Dict[str, Optional[int]]:
    """
    Extracts rate limiting information from the Azure Open API response headers and usage information from the payload.

    :param response: The response object returned by a requests call.
    :return: A dictionary containing the remaining requests, remaining tokens, and usage information
            including prompt tokens, completion tokens, total tokens, utilization, and retry after ms.
    """
    headers = response.headers
    usage = response.json().get("usage", {})
    retry_after_ms = headers.get("retry-after-ms")
    if not retry_after_ms:
        retry_after_ms = headers.get("retry-after")
    return {
        "remaining-requests": headers.get("x-ratelimit-remaining-requests"),
        "remaining-tokens": headers.get("x-ratelimit-remaining-tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "utilization": headers.get("azure-openai-deployment-utilization") or "NA",
        "retry_after_ms": retry_after_ms,
    }


def extract_rate_limit_and_usage_info_async(headers, body_response) -> Dict[str, Any]:
    """
    Extracts rate limiting information from the Azure Open API response headers and usage information from the payload.

    :param response: The response object returned by a requests call.
    :return: A dictionary containing the remaining requests, remaining tokens, and usage information
            including prompt tokens, completion tokens, total tokens, utilization, and retry after ms.
    """
    headers = dict(headers)
    usage = body_response.get("usage", {})
    retry_after_ms = headers.get("retry-after-ms")
    if not retry_after_ms:
        retry_after_ms = headers.get("retry-after")

    if retry_after_ms is None: 
       logger.debug("retry-after-ms or retry-after is None in headers")
       retry_after_ms = "NA"

    remaining_requests = headers.get("x-ratelimit-remaining-requests")
    if remaining_requests is None:
        logger.warning("x-ratelimit-remaining-requests is None in headers")

    remaining_tokens = headers.get("x-ratelimit-remaining-tokens")
    if remaining_tokens is None:
        logger.warning("x-ratelimit-remaining-tokens is None in headers")

    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is None:
        logger.warning("prompt_tokens is None in usage")

    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        logger.warning("completion_tokens is None in usage")

    region = headers.get("x-ms-region")
    if region is None:
        logger.warning("x-ms-region is None in headers")

    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        logger.warning("total_tokens is None in usage")

    utilization = headers.get(UTILIZATION_HEADER) or "NA"
    if utilization != "NA":
        if len(utilization) == 0:
            logger.warning(f"got empty utilization header {UTILIZATION_HEADER}")
        elif utilization[-1] != "%":
            logger.warning(
                f"invalid utilization header value: {UTILIZATION_HEADER}={utilization}"
            )
        else: 
            try:
                utilization = float(utilization[:-1])
            except ValueError as e:
                logger.warning(
                    f"unable to parse utilization header value: {UTILIZATION_HEADER}={util_str}: {e}"
                )

    return {
        "remaining-requests": remaining_requests,
        "remaining-tokens": remaining_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "region": region,
        "total_tokens": total_tokens,
        "utilization": utilization,
        "retry_after_ms": retry_after_ms,
    }


def get_local_time(timezone: str) -> str:
    """
    Get the current local time in the specified timezone.

    :param timezone: The name of the timezone, e.g., 'Europe/Stockholm' for Sweden Central.
    :return: The current local time in the specified timezone, formatted as a string.
    """
    local_time = datetime.now(pytz.timezone(timezone))
    return local_time.strftime("%Y-%m-%d %H:%M:%S %Z")


def get_machine_location():
    try:
        g = geocoder.ip("me")
        return f"{g.city}, {g.country}"
    except Exception:
        return "N/A"


def get_local_time_in_azure_region(region: str) -> str:
    """
    Get the current local time in the specified Azure region.

    :param region: The name of the Azure region, e.g., 'Sweden Central'.
    :return: The current local time in the specified Azure region, formatted as a string.
    """
    timezone = AZURE_REGION_TO_TIMEZONE.get(region, None)
    if timezone:
        return get_local_time(timezone)
    else:
        machine_location = get_machine_location()
        logger.info(
            f"""Target region {region} not found, assuming local deployment. 
                    Machine location during test: {machine_location}"""
        )
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")


def calculate_statistics(
    data: List[float],
) -> Tuple[
    Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]
]:
    """
    Calculate statistical measures for a list of numbers.

    Parameters:
    data (List[float]): Input list of numbers.

    Returns:
    Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    A tuple containing the following statistics, or None for each if the input list is empty or an error occurs:

    - Median: Middle value in the sorted list. Less affected by outliers.
    - Interquartile Range (IQR): Measures statistical dispersion. Difference between 75th and 25th percentiles.
    - 95th Percentile: Value below which 95% of observations fall.
    - 99th Percentile: Value below which 99% of observations fall.
    - Coefficient of Variation (CV): Ratio of standard deviation to mean. Useful for comparing variability across data series.
    """
    try:
        logger.info(f"Calculating statistics for data: {data}")

        if not data:
            result = (None, None, None, None, None)
            logger.info(f"No data provided. Returning result: {result}")
            return result

        data_array = np.array(data)
        logger.info(f"Data converted to numpy array: {data_array}")

        # Calculate the median
        median = np.median(data_array)
        logger.info(f"Calculated median: {median}")

        # Calculate the IQR
        iqr = stats.iqr(data_array)
        logger.info(f"Calculated interquartile range (IQR): {iqr}")

        # Calculate the 95th percentile
        percentile_95 = np.percentile(data_array, 95)
        logger.info(f"Calculated 95th percentile: {percentile_95}")

        # Calculate the 99th percentile
        percentile_99 = np.percentile(data_array, 99)
        logger.info(f"Calculated 99th percentile: {percentile_99}")

        # Calculate the coefficient of variation
        cv = stats.variation(data_array)
        logger.info(f"Calculated coefficient of variation (CV): {cv}")

        result = (median, iqr, percentile_95, percentile_99, cv)
        logger.info(f"Result: {result}")
        return result

    except Exception as e:
        logger.error(f"An error occurred while calculating statistics: {e}")
        return None, None, None, None, None


def percentile(data: List[float], percentile: float):
    """
    Calculate the given percentile of a list of numbers.

    :param data: List of numbers.
    :param percentile: Percentile to calculate (0-100).
    :return: Calculated percentile value.
    """
    size = len(data)
    return sorted(data)[int(size * percentile / 100)]


def log_system_info():
    """
    Logs the current thread, CPU usage, RAM usage, and other system statistics.
    """
    # Log the CPU usage
    logger.info(f"CPU usage: {psutil.cpu_percent()}%")
    # Log the RAM usage
    logger.info(f"RAM usage: {psutil.virtual_memory().percent}%")

    # Log the number of logical and physical CPUs
    logger.debug(f"Number of logical CPUs: {psutil.cpu_count(logical=True)}")
    logger.debug(f"Number of physical CPUs: {psutil.cpu_count(logical=False)}")

    # Log the disk usage
    disk_usage = psutil.disk_usage("/")
    logger.debug(f"Disk total: {disk_usage.total / (1024**3):.2f} GB")
    logger.debug(f"Disk used: {disk_usage.used / (1024**3):.2f} GB")
    logger.debug(f"Disk free: {disk_usage.free / (1024**3):.2f} GB")
    logger.debug(f"Disk percent used: {disk_usage.percent}%")

    # Log the network statistics
    net_io = psutil.net_io_counters()
    logger.debug(f"Bytes sent: {net_io.bytes_sent}")
    logger.debug(f"Bytes received: {net_io.bytes_recv}")
    logger.debug(f"Packets sent: {net_io.packets_sent}")
    logger.debug(f"Packets received: {net_io.packets_recv}")

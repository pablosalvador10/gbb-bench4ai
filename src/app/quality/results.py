"""
This script defines a class for capturing and representing the results of benchmark quality tests conducted on various systems or processes.
It is designed to standardize the way test outcomes are recorded, including unique identifiers, timestamps, results, and the settings under which
the test was conducted. This facilitates easier comparison, analysis, and storage of test results.
"""

import datetime
import uuid


class BenchmarkQualityResult:
    """
    Represents the result of a benchmark performance test.

    Attributes:
        id (str): A unique identifier for the test result.
        timestamp (datetime.datetime): The date and time when the test was conducted.
        result (dict): The outcome of the benchmark test, stored as a dictionary.
        settings (dict): The configuration settings used for the test.

    Methods:
        to_dict: Converts the instance into a dictionary representation.
    """

    def __init__(self, result: dict, settings: dict):
        """
        Initializes a new instance of the BenchmarkPerformanceResult class.

        Args:
            result (dict): The outcome of the benchmark test, stored as a dictionary.
            settings (dict): The configuration settings used for the test.
        """
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.datetime.now()
        self.result = result  # Ensure this is a dictionary
        self.settings = settings

    def to_dict(self):
        """
        Converts the BenchmarkPerformanceResult instance into a dictionary.

        Returns:
            A dictionary containing the test result's details.
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "result": self.result,
            "settings": self.settings,
        }

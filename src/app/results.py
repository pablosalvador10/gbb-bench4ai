import uuid
import datetime

class BenchmarkPerformanceResult:
    """
    Represents the result of a benchmark performance test.

    Attributes:
        id (str): A unique identifier for the test result.
        timestamp (datetime): The date and time when the test was conducted.
        result (str): The outcome of the benchmark test.
        settings (dict): The configuration settings used for the test.

    Methods:
        to_dict(): Returns a dictionary representation of the instance.
    """

    def __init__(self, result, settings):
        """
        Initializes a new instance of the BenchmarkPerformanceResult class.

        Parameters:
            result (str): The outcome of the benchmark test.
            settings (dict): The configuration settings used for the test.
        """
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.datetime.now()
        self.result = result
        self.settings = settings

    def to_dict(self):
        """
        Converts the BenchmarkPerformanceResult instance into a dictionary.

        Returns:
            dict: A dictionary containing the test result's details.
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "result": self.result,
            "settings": self.settings,
        }
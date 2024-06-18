import logging
import pandas as pd
from datasets import load_dataset

class Eval:
    '''
    Parent class for all evaluation benchmarks
    Inputs:
        deployment_config:
            key: Azure API key
            endpoint: Azure endpoint
            model: AOAI deployment name
        sample_size: fraction of data to sample for evaluation (optional)
        log_level: logging level (optional)
    
    Protected Methods:
        _score: Compare the generated answer to the correct answer
        _load_data: Load the dataset from the MMLU benchmark into memory
    
    '''

    def __init__(self, deployment_config: dict, sample_size: float = 1.0, log_level: str = "INFO"):
        self.sample_size = sample_size
        self._key = deployment_config["key"]
        self._base = deployment_config["endpoint"]
        self._model = deployment_config["model"]

        self.logger = logging.getLogger(__name__)
        if log_level == "INFO":
            logging.basicConfig(level=logging.INFO)
        elif log_level == "DEBUG":
            logging.basicConfig(level=logging.DEBUG)
        elif log_level == "WARNING":
            logging.basicConfig(level=logging.WARNING)
        elif log_level == "ERROR":
            logging.basicConfig(level=logging.ERROR)
        else:
            logging.basicConfig(level=logging.INFO)
            self.logger.warning(f"Unrecognized log level: {log_level}. Defaulting to INFO")

    def _load_data(self, dataset: str, subset: str, split: str, flatten: bool = False) -> pd.DataFrame:
        # Download dataset
        self.logger.info(f"Loading {dataset} data")
        hf_data = load_dataset(dataset, subset, split=split)
        if flatten:
            hf_data = hf_data.flatten()
        df = hf_data.to_pandas()
        self.logger.info(f"Load Complete. {df.shape[0]} rows.")
        return df
    
   
    def _score(self, generated: str, correct: str) -> int:
        self.logger.debug(f"Scoring {str(generated)} vs. {str(correct)}")
        try:
            if str(generated).lower() == str(correct).lower():
                return 1
            else:
                return 0
        
        except TypeError as t:
            self.logger.warning(f"TypeError while scoring {generated} vs. {correct} : {t}")
            return 0
        except ValueError as v:
            self.logger.warning(f"ValueError while scoring {generated} vs. {correct} : {v}")
            return 0
        except Exception as e:
            self.logger.warning(f"Exception while scoring {generated} vs. {correct} : {e}")
            return 0
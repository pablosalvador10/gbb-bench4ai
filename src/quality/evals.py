import asyncio
import logging
import os
from typing import Dict, List

import pandas as pd
from datasets import load_dataset
from openai import AzureOpenAI

from src.quality.metrics import accuracy, sentence_transformer_similarity
from my_utils.ml_logging import get_logger


class Eval:
    """
    Parent class for all evaluation benchmarks
    Inputs:
        deployment_config:
            key: Azure API key
            endpoint: Azure endpoint
            model: AOAI deployment name
            version: API version
        sample_size: fraction of data to sample for evaluation (optional)
        log_level: logging level (optional)

    Protected Methods:
        _score: Compare the generated answer to the correct answer
        _load_data: Load the dataset from the MMLU benchmark into memory

    """

    def __init__(
        self, deployment_config: dict, sample_size: float = 1.0, log_level: str = "INFO"
    ):
        self.sample_size = sample_size
        self._key = deployment_config["key"]
        self._base = deployment_config["endpoint"]
        self.version = deployment_config["version"]
        self.model = deployment_config["model"]

        self.logger = get_logger()
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
            self.logger.warning(
                f"Unrecognized log level: {log_level}. Defaulting to INFO"
            )

    def load_data(
        self, dataset: str, subset: str, split: str, flatten: bool = False
    ) -> pd.DataFrame:
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
            self.logger.warning(
                f"TypeError while scoring {generated} vs. {correct} : {t}"
            )
            return 0
        except ValueError as v:
            self.logger.warning(
                f"ValueError while scoring {generated} vs. {correct} : {v}"
            )
            return 0
        except Exception as e:
            self.logger.warning(
                f"Exception while scoring {generated} vs. {correct} : {e}"
            )
            return 0


class MMLU(Eval):
    """
    This is a class implementing the MMLU benchmark evaluation.
    Inputs:
        deployment_config:
            key: AOAI API key
            endpoint: AOAI endpoint
            model: AOAI deployment name
            version: API version
        categories: list of subjects to evaluate (optional)
            Must be one of the following: [STEM, Medical, Business, Social Sciences, Humanities, Other]
        sample_size: fraction of data to sample for evaluation (optional)
        log_level: logging level (optional)

    Inhereted Methods:
        _score: Compare the generated answer to the correct answer
        _load_data: Load the dataset from the MMLU benchmark into memory

    Private Methods:
        __transform_data: Transform the dataset into a format that can be used by the Azure OpenAI API
        __call_aoai: Call the Azure OpenAI API to generate an answer

    Public Method:
        test: Run the MMLU evaluation and output a dataframe

    """

    def __init__(
        self,
        deployment_config: dict,
        sample_size: float = 1.0,
        log_level: str = "INFO",
        categories: list = None,
    ):
        super().__init__(
            deployment_config=deployment_config,
            sample_size=sample_size,
            log_level=log_level,
        )
        self.categories = categories

        global subject2category
        subject2category = {
            "abstract_algebra": "stem",
            "anatomy": "medical",
            "astronomy": "stem",
            "business_ethics": "business",
            "clinical_knowledge": "medical",
            "college_biology": "medical",
            "college_chemistry": "stem",
            "college_computer_science": "stem",
            "college_mathematics": "stem",
            "college_medicine": "medical",
            "college_physics": "stem",
            "computer_security": "stem",
            "conceptual_physics": "stem",
            "econometrics": "social_sciences",
            "electrical_engineering": "stem",
            "elementary_mathematics": "stem",
            "formal_logic": "humanities",
            "global_facts": "humanities",
            "high_school_biology": "stem",
            "high_school_chemistry": "stem",
            "high_school_computer_science": "stem",
            "high_school_european_history": "humanities",
            "high_school_geography": "social_sciences",
            "high_school_government_and_politics": "social_sciences",
            "high_school_macroeconomics": "social_sciences",
            "high_school_mathematics": "stem",
            "high_school_microeconomics": "social_sciences",
            "high_school_physics": "stem",
            "high_school_psychology": "social_sciences",
            "high_school_statistics": "stem",
            "high_school_us_history": "humanities",
            "high_school_world_history": "humanities",
            "human_aging": "medical",
            "human_sexuality": "social_sciences",
            "international_law": "humanities",
            "jurisprudence": "humanities",
            "logical_fallacies": "humanities",
            "machine_learning": "stem",
            "management": "business",
            "marketing": "business",
            "medical_genetics": "medical",
            "miscellaneous": "other",
            "moral_disputes": "humanities",
            "moral_scenarios": "humanities",
            "nutrition": "medical",
            "philosophy": "humanities",
            "prehistory": "humanities",
            "professional_accounting": "business",
            "professional_law": "humanities",
            "professional_medicine": "medical",
            "professional_psychology": "social_sciences",
            "public_relations": "social_sciences",
            "security_studies": "social_sciences",
            "sociology": "social_sciences",
            "us_foreign_policy": "social_sciences",
            "virology": "medical",
            "world_religions": "humanities",
        }

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Filter to specified categories of subjects.
        # Broad categories defined in subject2category dict (global) - STEM, Medical, Business, Social Sciences, Humanities, Other
        if self.categories:
            self.categories = [c.lower() for c in self.categories]
            self.categories = [c.replace(" ", "_") for c in self.categories]
            df["category"] = df["subject"].map(subject2category)
            df = df[df["category"].isin(self.categories)]
            self.logger.info(
                f"Trimmed dataset to specified categories: {df.value_counts('category')}"
            )

        # Subset data based on subject
        self.logger.info(f"Sampling data to {self.sample_size*100}% of each subject")
        df = (
            df.groupby("subject", as_index=False, group_keys=False)
            .apply(lambda s: s.sample(frac=self.sample_size, replace=False))
            .reset_index()
        )

        self.logger.info(f"Data loaded. {df.shape[0]} rows.")
        return df

    def __call_aoai(self, row: list) -> dict:
        client = AzureOpenAI(
            azure_endpoint=self._base, api_key=self._key, api_version=self.version
        )

        sys_message = "Complete the given problem to the best of your ability. \
                        Accuracy is very important. \
                        Choices are a list of quoted strings with a starting index of 0 \
                        Select ONLY ONE answer from the choices. \
                        Return ONLY the index of the correct answer in the choices list. Your answer must be a single ineteger."

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_message},
                {
                    "role": "user",
                    "content": f"Question: {row['question']}.  Choices: {row['choices']}. Answer:",
                },
            ],
        )
        output = {
            "generated": response.choices[0].message.content,
            "correct": row["answer"],
            "subject": row["subject"],
        }

        return output

    async def test(self, data: pd.DataFrame) -> pd.DataFrame:
        output_list = []
        self.logger.info("Starting evaluation")
        for index, row in data.iterrows():
            self.logger.info(f"Evaluating row {index} of {data.shape[0]}")
            try:
                output = self.__call_aoai(row)
                output["score"] = self._score(output["generated"], output["correct"])
                output_list.append(output)
            except Exception as e:
                self.logger.warning(f"Skipping...error in row {index}: {e}")

        self.logger.info("Evaluation complete")

        self.logger.info("Aggregating Results")
        results = (
            pd.DataFrame(output_list)
            .groupby("subject")
            .agg({"score": "mean"})
            .reset_index()
        )
        results_dict = {
            "deployment": self.model,
            "test": "MMLU",
            "overall_score": results.loc[:, "score"].mean(),
        }

        return pd.DataFrame([results_dict])


class PubMedQA(Eval):
    """
    This is a class implementing the PubMedQA benchmark evaluation.
    Inputs:
        deployment_config:
            key: AOAI API key
            endpoint: AOAI endpoint
            model: AOAI deployment name
            version: API version
        sample_size: fraction of data to sample for evaluation (optional)
        log_level: logging level (optional)

    Inhereted Methods:
        _score: Compare the generated answer to the correct answer
        _load_data: Load the dataset from the benchmark into memory

    Private Methods:
        __transform_data: Transform the dataset into a format that can be used by the Azure OpenAI API
        __call_aoai: Call the Azure OpenAI API to generate an answer

    Public Method:
        test: Run the PubMedQA evaluation and output a dataframe

    """

    def __init__(
        self, deployment_config: dict, sample_size: float = 1.0, log_level: str = "INFO"
    ):
        super().__init__(
            deployment_config=deployment_config,
            sample_size=sample_size,
            log_level=log_level,
        )

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Take subset of data
        self.logger.info(f"Sampling data to {self.sample_size*100}% ")
        df = df.sample(frac=self.sample_size, replace=False).reset_index()
        return df

    def __call_aoai(self, row: list) -> dict:
        client = AzureOpenAI(
            azure_endpoint=self._base, api_key=self._key, api_version=self.version
        )

        sys_message = "Complete the given problem to the best of your ability. \
                    Accuracy is very important. \
                    Given a context, answer the research question with either a yes, no, or maybe \
                    THe context will be a list of strings. Each string is some relevant information to inform your decision\
                    Select ONLY ONE answer from the following choices [yes, no, maybe]. \
                    Your answer must be a single word, all lowercase, do not use quotations"

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_message},
                {
                    "role": "user",
                    "content": f"Question: {row['question']}.  Context: {row['context.contexts']}. Answer:",
                },
            ],
        )
        output = {
            "generated": response.choices[0].message.content,
            "correct": row["final_decision"],
        }

        return output

    async def test(self, data: pd.DataFrame) -> pd.DataFrame:
        output_list = []
        self.logger.info("Starting evaluation")
        for index, row in data.iterrows():
            self.logger.info(f"Evaluating row {index} of {data.shape[0]}")
            try:
                output = self.__call_aoai(row)
                output["score"] = self._score(output["generated"], output["correct"])
                output_list.append(output)
            except Exception as e:
                self.logger.warning(f"Skipping...error in row {index}: {e}")

        self.logger.info("Evaluation complete.")
        results = pd.DataFrame(output_list).reset_index()
        results_dict = {
            "deployment": self.model,
            "test": "MedPub QA",
            "overall_score": results.loc[:, "score"].mean(),
        }

        return pd.DataFrame([results_dict])


class TruthfulQA(Eval):
    """
    This is a class implementing the Truthful QA benchmark evaluation.
    Inputs:
        deployment_config:
            key: AOAI API key
            endpoint: AOAI endpoint
            model: AOAI deployment name
            version: API version
        sample_size: fraction of data to sample for evaluation (optional)
        log_level: logging level (optional)

    Inhereted Methods:
        _score: Compare the generated answer to the correct answer
        _load_data: Load the dataset from the benchmark into memory

    Private Methods:
        __transform_data: Transform the dataset into a format that can be used by the Azure OpenAI API
        __call_aoai: Call the Azure OpenAI API to generate an answer

    Public Method:
        test: Run the PubMedQA evaluation and output a dataframe

    """

    def __init__(
        self, deployment_config: dict, sample_size: float = 1.0, log_level: str = "INFO"
    ):
        super().__init__(
            deployment_config=deployment_config,
            sample_size=sample_size,
            log_level=log_level,
        )

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Take subset of data
        self.logger.info(f"Sampling data to {self.sample_size*100}% ")
        df = df.sample(frac=self.sample_size, replace=False).reset_index()
        return df

    def __call_aoai(self, row: list) -> dict:
        client = AzureOpenAI(
            azure_endpoint=self._base, api_key=self._key, api_version=self.version
        )

        sys_message = "Complete the given problem to the best of your ability. \
                    Accuracy is very important. \
                    Choices are a list of quoted strings with a starting index of 0 \
                    Select ONLY ONE answer from the choices. \
                    Return ONLY the index of the correct answer in the choices list. Your answer must be a single ineteger."

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_message},
                {
                    "role": "user",
                    "content": f"Question: {row['question']}.  Choices: {row['mc1_targets']['choices']}. Answer:",
                },
            ],
        )
        output = {
            "generated": response.choices[0].message.content,
            "correct": list(row["mc1_targets"]["labels"]).index(1),
        }

        return output

    async def test(self, data: pd.DataFrame) -> pd.DataFrame:
        output_list = []
        self.logger.info("Starting evaluation")
        for index, row in data.iterrows():
            self.logger.info(f"Evaluating row {index} of {data.shape[0]}")
            try:
                output = self.__call_aoai(row)
                output["score"] = self._score(output["generated"], output["correct"])
                output_list.append(output)
            except Exception as e:
                self.logger.warning(f"Skipping...error in row {index}: {e}")

        self.logger.info("Evaluation complete.")
        results = pd.DataFrame(output_list).reset_index()
        results_dict = {
            "deployment": self.model,
            "test": "Truthful QA",
            "overall_score": results.loc[:, "score"].mean(),
        }

        return pd.DataFrame([results_dict])


class CustomEval(Eval):
    """
    This is a class implementing Custom evaluation.
    Inputs:
        deployment_config:
            key: AOAI API key
            endpoint: AOAI endpoint
            model: AOAI deployment name
            version: API version
        custom_data: Custom data to evaluate
        metrics_list: List of metrics to evaluate
        sample_size: fraction of data to sample for evaluation (optional)
        log_level: logging level (optional)

    Private Methods:
        __call_aoai: Call the Azure OpenAI API to generate an answer
        __custom_score: Custom scoring function
            - Note: This function leverages the metrics modules

    Public Method:
        test: Run the custom evaluation and output a dataframe

    """

    def __init__(
        self,
        deployment_config: Dict,
        metrics_list: List,
        sample_size: float = 1.0,
        log_level: str = "INFO",
    ):
        super().__init__(
            deployment_config=deployment_config,
            sample_size=sample_size,
            log_level=log_level,
        )
        self.metrics_list = [x.lower().replace(" ", "_") for x in metrics_list]

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Take subset of data
        self.logger.info(f"Sampling data to {self.sample_size*100}% ")
        df = df.sample(frac=self.sample_size, replace=False).reset_index()
        return df

    def __call_aoai(self, row: Dict) -> Dict:
        client = AzureOpenAI(
            azure_endpoint=self._base, api_key=self._key, api_version=self.version
        )

        if "context" in row.keys():
            messages = [
                {"role": "user", "content": f"{row['context']}"},
                {"role": "system", "content": f"{row['prompt']}"},
            ]
        else:
            messages = [{"role": "user", "content": f"{row['prompt']}"}]

        response = client.chat.completions.create(model=self.model, messages=messages)

        row["answer"] = response.choices[0].message.content
        logging.debug(f"aoai call row: {row.keys()}")

        return row

    # Add other custom metrics here
    def __custom_score(self, row: dict) -> dict:
        if "accuracy" in self.metrics_list:
            row["accuracy"] = accuracy(row["ground_truth"], row["answer"])
        if "answer_similarity" in self.metrics_list:
            row["answer_similarity"] = sentence_transformer_similarity(
                row["ground_truth"], row["answer"]
            )
        if "context_similarity" in self.metrics_list:
            row["context_similarity"] = sentence_transformer_similarity(
                row["context"], row["answer"]
            )

        logging.debug(f"custom score row: {row.keys()}")
        return row

    async def test(self, data: pd.DataFrame) -> pd.DataFrame:
        output_list = []
        self.logger.info("Starting custom evaluation")
        for index, row in data.iterrows():
            self.logger.info(f"Evaluating row {index} of {data.shape[0]}")
            try:
                output = self.__call_aoai(row)
                output_list.append(output)
            except Exception as e:
                self.logger.warning(f"Skipping...error in row {index}: {e}")

        self.logger.info("Evaluation complete.")
        results = pd.DataFrame(output_list).reset_index()

        results = results.apply(lambda x: self.__custom_score(x), axis=1)

        self.logger.info("Aggregating Results")
        results_dict = []
        for metric in self.metrics_list:
            results_dict.append(
                {
                    "deployment": self.model,
                    "test": f"custom {metric}",
                    "overall_score": results.loc[:, metric].mean(),
                }
            )

        return pd.DataFrame(results_dict)


if __name__ == "__main__":
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())
    deploy_dict = {
        "model": os.getenv("AOAI_MODEL"),
        "endpoint": os.getenv("AOAI_ENDPOINT"),
        "key": os.getenv("AOAI_KEY"),
        "version": "2024-02-01",
    }

    result_list = []

    """pubmed_eval = PubMedQA(deploy_dict, sample_size=0.005, log_level="INFO")
    result_list.append(asyncio.run(pubmed_eval.test()))

    mmlu_eval = MMLU(deploy_dict, sample_size=0.005, categories = ['Business'], log_level="INFO")
    result_list.append(asyncio.run(mmlu_eval.test()))

    truthfulqa_eval = TruthfulQA(deploy_dict, sample_size=0.005, log_level="Info")
    result_list.append(asyncio.run(truthfulqa_eval.test()))"""

    test_df = pd.read_csv("src/quality/sample_data/openai_humaneval.csv")
    test_df = test_df.drop(columns=["Unnamed: 0", "task_id", "entry_point", "test"])
    custom_eval = CustomEval(
        deployment_config=deploy_dict,
        custom_data=test_df,
        metrics_list=["Accuracy"],
        sample_size=0.01,
        log_level="INFO",
    )
    result_list.append(asyncio.run(custom_eval.test()))

    for result in result_list:
        print(f"Results: \n{result}")

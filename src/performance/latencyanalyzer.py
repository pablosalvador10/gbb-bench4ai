from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from my_utils.ml_logging import get_logger

# Set up logger
logger = get_logger()


class BenchmarkVisualizer:
    def __init__(self, benchmark_results: Dict[str, Any]) -> None:
        """
        Initialize the BenchmarkVisualizer with benchmark results.

        :param benchmark_results: Dictionary containing benchmarking results.
        """
        self.benchmark_results = benchmark_results
        self.df = self._create_dataframe()
        self.model_dfs = self._create_model_dfs()

    def _create_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame from the benchmark results.

        :return: DataFrame containing all benchmark data.
        """
        try:
            all_data = []
            for key, value in self.benchmark_results.items():
                model_name, tokens = key.rsplit("_", 1)
                for i in range(len(next(iter(value.values())))):
                    all_data.append(
                        {
                            "model_name": model_name,
                            "tokens": int(tokens),
                            "ttlt_successfull": value["ttlt_successfull"][i],
                            "completion_tokens": value["completion_tokens"][i],
                            "prompt_tokens": value["prompt_tokens"][i],
                        }
                    )
            df = pd.DataFrame(all_data)
            logger.info("DataFrame created successfully")
            return df
        except Exception as e:
            logger.error(f"Error in creating DataFrame: {e}")
            raise

    def _create_model_dfs(self) -> Dict[str, pd.DataFrame]:
        """
        Create a dictionary of DataFrames for each unique model.

        :return: Dictionary with model names as keys and DataFrames as values.
        """
        try:
            model_dfs = {}
            self.unique_model_names = self.df["model_name"].unique()
            for model_name in self.unique_model_names:
                model_dfs[model_name] = self.df[self.df["model_name"] == model_name]
            logger.info("Model DataFrames created successfully")
            return model_dfs
        except Exception as e:
            logger.error(f"Error in creating model DataFrames: {e}")
            raise

    def _calculate_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the correlation matrix for the provided DataFrame.

        :param df: DataFrame for which the correlation matrix is calculated.
        :return: Correlation matrix DataFrame.
        """
        try:
            correlation_matrix = df[
                ["ttlt_successfull", "completion_tokens", "prompt_tokens"]
            ].corr()
            logger.info("Correlation matrix calculated successfully")
            return correlation_matrix
        except Exception as e:
            logger.error(f"Error in calculating correlation matrix: {e}")
            raise

    def _calculate_residuals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate residuals for completion tokens and prompt tokens for the provided DataFrame.

        :param df: DataFrame for which residuals are calculated.
        :return: DataFrame with residuals columns added.
        """
        try:
            slope, intercept, _, _, _ = stats.linregress(
                df["completion_tokens"], df["ttlt_successfull"]
            )
            df["Residuals Completion Tokens"] = df["ttlt_successfull"] - (
                slope * df["completion_tokens"] + intercept
            )

            slope, intercept, _, _, _ = stats.linregress(
                df["prompt_tokens"], df["ttlt_successfull"]
            )
            df["Residuals Prompt Tokens"] = df["ttlt_successfull"] - (
                slope * df["prompt_tokens"] + intercept
            )

            logger.info("Residuals calculated successfully")
            return df
        except Exception as e:
            logger.error(f"Error in calculating residuals: {e}")
            raise

    def _identify_top_outliers(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identify the top outliers for completion tokens and prompt tokens residuals for the provided DataFrame.

        :param df: DataFrame for which top outliers are identified.
        :return: Two DataFrames containing the top outliers for completion tokens and prompt tokens.
        """
        try:
            top_outliers_completion = df.iloc[
                np.abs(df["Residuals Completion Tokens"]).argsort()[-5:]
            ]
            top_outliers_prompt = df.iloc[
                np.abs(df["Residuals Prompt Tokens"]).argsort()[-5:]
            ]
            logger.info("Top outliers identified successfully")
            return top_outliers_completion, top_outliers_prompt
        except Exception as e:
            logger.error(f"Error in identifying top outliers: {e}")
            raise

    def plot_correlation_matrix(self, model_name: str) -> plt:
        """
        Plot the correlation matrix as a heatmap for the specified model.

        :param model_name: Name of the model for which the correlation matrix is plotted.
        :return: Matplotlib plot object.
        """
        try:
            df = self.model_dfs[model_name]
            correlation_matrix = self._calculate_correlation_matrix(df)
            plt.figure(figsize=(12, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title(f"Correlation Matrix for {model_name}")
            logger.info(
                f"Correlation matrix plot created successfully for {model_name}"
            )
            return plt
        except Exception as e:
            logger.error(f"Error in plotting correlation matrix for {model_name}: {e}")
            raise

    def plot_completion_tokens(self, model_name: str) -> plt:
        """
        Plot TTLT Successful vs Completion Tokens with regression line and top outliers annotated for the specified model.

        :param model_name: Name of the model for which the plot is created.
        :return: Matplotlib plot object.
        """
        try:
            df = self.model_dfs[model_name]
            df = self._calculate_residuals(df)
            top_outliers_completion, _ = self._identify_top_outliers(df)
            correlation_matrix = self._calculate_correlation_matrix(df)

            plt.figure(figsize=(12, 6))
            sns.regplot(
                x="completion_tokens",
                y="ttlt_successfull",
                data=df,
                color="blue",
                label="Completion Tokens vs TTLT",
            )
            for i in range(top_outliers_completion.shape[0]):
                plt.annotate(
                    f"{top_outliers_completion.iloc[i]['completion_tokens']:.2f}, {top_outliers_completion.iloc[i]['ttlt_successfull']:.2f}",
                    (
                        top_outliers_completion.iloc[i]["completion_tokens"],
                        top_outliers_completion.iloc[i]["ttlt_successfull"],
                    ),
                )
            plt.xlabel("Completion Tokens")
            plt.ylabel("TTLT Successful")
            plt.title(f"TTLT Successful vs Completion Tokens for {model_name}")
            plt.legend()
            plt.text(
                0.2,
                0.8,
                f"Correlation: {correlation_matrix.loc['completion_tokens', 'ttlt_successfull']:.2f}",
                transform=plt.gca().transAxes,
            )
            plt.text(
                0.2, 0.7, f"Note: Data from {model_name}", transform=plt.gca().transAxes
            )
            logger.info(f"Completion tokens plot created successfully for {model_name}")
            return plt
        except Exception as e:
            logger.error(f"Error in plotting completion tokens for {model_name}: {e}")
            raise

    def plot_prompt_tokens(self, model_name: str) -> plt:
        """
        Plot TTLT Successful vs Prompt Tokens with regression line and top outliers annotated for the specified model.

        :param model_name: Name of the model for which the plot is created.
        :return: Matplotlib plot object.
        """
        try:
            df = self.model_dfs[model_name]
            df = self._calculate_residuals(df)
            _, top_outliers_prompt = self._identify_top_outliers(df)
            correlation_matrix = self._calculate_correlation_matrix(df)

            plt.figure(figsize=(12, 6))
            sns.regplot(
                x="prompt_tokens",
                y="ttlt_successfull",
                data=df,
                color="red",
                label="Prompt Tokens vs TTLT",
            )
            for i in range(top_outliers_prompt.shape[0]):
                plt.annotate(
                    f"{top_outliers_prompt.iloc[i]['prompt_tokens']:.2f}, {top_outliers_prompt.iloc[i]['ttlt_successfull']:.2f}",
                    (
                        top_outliers_prompt.iloc[i]["prompt_tokens"],
                        top_outliers_prompt.iloc[i]["ttlt_successfull"],
                    ),
                )
            plt.xlabel("Prompt Tokens")
            plt.ylabel("TTLT Successful")
            plt.title(f"TTLT Successful vs Prompt Tokens for {model_name}")
            plt.legend()
            plt.text(
                0.2,
                0.8,
                f"Correlation: {correlation_matrix.loc['prompt_tokens', 'ttlt_successfull']:.2f}",
                transform=plt.gca().transAxes,
            )
            plt.text(
                0.2, 0.7, f"Note: Data from {model_name}", transform=plt.gca().transAxes
            )
            logger.info(f"Prompt tokens plot created successfully for {model_name}")
            return plt
        except Exception as e:
            logger.error(f"Error in plotting prompt tokens for {model_name}: {e}")
            raise

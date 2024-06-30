from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.ml_logging import get_logger

# Set up logger
logger = get_logger()

class ModelPerformanceVisualizer:
    def __init__(self, data: Dict[str, Any]) -> None:
        """
        Initialize the visualizer with the given data.

        :param data: A dictionary containing the performance data for each model.
        """
        self.data = data
        self.df = pd.DataFrame()

    def transpose_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parse the JSON data into a DataFrame, flattening the best_run and worst_run dictionaries.
        Separate the best_run and worst_run into a different DataFrame.
        Return the string representations of both DataFrames.
        """
        records = []
        best_worst_records = []
        for model, stats in self.data.items():
            flattened_stats = {"model": model}
            best_worst_stats = {"model": model}
            for key, value in stats.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if key in ["best_run", "worst_run"]:
                            best_worst_stats[f"{key}_{sub_key}"] = sub_value
                        else:
                            flattened_stats[f"{key}_{sub_key}"] = sub_value
                else:
                    flattened_stats[key] = value
            records.append(flattened_stats)
            best_worst_records.append(best_worst_stats)
        df = pd.DataFrame.from_records(records)
        df.rename(columns={"model": "ModelName_MaxTokens"}, inplace=True)
        df.set_index("ModelName_MaxTokens", inplace=True)
        df_best_and_worst = pd.DataFrame.from_records(best_worst_records)
        df_best_and_worst.rename(columns={"model": "ModelName_MaxTokens"}, inplace=True)
        df_best_and_worst.set_index("ModelName_MaxTokens", inplace=True)
        return df, df_best_and_worst

    def parse_data(self) -> None:
        """
        Parse the JSON data into a DataFrame for easier manipulation.
        """
        records = []
        for model, stats in self.data.items():
            record = {**{"model": model}, **stats}
            # Flatten nested dictionaries like best_run and worst_run
            for key in ["best_run", "worst_run"]:
                if key in stats:
                    for subkey, value in stats[key].items():
                        record[f"{key}_{subkey}"] = value
            # Flatten other nested dictionaries
            for key, value in stats.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        record[f"{key}_{subkey}"] = subvalue
            records.append(record)
        self.df = pd.DataFrame.from_records(records)

    def plot_times(self):
        """
        Plot response times for comparison and return the figure objects.
        """
        # First figure for boxplot
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        sns.boxplot(x="model", y="median_ttlt", data=self.df, ax=ax1)
        ax1.set_title("Median Response Time by Model")
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylabel("Time (s)")

        # Second figure for pairplot
        fig2 = sns.pairplot(
            self.df,
            vars=[
                "median_ttlt",
                "percentile_95_ttlt",
                "percentile_99_ttlt",
            ],
            hue="model",
        )

        return fig1, fig2

    def plot_tokens(self) -> plt.Figure:
        """
        Plot token statistics and return the figure object.
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        token_features = ["median_prompt_tokens", "median_completion_tokens"]
        melted_df = self.df.melt(id_vars=["model"], value_vars=token_features)
        sns.barplot(x="model", y="value", hue="variable", data=melted_df, ax=ax)
        ax.set_title("Token Metrics by Model")
        ax.set_ylabel("Tokens")
        ax.set_xlabel("Model")
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig

    def plot_errors(self) -> plt.Figure:
        """
        Plot error rates and return the figure object.
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        sns.barplot(x="model", y="error_rate", data=self.df, ax=ax)
        ax.set_title("Error Rate by Model")
        ax.set_ylabel("Error Rate (%)")
        ax.set_xlabel("Model")
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig

    def plot_best_worst_runs(self) -> plt.Figure:
        """
        Compare the best and worst run times and return the figure object.
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        # Check if columns exist in the DataFrame
        if "best_run_time" in self.df.columns and "worst_run_time" in self.df.columns:
            melted_df = self.df.melt(id_vars=["model"], value_vars=["best_run_time", "worst_run_time"])
            sns.barplot(x="model", y="value", hue="variable", data=melted_df, ax=ax)
            ax.set_title("Best vs Worst Run Times by Model")
            ax.set_ylabel("Time (s)")
            ax.set_xlabel("Model")
            ax.tick_params(axis='x', rotation=45)
        else:
            logger.error("The columns 'best_run_time' and 'worst_run_time' are not present in the DataFrame")

        plt.tight_layout()
        return fig

    def plot_heatmaps(self) -> plt.Figure:
        """
        Plot a heatmap of performance metrics by region and model, if regions data exists, and return the figure object.
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        if "regions" in self.df.columns:
            sns.heatmap(
                self.df.pivot_table(
                    index="model", columns="regions", values="median_time"
                ),
                annot=True,
                fmt=".1f",
                cmap="coolwarm",
                ax=ax
            )
            ax.set_title("Performance Heatmap by Region and Model")
        else:
            logger.info("No regional data available for heatmap.")
        return fig

    def plot_time_vs_tokens(self) -> plt.Figure:
        """
        Plot the relationship between time and tokens and return the figure object.
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        sns.scatterplot(x="median_prompt_tokens", y="median_ttlt", hue="model", data=self.df, ax=ax)
        ax.set_title("Response Time vs. Prompt Tokens by Model")
        ax.set_xlabel("Median Prompt Tokens")
        ax.set_ylabel("Median Time (s)")

        plt.tight_layout()
        return fig

    def visualize_all(self) -> None:
        """
        Visualize all the data by generating all plots.
        """
        self.parse_data()

        fig1, fig2 = self.plot_times()
        fig_tokens = self.plot_tokens()
        fig_errors = self.plot_errors()
        fig_best_worst = self.plot_best_worst_runs()
        fig_heatmap = self.plot_heatmaps()
        fig_time_vs_tokens = self.plot_time_vs_tokens()

        fig1.show()
        fig2.savefig('pairplot.png')
        fig_tokens.show()
        fig_errors.show()
        fig_best_worst.show()
        fig_heatmap.show()
        fig_time_vs_tokens.show()
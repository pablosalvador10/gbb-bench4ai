import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from typing import Dict, Any, Tuple
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
        self.models = list(data.keys())
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
            flattened_stats = {'model': model}
            best_worst_stats = {'model': model}
            for key, value in stats.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if key in ['best_run', 'worst_run']:
                            best_worst_stats[f"{key}_{sub_key}"] = sub_value
                        else:
                            flattened_stats[f"{key}_{sub_key}"] = sub_value
                else:
                    flattened_stats[key] = value
            records.append(flattened_stats)
            best_worst_records.append(best_worst_stats)
        df = pd.DataFrame.from_records(records)
        df.rename(columns={'model': 'ModelName_MaxTokens'}, inplace=True)
        df.set_index('ModelName_MaxTokens', inplace=True)
        df = df.transpose()
        df_best_and_worst = pd.DataFrame.from_records(best_worst_records)
        df_best_and_worst.rename(columns={'model': 'ModelName_MaxTokens'}, inplace=True)
        df_best_and_worst.set_index('ModelName_MaxTokens', inplace=True)
        df_best_and_worst = df_best_and_worst.transpose()
        return df, df_best_and_worst
        

    def parse_data(self) -> None:
        """
        Parse the JSON data into a DataFrame for easier manipulation.
        """
        records = []
        for model, stats in self.data.items():
            record = {**{'model': model}, **stats}
            # Flatten nested dictionaries like best_run and worst_run
            for key in ['best_run', 'worst_run']:
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

    def plot_times(self) -> None:
        """
        Plot response times for comparison.
        """
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="model", y="median_time", data=self.df)
        plt.title("Median Response Time by Model")
        plt.xticks(rotation=45)
        plt.ylabel("Time (s)")
        plt.show()

        # Detailed time plots
        sns.pairplot(self.df, vars=["median_time", "iqr_time", "percentile_95_time", "percentile_99_time"], hue="model")
        plt.show()

    def plot_tokens(self) -> None:
        """
        Plot token statistics.
        """
        token_features = ["median_prompt_tokens", "median_completion_tokens"]
        self.df.melt(id_vars=["model"], value_vars=token_features)
        sns.catplot(x="model", y="value", hue="variable", data=self.df.melt(id_vars="model", value_vars=token_features), kind="bar")
        plt.title("Token Metrics by Model")
        plt.xticks(rotation=45)
        plt.show()

    def plot_errors(self) -> None:
        """
        Plot error rates.
        """
        sns.barplot(x="model", y="error_rate", data=self.df)
        plt.title("Error Rate by Model")
        plt.xticks(rotation=45)
        plt.ylabel("Error Rate (%)")
        plt.show()

    def plot_best_worst_runs(self) -> None:
        """
        Compare the best and worst run times.
        """
        self.df[['model', 'best_run_time', 'worst_run_time']].set_index('model').plot(kind='bar', figsize=(12, 6))
        plt.title("Best vs Worst Run Times")
        plt.ylabel("Time (s)")
        plt.show()

    def plot_heatmaps(self) -> None:
        """
        Plot a heatmap of performance metrics by region and model, if regions data exists.
        """
        try:
            if 'regions' in self.df.columns:
                plt.figure(figsize=(10, 8))
                sns.heatmap(self.df.pivot_table(index='model', columns='regions', values='median_time'), annot=True, fmt=".1f", cmap="coolwarm")
                plt.title("Performance Heatmap by Region and Model")
                plt.show()
            else:
                logger.info("No regional data available for heatmap.")
        except Exception as e:
            logger.error(f"An error occurred while plotting the heatmap: {e}")

    def visualize_all(self) -> None:
        """
        Visualize all the data.
        """
        self.parse_data()
        self.plot_times()
        self.plot_tokens()
        self.plot_errors()
        self.plot_best_worst_runs()
        self.plot_heatmaps()
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

    def parse_data(self) -> None:
        """
        Parse the JSON data into a DataFrame for easier manipulation.
        """
        records = []
        for model, stats in self.data.items():
            record = {"model": model}
            for key, value in stats.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        record[f"{key}_{subkey}"] = subvalue
                else:
                    record[key] = value
            records.append(record)
        self.df = pd.DataFrame.from_records(records)
        logger.info(f"DataFrame after parsing: {self.df.head()}")

    def plot_completion_tokens(self):
        """
        Plot completion tokens by model using Plotly, with enhanced median representation.
        """
        fig_width, fig_height = 1200, 600
        fig = px.box(
            self.df,
            x="model",
            y="median_completion_tokens",
            title="Completion Tokens by Model",
            template="plotly_white",
            notched=True,
            points="all",
        )

        fig.add_trace(
            go.Scatter(
                x=self.df["model"],
                y=self.df["median_completion_tokens"],
                mode="markers+text",
                name="Median Tokens 🤖",
                marker=dict(color="lightblue", size=10, symbol="circle"),
                text=["🤖" for _ in self.df["model"]],
                textposition="bottom center",
                hoverinfo="text+name",
                hovertext=self.df.apply(
                    lambda row: f"Model: {row['model']}\nMedian Tokens: {row['median_completion_tokens']}",
                    axis=1,
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=self.df["model"],
                y=self.df["average_ttlt"],
                mode="markers+text",
                name="⚡Avg TTL",
                marker=dict(color="blue", size=12, symbol="circle"),
                text=["⚡" for _ in self.df["average_ttlt"]],
                textposition="top center",
                textfont=dict(color="blue", size=16),
                hovertext=self.df.apply(
                    lambda row: f"{row['model']}: {row['average_ttlt']:.2f}s", axis=1
                ),
                hoverinfo="text",
            )
        )

        fig.update_xaxes(tickangle=45)
        fig.update_layout(autosize=False, width=fig_width, height=fig_height)
        logger.info(f"Plot completion tokens figure: {fig}")
        return fig

    def plot_prompt_tokens(self):
        """
        Plot prompt tokens by model using Plotly, with enhanced median representation and human emoji.
        """
        fig_width, fig_height = 1200, 600
        fig = px.box(
            self.df,
            x="model",
            y="median_prompt_tokens",
            title="Prompt Tokens by Model",
            template="plotly_white",
        )

        fig.add_trace(
            go.Scatter(
                x=self.df["model"],
                y=self.df["average_ttlt"],
                mode="markers+text",
                name="⚡Avg TTL",
                marker=dict(color="blue", size=12, symbol="circle"),
                text=["⚡" for _ in self.df["average_ttlt"]],
                textposition="top center",
                textfont=dict(color="blue", size=16),
                hovertext=self.df.apply(
                    lambda row: f"{row['model']}: {row['average_ttlt']:.2f}s", axis=1
                ),
                hoverinfo="text",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=self.df["model"],
                y=self.df["median_prompt_tokens"],
                mode="markers+text",
                name="Median Tokens 👤",
                marker=dict(color="lightblue", size=10, symbol="circle"),
                text=["👤" for _ in self.df["model"]],
                textposition="bottom center",
                hoverinfo="text+name",
                hovertext=self.df.apply(
                    lambda row: f"Model: {row['model']}\nMedian Prompt Tokens: {row['median_prompt_tokens']}",
                    axis=1,
                ),
            )
        )

        fig.update_xaxes(tickangle=45)
        fig.update_layout(autosize=False, width=fig_width, height=fig_height)
        logger.info(f"Plot prompt tokens figure: {fig}")
        return fig

    def plot_response_time_metrics_comparison(self):
        """
        Plot a comparison of response time metrics by model using Plotly, including generation tokens and input tokens with emojis.
        Adjusted to use a secondary y-axis for token metrics.
        """
        fig_width, fig_height = 1200, 600
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for metric in [
            "median_ttlt",
            "average_ttlt",
            "percentile_95_ttlt",
            "percentile_99_ttlt",
        ]:
            fig.add_trace(
                go.Bar(x=self.df["model"], y=self.df[metric], name=metric),
                secondary_y=False,
            )

        fig.add_trace(
            go.Scatter(
                x=self.df["model"],
                y=self.df["median_prompt_tokens"],
                mode="markers+text",
                name="Median Prompt Tokens 👤",
                marker=dict(color="lightblue", size=10, symbol="circle"),
                text=["👤" for _ in self.df["model"]],
                textposition="bottom center",
                hoverinfo="text+name",
                hovertext=self.df.apply(
                    lambda row: f"Model: {row['model']}\nMedian Prompt Tokens: {row['median_prompt_tokens']}",
                    axis=1,
                ),
            ),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(
                x=self.df["model"],
                y=self.df["median_completion_tokens"],
                mode="markers+text",
                name="Median Completion Tokens 🤖",
                marker=dict(color="lightblue", size=10, symbol="circle"),
                text=["🤖" for _ in self.df["model"]],
                textposition="bottom center",
                hoverinfo="text+name",
                hovertext=self.df.apply(
                    lambda row: f"Model: {row['model']}\nMedian Completion Tokens: {row['median_completion_tokens']}",
                    axis=1,
                ),
            ),
            secondary_y=True,
        )

        fig.update_layout(
            barmode="group",
            title="Response Time Metrics by Model",
            xaxis_title="Model",
            yaxis_title="Time (seconds)",
            autosize=False,
            width=fig_width,
            height=fig_height,
            template="plotly_white",
            title_font_size=20,
        )

        fig.update_yaxes(title_text="Token Count", secondary_y=True)
        logger.info(f"Plot response time metrics comparison figure: {fig}")
        return fig

    def plot_tokens(self):
        """
        Plot token statistics using Plotly and return the figure object.
        """
        token_features = ["median_prompt_tokens", "median_completion_tokens"]
        melted_df = self.df.melt(id_vars=["model"], value_vars=token_features)
        fig = px.bar(
            melted_df,
            x="model",
            y="value",
            color="variable",
            barmode="group",
            labels={"value": "Tokens", "variable": "Token Type"},
            title="Token Metrics by Model",
        )
        fig.update_xaxes(tickangle=45)
        logger.info(f"Plot tokens figure: {fig}")
        return fig

    def plot_errors(self):
        """
        Plot error rates using Plotly and return the figure object.
        """
        fig = px.bar(self.df, x="model", y="error_rate", title="Error Rate by Model")
        fig.update_traces(marker_color="indianred")
        fig.update_xaxes(tickangle=45)
        logger.info(f"Plot errors figure: {fig}")
        return fig

    def plot_best_worst_runs(self):
        """
        Improved function to compare the best and worst run times using Plotly, including detailed traces for input and output tokens.
        This version uses a dual-axis plot to handle the different scales between run times and token counts.
        """
        fig_width, fig_height = 1200, 600
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Plotting best and worst run times on the primary y-axis
        for run_type, color in zip(
            ["best_run_ttlt", "worst_run_ttlt"], ["green", "red"]
        ):
            fig.add_trace(
                go.Bar(
                    x=self.df["model"],
                    y=self.df[run_type],
                    name=run_type,
                    marker_color=color,
                ),
                secondary_y=False,
            )

        # Adding scatter traces for prompt and completion tokens in the worst runs on the secondary y-axis
        token_types = [
            ("worst_run_prompt_tokens", "👤", "orange"),
            ("worst_run_completion_tokens", "🤖", "blue"),
        ]
        for token_type, emoji, color in token_types:
            fig.add_trace(
                go.Scatter(
                    x=self.df["model"],
                    y=self.df[token_type],
                    mode="markers+text",
                    name=f"{token_type} {emoji}",
                    marker=dict(color=color, size=10),
                    text=[emoji for _ in self.df["model"]],
                    textposition="bottom center",
                    hoverinfo="text+name",
                    hovertext=self.df.apply(
                        lambda row: f"Model: {row['model']}\n{token_type}: {row[token_type]}",
                        axis=1,
                    ),
                ),
                secondary_y=True,
            )

        token_types_best = [
            ("best_run_prompt_tokens", "👤", "lightgreen"),
            ("best_run_completion_tokens", "🤖", "cyan"),
        ]
        for token_type, emoji, color in token_types_best:
            fig.add_trace(
                go.Scatter(
                    x=self.df["model"],
                    y=self.df[token_type],
                    mode="markers+text",
                    name=f"{token_type} {emoji}",
                    marker=dict(color=color, size=10),
                    text=[emoji for _ in self.df["model"]],
                    textposition="bottom center",
                    hoverinfo="text+name",
                    hovertext=self.df.apply(
                        lambda row: f"Model: {row['model']}\n{token_type}: {row[token_type]}",
                        axis=1,
                    ),
                ),
                secondary_y=True,
            )

        # Update layout for clarity, readability, and aesthetics
        fig.update_layout(
            barmode="group",
            title="Best vs Worst Run Times by Model with Token Details",
            xaxis_title="Model",
            yaxis_title="Run Time (seconds)",
            autosize=False,
            width=fig_width,
            height=fig_height,
            template="plotly_white",
            title_font_size=20,
        )

        fig.update_yaxes(title_text="Token Count", secondary_y=True)

        logger.info(f"Plot best vs worst runs with dual axis figure: {fig}")
        return fig

    def plot_heatmaps(self):
        """
        Plot a heatmap of performance metrics by region and model using Plotly and return the figure object.
        """
        if "regions" in self.df.columns:
            fig = px.imshow(
                self.df.pivot_table(
                    index="model", columns="regions", values="median_time"
                ),
                labels=dict(x="Region", y="Model", color="Median Time"),
                title="Performance Heatmap by Region and Model",
            )
        else:
            logger.info("No regional data available for heatmap.")
            fig = None
        logger.info(f"Plot heatmaps figure: {fig}")
        return fig

    def plot_time_vs_tokens(self):
        """
        Plot the relationship between time and tokens using Plotly and return the figure object.
        """
        fig = px.scatter(
            self.df,
            x="median_prompt_tokens",
            y="median_ttlt",
            color="model",
            title="Response Time vs. Prompt Tokens by Model",
        )
        logger.info(f"Plot time vs tokens figure: {fig}")
        return fig

    def visualize_all(self):
        """
        Visualize all the data by generating all plots using Plotly.
        """
        self.parse_data()

        fig1 = self.plot_completion_tokens()
        fig2 = self.plot_prompt_tokens()
        fig3 = self.plot_response_time_metrics_comparison()
        fig_tokens = self.plot_tokens()
        fig_errors = self.plot_errors()
        fig_best_worst = self.plot_best_worst_runs()
        fig_heatmap = self.plot_heatmaps()
        fig_time_vs_tokens = self.plot_time_vs_tokens()

        fig1.show()
        fig2.show()
        fig3.show()
        fig_tokens.show()
        fig_errors.show()
        if fig_best_worst is not None:
            fig_best_worst.show()
        if fig_heatmap is not None:
            fig_heatmap.show()
        fig_time_vs_tokens.show()

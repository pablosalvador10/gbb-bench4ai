from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
            notched=True,  # Adds a notch to indicate the confidence interval around the median
            points='all'  # Shows all points to provide a fuller picture of the distribution
        )
        
        # Adjusted custom markers for median values with detailed hover information and robot emoji
        fig.add_trace(
            go.Scatter(
                x=self.df["model"],
                y=self.df['median_completion_tokens'],
                mode='markers+text',  # Combine markers and text for display
                name='Median Tokens 🤖',  # Updated name with robot emoji
                marker=dict(
                    color='lightblue',  # Use light blue color for a lighter appearance
                    size=10,  # Adjust size as needed
                    symbol='circle'  # Use 'circle' symbol for a softer look
                ),
                text=["🤖" for _ in self.df['model']],  # Use robot emoji for each point
                textposition="bottom center",  # Position text below the marker
                hoverinfo='text+name',  # Show custom text and name on hover
                hovertext=self.df.apply(lambda row: f"Model: {row['model']}\nMedian Tokens: {row['median_completion_tokens']}", axis=1)  # Detailed hover text
            )
        )

         # Add average TTL as a separate trace with lightning emoji markers
        fig.add_trace(
            go.Scatter(
                x=self.df["model"],
                y=self.df['average_ttlt'],
                mode='markers+text',
                name='⚡Avg TTL',
                marker=dict(
                    color='blue',
                    size=12,
                    symbol='circle'
                ),
                text=["⚡" for _ in self.df['average_ttlt']],
                textposition="top center",
                textfont=dict(
                    color='blue',
                    size=16
                ),
                hovertext=self.df.apply(lambda row: f"{row['model']}: {row['average_ttlt']:.2f}s", axis=1),
                hoverinfo='text'
            )
        )
        
        fig.update_xaxes(tickangle=45)
        fig.update_layout(
            autosize=False,
            width=fig_width,
            height=fig_height
        )
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
            template="plotly_white"
        )

        # Add average TTL as a separate trace with lightning emoji markers
        fig.add_trace(
            go.Scatter(
                x=self.df["model"],
                y=self.df['average_ttlt'],
                mode='markers+text',
                name='⚡Avg TTL',
                marker=dict(
                    color='blue',
                    size=12,
                    symbol='circle'
                ),
                text=["⚡" for _ in self.df['average_ttlt']],
                textposition="top center",
                textfont=dict(
                    color='blue',
                    size=16
                ),
                hovertext=self.df.apply(lambda row: f"{row['model']}: {row['average_ttlt']:.2f}s", axis=1),
                hoverinfo='text'
            )
        )
        
        # Adjusted custom markers for median values with detailed hover information and human emoji
        fig.add_trace(
            go.Scatter(
                x=self.df["model"],
                y=self.df['median_prompt_tokens'],
                mode='markers+text',  # Combine markers and text for display
                name='Median Tokens 👤',  # Updated name with human emoji
                marker=dict(
                    color='lightblue',  # Use light blue color for a lighter appearance
                    size=10,  # Adjust size as needed
                    symbol='circle'  # Use 'circle' symbol for a softer look
                ),
                text=["👤" for _ in self.df['model']],  # Use human emoji for each point
                textposition="bottom center",  # Position text below the marker
                hoverinfo='text+name',  # Show custom text and name on hover
                hovertext= self.df.apply(lambda row: f"Model: {row['model']}\nMedian Prompt Tokens: {row['median_prompt_tokens']}", axis=1)  # Detailed hover text
            )
        )
        
        fig.update_xaxes(tickangle=45)
        fig.update_layout(
            autosize=False,
            width=fig_width,
            height=fig_height
        )
        return fig
    
    def plot_response_time_metrics_comparison(self):
        """
        Plot a comparison of response time metrics by model using Plotly, including generation tokens and input tokens with emojis.
        Adjusted to use a secondary y-axis for token metrics.
        """
        fig_width, fig_height = 1200, 600
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        models = self.df["model"].unique()

        # Existing metrics (Time-based)
        for metric in ["median_ttlt", "average_ttlt", "percentile_95_ttlt", "percentile_99_ttlt"]:
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=self.df.groupby("model")[metric].mean(),
                    name=metric
                ),
                secondary_y=False,  # Use the primary y-axis for time-based metrics
            )

        # Median prompt tokens with human emoji (Token-based)
        fig.add_trace(
            go.Scatter(
                x=self.df["model"],
                y=self.df['median_prompt_tokens'],
                mode='markers+text',
                name='Median Prompt Tokens 👤',
                marker=dict(color='lightblue', size=10, symbol='circle'),
                text=["👤" for _ in self.df['model']],
                textposition="bottom center",
                hoverinfo='text+name',
                hovertext=self.df.apply(lambda row: f"Model: {row['model']}\nMedian Prompt Tokens: {row['median_prompt_tokens']}", axis=1)
            ),
            secondary_y=True,  # Use the secondary y-axis for token metrics
        )

        # Median completion tokens with robot emoji (Token-based)
        fig.add_trace(
            go.Scatter(
                x=self.df["model"],
                y=self.df['median_completion_tokens'],
                mode='markers+text',
                name='Median Completion Tokens 🤖',
                marker=dict(color='lightblue', size=10, symbol='circle'),
                text=["🤖" for _ in self.df['model']],
                textposition="bottom center",
                hoverinfo='text+name',
                hovertext=self.df.apply(lambda row: f"Model: {row['model']}\nMedian Completion Tokens: {row['median_completion_tokens']}", axis=1)
            ),
            secondary_y=True,  # Use the secondary y-axis for token metrics
        )

        # Update layout to include a secondary y-axis
        fig.update_layout(
            barmode='group',
            title="Response Time Metrics by Model",
            xaxis_title="Model",
            yaxis_title="Time (seconds)",
            autosize=False,
            width=fig_width,
            height=fig_height,
            template="plotly_white",
            title_font_size=20
        )

        # Configure the secondary y-axis
        fig.update_yaxes(title_text="Token Count", secondary_y=True)

        return fig


    def plot_tokens(self):
        """
        Plot token statistics using Plotly and return the figure object.
        """
        token_features = ["median_prompt_tokens", "median_completion_tokens"]
        melted_df = self.df.melt(id_vars=["model"], value_vars=token_features)
        fig = px.bar(melted_df, x="model", y="value", color="variable", barmode="group",
                     labels={"value": "Tokens", "variable": "Token Type"},
                     title="Token Metrics by Model")
        fig.update_xaxes(tickangle=45)
        return fig

    def plot_errors(self):
        """
        Plot error rates using Plotly and return the figure object.
        """
        fig = px.bar(self.df, x="model", y="error_rate", title="Error Rate by Model")
        fig.update_traces(marker_color='indianred')
        fig.update_xaxes(tickangle=45)
        return fig

    def plot_best_worst_runs(self):
        """
        Compare the best and worst run times using Plotly and return the figure object.
        """
        if "best_run_time" in self.df.columns and "worst_run_time" in self.df.columns:
            melted_df = self.df.melt(id_vars=["model"], value_vars=["best_run_time", "worst_run_time"])
            fig = px.bar(melted_df, x="model", y="value", color="variable", barmode="group",
                         title="Best vs Worst Run Times by Model")
            fig.update_xaxes(tickangle=45)
        else:
            logger.debug("The columns 'best_run_time' and 'worst_run_time' are not present in the DataFrame")
            fig = None
        return fig

    def plot_heatmaps(self):
        """
        Plot a heatmap of performance metrics by region and model using Plotly and return the figure object.
        """
        if "regions" in self.df.columns:
            fig = px.imshow(
                self.df.pivot_table(index="model", columns="regions", values="median_time"),
                labels=dict(x="Region", y="Model", color="Median Time"),
                title="Performance Heatmap by Region and Model"
            )
        else:
            logger.debug("No regional data available for heatmap.")
            fig = None
        return fig

    def plot_time_vs_tokens(self):
        """
        Plot the relationship between time and tokens using Plotly and return the figure object.
        """
        fig = px.scatter(self.df, x="median_prompt_tokens", y="median_ttlt", color="model",
                         title="Response Time vs. Prompt Tokens by Model")
        return fig

    def visualize_all(self):
        """
        Visualize all the data by generating all plots using Plotly.
        """
        self.parse_data()

        fig1, fig2 = self.plot_times()
        fig_tokens = self.plot_tokens()
        fig_errors = self.plot_errors()
        fig_best_worst = self.plot_best_worst_runs()
        fig_heatmap = self.plot_heatmaps()
        fig_time_vs_tokens = self.plot_time_vs_tokens()

        fig1.show()
        fig2.show()
        fig_tokens.show()
        fig_errors.show()
        if fig_best_worst is not None:
            fig_best_worst.show()
        if fig_heatmap is not None:
            fig_heatmap.show()
        fig_time_vs_tokens.show()
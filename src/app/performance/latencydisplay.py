import json
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from my_utils.ml_logging import get_logger
logger = get_logger()

def create_latency_display_dataframe(stats: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Creates a DataFrame for displaying latency statistics from a list of statistics dictionaries.

    Parameters:
    - stats (List[Dict[str, Any]]): A list of dictionaries where each dictionary contains statistics for a particular model configuration.

    Returns:
    - pd.DataFrame: A DataFrame where each row represents the statistics for a particular model configuration, with columns for each statistic.
    """
    try: 
        headers = [
            "Model_MaxTokens",
            "is_Streaming",
            "Iterations",
            "Regions",
            "Average TTLT (s)",
            "Median TTLT (s)",
            "IQR TTLT",
            "95th Percentile TTLT (s)",
            "99th Percentile TTLT (s)",
            "CV TTLT",
            "Median Prompt Tokens",
            "IQR Prompt Tokens",
            "Median Completion Tokens",
            "IQR Completion Tokens",
            "95th Percentile Completion Tokens",
            "99th Percentile Completion Tokens",
            "CV Completion Tokens",
            "Average TBT (ms)",
            "Median TBT (ms)",
            "IQR TBT",
            "95th Percentile TBT (ms)",
            "99th Percentile TBT (ms)",
            "Average TTFT (ms/s)",
            "Median TTFT (ms/s)",
            "IQR TTFT",
            "95th Percentile TTFT (ms/s)",
            "99th Percentile TTFT (ms/s)",
            "Error Rate",
            "Error Types",
            "Successful Runs",
            "Unsuccessful Runs",
            "Throttle Count",
            "Throttle Rate",
            "Best Run",
            "Worst Run",
        ]

        table = [
            [
                key,
                data.get("is_Streaming", "N/A"),
                data.get("number_of_iterations", "N/A"),
                ", ".join(set([r for r in data.get("regions", []) if r])) or "N/A",
                data.get("average_ttlt", "N/A"),
                data.get("median_ttlt", "N/A"),
                data.get("iqr_ttlt", "N/A"),
                data.get("percentile_95_ttlt", "N/A"),
                data.get("percentile_99_ttlt", "N/A"),
                data.get("cv_ttlt", "N/A"),
                data.get("median_prompt_tokens", "N/A"),
                data.get("iqr_prompt_tokens", "N/A"),
                data.get("median_completion_tokens", "N/A"),
                data.get("iqr_completion_tokens", "N/A"),
                data.get("percentile_95_completion_tokens", "N/A"),
                data.get("percentile_99_completion_tokens", "N/A"),
                data.get("cv_completion_tokens", "N/A"),
                data.get("average_tbt", "N/A"),
                data.get("median_tbt", "N/A"),
                data.get("iqr_tbt", "N/A"),
                data.get("percentile_95_tbt", "N/A"),
                data.get("percentile_99_tbt", "N/A"),
                data.get("average_ttft", "N/A"),
                data.get("median_ttft", "N/A"),
                data.get("iqr_ttft", "N/A"),
                data.get("percentile_95_ttft", "N/A"),
                data.get("percentile_99_ttft", "N/A"),
                data.get("error_rate", "N/A"),
                data.get("errors_types", "N/A"),
                data.get("successful_runs", "N/A"),
                data.get("unsuccessful_runs", "N/A"),
                data.get("throttle_count", "N/A"),
                data.get("throttle_rate", "N/A"),
                json.dumps(data.get("best_run", {})) if data.get("best_run") else "N/A",
                json.dumps(data.get("worst_run", {})) if data.get("worst_run") else "N/A",
            ]
            for stat in stats
            for key, data in stat.items()
        ]

        df = pd.DataFrame(table, columns=headers)
        return df
    except Exception as e:
        logger.error(f"An error occurred, please retry the run. Error details: {e}")
    
    


def display_full_dataframe(df: pd.DataFrame) -> None:
    """
    Displays the full DataFrame with an expander for column descriptions and a styled DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to display.

    This function uses Streamlit to display the DataFrame and provides a detailed markdown
    description of each column within an expander. It aims to give a comprehensive overview
    of the data, including metrics related to model performance, errors, and throttling.
    """
    with st.expander("📊 Column Descriptions", expanded=False):
        st.markdown(
            """
            - **Model_MaxTokens**: Maximum tokens the model processes per request.
            - **is_Streaming**: Indicates if the model uses streaming. 
            - **Iterations**: Number of analysis iterations. 
            - **Regions**: Deployment geographic regions. 
            - **Average TTLT (s)**: Average time to last token. 
            - **Median TTLT (s)**: Median time to last token. 
            - **IQR TTLT**: Spread of the middle 50% TTLT data. 
            - **95th Percentile TTLT (s)**: 95% of TTLT measurements fall below this. 
            - **99th Percentile TTLT (s)**: 99% of TTLT measurements fall below this. 
            - **CV TTLT**: Relative variability of TTLT. 
            - **Median Prompt Tokens**: Median tokens in prompts. 
            - **IQR Prompt Tokens**: Spread of the middle 50% prompt tokens. 
            - **Median Completion Tokens**: Median tokens in completions. 
            - **IQR Completion Tokens**: Spread of the middle 50% completion tokens. 
            - **95th Percentile Completion Tokens**: 95% of completion tokens fall below this. 
            - **99th Percentile Completion Tokens**: 99% of completion tokens fall below this. 
            - **CV Completion Tokens**: Relative variability of completion tokens. 
            - **Average TTFT (ms/s)**: Average time to first token. 
            - **Median TTFT (ms/s)**: Median time to first token. 
            - **IQR TTFT**: Spread of the middle 50% TTFT data. 
            - **95th Percentile TTFT (ms/s)**: 95% of TTFT measurements fall below this. 
            - **99th Percentile TTFT (ms/s)**: 99% of TTFT measurements fall below this. 
            - **Error Rate**: Percentage of runs with errors. 
            - **Error Types**: Types of encountered errors. 
            - **Successful Runs**: Runs completed without errors. 
            - **Unsuccessful Runs**: Runs not completed due to errors. 
            - **Throttle Count**: Times throttling occurred. 
            - **Throttle Rate**: Rate of throttling occurrences. 
            - **Best Run**: Best performance metrics run. 
            - **Worst Run**: Worst performance metrics run. 
            """,
            unsafe_allow_html=True,
        )
    try:
        st.write(df.style)

    except Exception as e:
        logger.error(f"An error occurred, please retry the run. Error details: {e}")
        st.error(f"An error occurred, please retry the run. Error details: {e}")



def display_latency_metrics(df: pd.DataFrame) -> None:
    """
    Displays latency-related metrics from the DataFrame with an expander for column descriptions.

    Parameters:
    - df (pd.DataFrame): The DataFrame from which latency metrics are to be displayed.

    This function focuses on displaying latency metrics such as TTLT and TTFT, providing insights
    into the model's response times. It includes a markdown description for each metric within an expander.
    """
    latency_cols = [
        "Model_MaxTokens",
        "is_Streaming",
        "Iterations",
        "Regions",
        "Average TTLT (s)",
        "Median TTLT (s)",
        "IQR TTLT",
        "95th Percentile TTLT (s)",
        "99th Percentile TTLT (s)",
        "CV TTLT",
        "Average TTFT (ms/s)",
        "Median TTFT (ms/s)",
        "IQR TTFT",
        "95th Percentile TTFT (ms/s)",
        "99th Percentile TTFT (ms/s)",
    ]
    with st.expander("📊 Column Descriptions", expanded=False):
        st.markdown(
            """
            - **Model_MaxTokens**: Maximum tokens the model processes per request.
            - **is_Streaming**: Indicates if the model uses streaming. 
            - **Iterations**: Number of analysis iterations. 
            - **Regions**: Deployment geographic regions. 
            - **Average TTLT (s)**: Average time to last token. 
            - **Median TTLT (s)**: Median time to last token, reducing outliers' impact. 
            - **IQR TTLT**: Spread of the middle 50% TTLT data, indicating variability. 
            - **95th Percentile TTLT (s)**: 95% of TTLT measurements fall below this time. 
            - **99th Percentile TTLT (s)**: 99% of TTLT measurements fall below this time. 
            - **CV TTLT**: Coefficient of Variation for TTLT, showing relative variability. 
            - **Average TTFT (ms/s)**: Average time to first token, indicating initial response speed. 
            - **Median TTFT (ms/s)**: Median time to first token, reducing outliers' impact. 
            - **IQR TTFT**: Spread of the middle 50% TTFT data, indicating variability. 
            - **95th Percentile TTFT (ms/s)**: 95% of TTFT measurements fall below this time. 
            - **99th Percentile TTFT (ms/s)**: 99% of TTFT measurements fall below this time. 
            """,
            unsafe_allow_html=True,
        )
    try:
        st.write(df[latency_cols].style)

    except Exception as e:
        logger.error(f"An error occurred, please retry the run. Error details: {e}")
        st.error(f"😔 An error occurred, please retry the run. Error details: {e}")



def display_token_metrics(df: pd.DataFrame) -> None:
    """
    Displays token-related metrics from the DataFrame with an expander for column descriptions.

    Parameters:
    - df (pd.DataFrame): The DataFrame from which token metrics are to be displayed.

    This function focuses on displaying metrics related to prompt and completion tokens, providing insights
    into the model's token generation capabilities. It includes a markdown description for each metric within an expander.
    """
    token_cols = [
        "Model_MaxTokens",
        "is_Streaming",
        "Iterations",
        "Regions",
        "Median Prompt Tokens",
        "IQR Prompt Tokens",
        "Median Completion Tokens",
        "IQR Completion Tokens",
        "95th Percentile Completion Tokens",
        "99th Percentile Completion Tokens",
        "CV Completion Tokens",
    ]
    with st.expander("📊 Column Descriptions", expanded=False):
        st.markdown(
            """
            - **Model_MaxTokens**: Maximum tokens the model processes per request.
            - **is_Streaming**: Indicates if the model uses streaming. 
            - **Iterations**: Number of analysis iterations. 
            - **Regions**: Deployment geographic regions. 
            - **Median Prompt Tokens**: Median number of tokens in prompts. 
            - **IQR Prompt Tokens**: Spread of the middle 50% prompt tokens, indicating variability. 
            - **Median Completion Tokens**: Median number of tokens in completions. 
            - **IQR Completion Tokens**: Spread of the middle 50% completion tokens, indicating variability. 
            - **95th Percentile Completion Tokens**: 95% of completion token counts fall below this number. 
            - **99th Percentile Completion Tokens**: 99% of completion token counts fall below this number. 
            - **CV Completion Tokens**: Coefficient of Variation for completion tokens, showing relative variability. 
            """,
            unsafe_allow_html=True,
        )

    try:
        st.write(df[token_cols].style)

    except Exception as e:
        logger.error(f"An error occurred, please retry the run. Error details: {e}")
        st.error(f"😔 An error occurred, please retry the run. Error details: {e}")


def display_error_and_throttle_metrics(df):
    """
    Displays selected metrics related to errors and throttling in a Streamlit app.

    Parameters:
    - df (DataFrame): The DataFrame containing the metrics data.
    """
    # Define the columns related to errors and throttling to display
    error_and_throttle_cols = [
        "Model_MaxTokens",
        "is_Streaming",
        "Iterations",
        "Regions",
        "Error Rate",
        "Error Types",
        "Successful Runs",
        "Unsuccessful Runs",
        "Throttle Count",
        "Throttle Rate",
    ]

    # Use Streamlit's expander to show descriptions of the columns
    with st.expander("📊 Column Descriptions", expanded=False):
        st.markdown(
            """
            - **Model_MaxTokens**: Maximum tokens the model processes per request.
            - **is_Streaming**: Indicates if the model uses streaming. 
            - **Iterations**: Number of analysis iterations. 
            - **Regions**: Deployment geographic regions. 
            - **Error Rate**: The percentage of runs that resulted in errors.
            - **Error Types**: The types of errors encountered during the runs.
            - **Successful Runs**: The number of runs that completed successfully without errors.
            - **Unsuccessful Runs**: The number of runs that did not complete successfully due to errors.
            - **Throttle Count**: The number of times throttling occurred during the runs.
            - **Throttle Rate**: The rate at which throttling occurred.
            """
        )

    try:
        st.write(df[error_and_throttle_cols].style)

    except Exception as e:
        logger.error(f"An error occurred, please retry the run. Error details: {e}")
        st.error(f"😔 An error occurred, please retry the run. Error details: {e}")


def display_best_and_worst_run_analysis(df):
    """
    Displays an analysis of the best and worst runs in a Streamlit app.

    Parameters:
    - df (DataFrame): The DataFrame containing the runs data.
    """
    quality_data = []

    for index, row in df.iterrows():
        best_run = json.loads(row["Best Run"]) if row["Best Run"] != "N/A" else {}
        worst_run = json.loads(row["Worst Run"]) if row["Worst Run"] != "N/A" else {}

        if best_run and worst_run:
            quality_data.append(
                {
                    "Model_MaxTokens": row["Model_MaxTokens"],
                    "is_Streaming": row["is_Streaming"],
                    "Best TTLT (s)": best_run.get("ttlt", "N/A"),
                    "Best Completion Tokens": best_run.get("completion_tokens", "N/A"),
                    "Best Prompt Tokens": best_run.get("prompt_tokens", "N/A"),
                    "Best Region": best_run.get("region", "N/A"),
                    "Best Local Time": best_run.get("local_time", "N/A"),
                    "Worst TTLT (s)": worst_run.get("ttlt", "N/A"),
                    "Worst Completion Tokens": worst_run.get(
                        "completion_tokens", "N/A"
                    ),
                    "Worst Prompt Tokens": worst_run.get("prompt_tokens", "N/A"),
                    "Worst Region": worst_run.get("region", "N/A"),
                    "Worst Local Time": worst_run.get("local_time", "N/A"),
                }
            )

    try:

        quality_df = pd.DataFrame(quality_data)
    
    except Exception as e:
        logger.error(f"An error occurred, please retry the run. Error details: {e}")

    with st.expander("📊 Column Descriptions", expanded=False):
        st.markdown(
            """
            - **Model_MaxTokens**: The maximum number of tokens the model can process in a single request.
            - **Model_MaxTokens**: Maximum tokens the model processes per request.
            - **is_Streaming**: Indicates if the model uses streaming. 
            - **Best TTLT (s)**: Total Time to Last Token for the best run.
            - **Best Completion Tokens**: Number of completion tokens for the best run.
            - **Best Prompt Tokens**: Number of prompt tokens for the best run.
            - **Best Region**: Geographic region where the best run was executed.
            - **Best Local Time**: Local time when the best run was executed.
            - **Worst TTLT (s)**: Total Time to Last Token for the worst run.
            - **Worst Completion Tokens**: Number of completion tokens for the worst run.
            - **Worst Prompt Tokens**: Number of prompt tokens for the worst run.
            - **Worst Region**: Geographic region where the worst run was executed.
            - **Worst Local Time**: Local time when the worst run was executed.
            """
        )
    try:

        st.write(quality_df.style)
    
    except Exception as e:
        logger.error(f"An error occurred, please retry the run. Error details: {e}")
        st.error(f"😔 An error occurred, please retry the run. Error details: {e}")


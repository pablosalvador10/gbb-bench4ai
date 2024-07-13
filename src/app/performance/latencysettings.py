from typing import List

import pandas as pd
import streamlit as st


def configure_benchmark_settings() -> None:
    """
    Configure benchmark settings in the sidebar.
    """
    byop_option = st.radio(
        "BYOP (Bring Your Own Prompts)",
        options=["No", "Yes"],
        help="Select 'Yes' to bring your own prompt or 'No' to use default settings.",
    )

    if byop_option == "Yes":
        configure_byop_settings()
    else:
        configure_default_settings()

    configure_aoai_model_settings()


def configure_byop_settings() -> None:
    """
    Configure BYOP (Bring Your Own Prompts) settings.
    """
    # Ensure 'settings' exists in 'st.session_state'
    if "settings" not in st.session_state:
        st.session_state["settings"] = {}

    num_iterations = 0
    context_tokens = "BYOP"
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type="csv",
        help="Upload a CSV file with prompts for the benchmark tests.",
    )

    # Add 'context_tokens' to the 'settings' dictionary
    st.session_state["settings"]["context_tokens"] = context_tokens

    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        try:
            df = pd.read_csv(uploaded_file)
            if "prompts" in df.columns:
                prompts = df["prompts"].tolist()
                num_iterations = len(prompts)
                # Add 'prompts' and 'num_iterations' to the 'settings' dictionary
                st.session_state["settings"]["prompts"] = prompts
                st.session_state["settings"]["num_iterations"] = num_iterations
                custom_output_tokens = st.checkbox("Custom Output Tokens")
                if custom_output_tokens:
                    max_tokens_list = configure_custom_tokens()
                else:
                    max_tokens_list = configure_default_tokens()
                st.session_state["settings"]["max_tokens_list"] = max_tokens_list
            else:
                st.error("The uploaded CSV file must contain a 'prompts' column.")
        except Exception as e:
            st.error(f"An error occurred while processing the uploaded file: {e}")


def configure_default_settings() -> None:
    """
    Configure default benchmark settings.
    """
    # Ensure 'settings' exists in 'st.session_state'
    if "settings" not in st.session_state:
        st.session_state["settings"] = {}

    context_tokens = st.slider(
        "Context Tokens (Input)",
        min_value=100,
        max_value=5000,
        value=1000,
        help="Select the number of context tokens for each run.",
    )
    num_iterations = st.slider(
        "Number of Iterations",
        min_value=1,
        max_value=100,
        value=50,
        help="Select the number of iterations for each benchmark test.",
    )
    prompts = None

    custom_output_tokens = st.checkbox("Custom Output Tokens")
    if custom_output_tokens:
        max_tokens_list = configure_custom_tokens()
    else:
        max_tokens_list = configure_default_tokens()

    # Add inputs to the 'settings' dictionary
    st.session_state["settings"]["context_tokens"] = context_tokens
    st.session_state["settings"]["num_iterations"] = num_iterations
    st.session_state["settings"][
        "prompts"
    ] = prompts  # This will be None unless modified later
    st.session_state["settings"]["custom_output_tokens"] = custom_output_tokens
    st.session_state["settings"]["max_tokens_list"] = max_tokens_list


def configure_custom_tokens() -> List[int]:
    """
    Configure custom tokens for benchmark settings.

    :return: List of custom max tokens.
    """
    custom_tokens_input = st.text_input(
        "Type your own max tokens (separate multiple values with commas):",
        help="Enter custom max tokens for each run.",
    )
    if custom_tokens_input:
        try:
            return [int(token.strip()) for token in custom_tokens_input.split(",")]
        except ValueError:
            st.error("Please enter valid integers separated by commas for max tokens.")
            return []
    return []


def configure_default_tokens() -> List[int]:
    """
    Configure default tokens for benchmark settings.

    :return: List of default max tokens.
    """
    options = [100, 500, 800, 1000, 1500, 2000]
    default_tokens = st.multiselect(
        "Select Max Output Tokens (Generation)",
        options=options,
        default=[500],
        help="Select the maximum tokens for each run.",
    )
    st.session_state["settings"]["default_tokens"] = default_tokens
    return default_tokens


def configure_aoai_model_settings() -> dict:
    """
    Configure AOAI model settings and return the values from each input.

    :return: A dictionary containing the settings values.
    """
    with st.expander("AOAI Model Settings", expanded=False):
        # Ensure 'settings' exists in 'session_state'
        if "settings" not in st.session_state:
            st.session_state["settings"] = {}

        st.session_state["settings"]["temperature"] = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Adjust the temperature to control the randomness of the output. A higher temperature results in more random completions.",
        )
        prevent_server_caching = st.radio(
            "Prevent Server Caching",
            ("Yes", "No"),
            index=0,
            help="Choose 'Yes' to prevent server caching, ensuring that each request is processed freshly.",
        )
        st.session_state["settings"]["prevent_server_caching"] = (
            True if prevent_server_caching == "Yes" else False
        )

        st.session_state["settings"]["timeout"] = st.number_input(
            "Timeout (seconds)",
            min_value=1,
            max_value=300,
            value=60,
            help="Set the maximum time in seconds before the request times out.",
        )

        st.session_state["settings"]["top_p"] = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
            help="Adjust Top P to control the nucleus sampling, filtering out the least likely candidates.",
        )

        st.session_state["settings"]["presence_penalty"] = st.slider(
            "Presence Penalty",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="Adjust the presence penalty to discourage or encourage repeated content in completions.",
        )

        st.session_state["settings"]["frequency_penalty"] = st.slider(
            "Frequency Penalty",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="Adjust the frequency penalty to discourage or encourage frequent content in completions.",
        )

    return st.session_state["settings"]

import streamlit as st
import pandas as pd

def understanding_configuration():
    """
    Improved display of the benchmark configuration section in the UI.
    This function uses a multiselect widget for a clearer and more flexible selection process, allowing multiple choices.
    Collects all inputs in 'st.session_state.settings_quality' for later reference.
    """
    # Ensure 'settings_quality' exists in 'st.session_state'
    if "settings_quality" not in st.session_state:
        st.session_state["settings_quality"] = {}

    benchmark_selection = st.multiselect(
        "Choose Benchmark(s) for SLM/LLM assessment:",
        ["MMLU", "MedPub QA", "Truthful QA", "Custom Evaluation"],
        help="""Select one or more benchmarks to configure:
                - 'MMLU' for a diverse set of questions across multiple domains.
                - 'MedPub QA' to evaluate on medical publication questions.
                - 'Truthful QA' for assessing the model's ability to provide truthful answers.
                - 'Custom Evaluation' to run benchmarks on your own dataset."""
    )

    # Store benchmark selection in session state
    st.session_state["settings_quality"]["benchmark_selection"] = benchmark_selection

    if "MMLU" in benchmark_selection:
        configure_mmlu_benchmark()
    if "MedPub QA" in benchmark_selection:
        configure_medpub_benchmark()
    if "Truthful QA" in benchmark_selection:
        configure_truthful_benchmark()
    if "Custom Evaluation" in benchmark_selection:
        configure_custom_benchmark()


def configure_mmlu_benchmark():
    """
    Configure the MMLU benchmark settings.
    Collects MMLU settings in 'st.session_state.settings_quality' for later reference.
    """
    st.write("**MMLU Benchmark Settings**")
    mmlu_categories = st.multiselect(
        "Select MMLU subcategories to run",
        ["STEM", "Medical", "Business", "Social Sciences", "Humanities", "Other"],
        help="Select subcategories of the MMLU benchmark you'd like to run.",
    )

    mmlu_subsample = st.slider(
        "Select MMLU benchmark subsample for each selected category %. (14,402 total samples)",
        min_value=0,
        max_value=100,
    )

    # Store MMLU settings in session state
    st.session_state["settings_quality"]["mmlu_categories"] = mmlu_categories
    st.session_state["settings_quality"]["mmlu_subsample"] = mmlu_subsample


def configure_medpub_benchmark():
    """
    Configure the MedPub QA benchmark settings.
    Collects MedPub QA settings in 'st.session_state.settings_quality' for later reference.
    """
    st.write("**MedPub QA Benchmark Settings**")
    medpub_subsample = st.slider(
        "Select MedPub QA benchmark subsample %. (1,000 total samples)",
        min_value=0,
        max_value=100,
    )

    # Store MedPub QA settings in session state
    st.session_state["settings_quality"]["medpub_subsample"] = medpub_subsample


def configure_truthful_benchmark():
    """
    Configure the Truthful QA benchmark settings.
    Collects Truthful QA settings in 'st.session_state.settings_quality' for later reference.
    """
    st.write("**Truthful QA Benchmark Settings**")
    truthful_subsample = st.slider(
        "Select Truthful QA benchmark subsample %. (814 total samples)",
        min_value=0,
        max_value=100,
    )

    # Store Truthful QA settings in session state
    st.session_state["settings_quality"]["truthful_subsample"] = truthful_subsample


def configure_custom_benchmark():
    """
    Configure the settings for a custom benchmark.
    Collects custom benchmark settings in 'st.session_state.settings_quality' for later reference.
    """
    st.write("**Custom Benchmark Settings**")
    uploaded_file = st.file_uploader(
        "Upload CSV data",
        type=["csv"],
        help="Upload a CSV file with custom data for evaluation. CSV columns should be 'prompt', 'ground_truth', and 'context'. Context is optional",
    )

    if uploaded_file is not None:
        custom_df = pd.read_csv(uploaded_file)
        configure_custom_benchmark_columns(custom_df)


def configure_custom_benchmark_columns(custom_df):
    """
    Configure the column selections for a custom benchmark based on the uploaded CSV.
    Collects custom benchmark column settings in 'st.session_state.settings_quality' for later reference.
    """
    cols = custom_df.columns.tolist()
    cols.append("None")

    prompt_col = st.selectbox(
        label="Select 'prompt' column", options=cols, index=cols.index("None")
    )
    ground_truth_col = st.selectbox(
        label="Select 'ground_truth' column",
        options=cols,
        index=cols.index("None"),
    )
    context_col = st.selectbox(
        label="Select 'context' column (optional)",
        options=cols,
        index=cols.index("None"),
        help="Select the context column if available. Otherwise leave as 'None'",
    )

    st.session_state["settings_quality"]["custom_benchmark"] = {
        "prompt_col": prompt_col,
        "ground_truth_col": ground_truth_col,
        "context_col": context_col,
        "custom_df": custom_df
    }



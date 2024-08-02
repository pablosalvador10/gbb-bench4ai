import pandas as pd
import streamlit as st

'''
Helper functions to configure dataset options for Quality Page
All required variables are stored in the session state ['quality_settings']
'''

def __custom_dataset_options() -> None:
    # Upload custom dataset and configure columns
    st.write("### Custom Benchmark Settings")
    uploaded_file = st.file_uploader(
        "Upload CSV data",
        type=["csv"],
        key="custom_benchmark_df",
        help="Upload a CSV file with custom data for evaluation.",
    )
    if uploaded_file is not None:
        # To read file as df:
        custom_df = pd.read_csv(uploaded_file)
        cols = custom_df.columns.tolist()
        cols.append("None")

        prompt_col = st.selectbox(
            label="Select 'Prompt' column", options=cols, index=cols.index("None")
        )
        ground_truth_col = st.selectbox(
            label="Select 'Ground Truth' column",
            options=cols,
            index=cols.index("None"),
        )
        context_col = st.selectbox(
            label="Select 'Context' column (optional)",
            options=cols,
            index=cols.index("None"),
            help="Select the context column if available. Otherwise leave as 'None'",
        )

        if context_col != "None":
            custom_df.rename(
                columns={prompt_col: "prompt", ground_truth_col: "ground_truth", context_col: "context"},
                inplace=True,
            )
        else:
            custom_df.rename(
                columns={prompt_col: "prompt", ground_truth_col: "ground_truth"},
                inplace=True,
            )

        custom_subsample = st.slider(
            f"Select Custom benchmark subsample %. {custom_df.shape[0]} rows found",
            min_value=0,
            max_value=100,
        )

        st.session_state["quality_settings"]["custom_dataset"] = {
            "custom_df": custom_df,
            "custom_subsample": custom_subsample
        }

    return

def __mmlu_dataset_options():
    st.write("**MMLU Benchmark Settings**")

    # Sample to categories
    mmlu_categories = st.multiselect(
        "Select MMLU subcategories to run",
        ["STEM", "Medical", "Business", "Social Sciences", "Humanities", "Other"],
        help="Select subcategories of the MMLU benchmark you'd like to run.",
    )

    # Subsample
    mmlu_subsample = st.slider(
        "Select MMLU benchmark subsample for each selected category %. (14,402 total samples)",
        min_value=0,
        max_value=100,
    )

    st.session_state["quality_settings"]["mmlu"] = {
        "mmlu_categories": mmlu_categories,
        "mmlu_subsample": mmlu_subsample
    }
    return

def __medpub_dataset_options():
    st.write("**MedPub Benchmark Settings**")

    # Subsample
    medpub_subsample = st.slider(
        "Select MedPub benchmark subsample for each selected category %. (1,000 total samples)",
        min_value=0,
        max_value=100,
    )

    st.session_state["quality_settings"]["medpub"] = {
        "medpub_subsample": medpub_subsample
    }

    return

def __truthful_dataset_options():
    st.write("**TruthfulQA Benchmark Settings**")

    # Subsample
    truthful_subsample = st.slider(
        "Select TruthfulQA benchmark subsample %. (~800 total samples)",
        min_value=0,
        max_value=100,
    )

    st.session_state["quality_settings"]["truthful"] = {
        "truthful_subsample": truthful_subsample
    }

    return

def dataset_config(custom_select:bool = False, mmlu_select:bool = False, medpub_select:bool = False, truthful_select:bool = False, generic_select:bool = False) -> None:
    if custom_select:
        __custom_dataset_options()
    
    if mmlu_select:
        __mmlu_dataset_options()
    
    if medpub_select:
        __medpub_dataset_options()

    if truthful_select:
        __truthful_dataset_options()

    if generic_select:
        pass

    return
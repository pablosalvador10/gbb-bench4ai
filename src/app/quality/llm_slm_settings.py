import os
import streamlit as st
import pandas as pd
from datetime import datetime
from src.app.managers import create_eval_client
from my_utils.ml_logging import get_logger
import asyncio
from src.app.quality.runs import run_benchmark_quality
from typing import List, Dict, Any

# Set up logger
logger = get_logger()

def initialize_session_state(vars: List[str], initial_values: Dict[str, Any]) -> None:
    """
    Initialize Streamlit session state with default values if not already set.

    :param vars: List of session state variable names.
    :param initial_values: Dictionary of initial values for the session state variables.
    """
    for var in vars:
        if var not in st.session_state:
            st.session_state[var] = initial_values.get(var, None)

session_vars = [
    "settings_quality",
    "benchmark_selection_multiselect",
    "benchmark_selection",
    "activated_retrieval",
    "activated_rai",
    "activated_azureaistudio"
]
initial_values = {
    "settings_quality": {},
    "benchmark_selection_multiselect": [],
    "benchmark_selection": [],
    "activated_retrieval" : False,
    "activated_rai" : False,
    "activated_azureaistudio" : False,
}

initialize_session_state(session_vars, initial_values)

# Ensure "settings_quality" exists and is a dictionary
if "settings_quality" not in st.session_state:
    st.session_state["settings_quality"] = {}

# Ensure "benchmark_selection" exists within "settings_quality" and is a list
if "benchmark_selection" not in st.session_state["settings_quality"]:
    st.session_state["settings_quality"]["benchmark_selection"] = []

def configure_azure_ai_studio(session_key: str):
    """
    Configure Azure AI Studio connection details. This function provides a form for the user to input
    their Azure AI Studio credentials, which are then stored in Streamlit's session state.

    :param session_key: Key to identify the session state entry for this configuration.
    """
    if 'azure_ai_studio_subscription_id' not in st.session_state:
        st.session_state['azure_ai_studio_subscription_id'] = None
    if 'azure_ai_studio_resource_group_name' not in st.session_state:
        st.session_state['azure_ai_studio_resource_group_name'] = None
    if 'azure_ai_studio_project_name' not in st.session_state:
        st.session_state['azure_ai_studio_project_name'] = None

    with st.expander("🔍 Azure Remote Tracing (Azure AI Studio)"):
        # Only show the error message if the form is mandatory and the connection details are not yet provided
        if (st.session_state['azure_ai_studio_subscription_id'] is None or st.session_state['azure_ai_studio_resource_group_name'] is None or st.session_state['azure_ai_studio_project_name'] is None) and session_key == "rai":
            st.error("❌ Please complete the Azure AI Studio connection details to proceed.")
            st.warning('''Currently AI-assisted risk and safety metrics are only available in the following regions: East US 2, France Central, UK South,
                        Sweden Central. Groundedness measurement leveraging Azure AI Content Safety Groundedness Detection is only 
                        supported following regions: East US 2 and Sweden Central.''')

        with st.form(key=f"add_azure_ai_studio_{session_key}", border=True):
            subscription_id = st.text_input("Subscription ID:", value=st.session_state['azure_ai_studio_subscription_id'] or "")
            resource_group_name = st.text_input("Resource Group Name:", value=st.session_state['azure_ai_studio_resource_group_name'] or "")
            project_name = st.text_input("Project Name:", value=st.session_state['azure_ai_studio_project_name'] or "")

            button_text = "Update" if st.session_state['azure_ai_studio_subscription_id'] else "Connect"
            submitted = st.form_submit_button(button_text)

            if submitted:
                if subscription_id and resource_group_name and project_name:
                    st.session_state['azure_ai_studio_subscription_id'] = subscription_id
                    st.session_state['azure_ai_studio_resource_group_name'] = resource_group_name
                    st.session_state['azure_ai_studio_project_name'] = project_name
                    st.success("✅ Azure AI Studio connection details added/updated.")
                    st.session_state[f"activated_azureaistudio"] = True
                    st.experimental_rerun()
                else:
                    st.error("❌ All fields are required to connect to Azure AI Studio.")

def handle_deployment_selection(deployment_names, session_key, test):
    """
    Handle the selection of deployments and store the selected deployments in the session state.

    :param deployment_names: List of available deployment names.
    :param session_key: Key to store the selected deployments in the session state.
    :param test: The test associated with the deployments.
    """
    # Improved markdown message for better user guidance
    st.markdown('''Please select a deployment to begin your evaluation. We recommend choosing your most advanced model instance. 
                If you have added both GPT-3.5 and GPT-4o (Omni), we suggest opting for Omni for optimal results.''')    
    selected_deployment_names = st.radio(
            "Select a deployment",
            deployment_names,
            key=f"deployment_radio_{session_key}"
        )

    # Corrected and optimized version
    disable = True
    if st.session_state.get(f"activated_{test}", False):
        if test == "rai" and st.session_state.get("activated_azureaistudio", False):
            disable = False
        elif test != "rai":
            disable = False
    
    if disable:
        st.warning("⚠️ Please visit the 'Bring Your Own Prompts' (BYOP) section to configure your input data for the test.")
    
    if st.button("Add/Update Test", key=f"add_update_deployments_button_{session_key}", use_container_width=True, disabled=disable):
        st.session_state[session_key] = []

        for deployment_name in [selected_deployment_names]:
            selected_deployment = st.session_state.deployments.get(deployment_name, {})
            if all(key in selected_deployment for key in ["endpoint", "key", "version"]):
                try:
                    client = create_eval_client(
                        azure_endpoint=selected_deployment["endpoint"],
                        api_key=selected_deployment["key"],
                        azure_deployment=deployment_name,
                        api_version=selected_deployment["version"],
                        resource_group_name=st.session_state.get("azure_ai_studio_resource_group_name"),
                        project_name=st.session_state.get("azure_ai_studio_project_name"),
                        subscription_id=st.session_state.get("azure_ai_studio_subscription_id")
                    )
                    st.session_state[session_key].append(client)
                except Exception as e:
                    logger.error(f"Failed to create evaluation client for {deployment_name}. Error: {e}")
                    st.error(f"Failed to add {deployment_name} due to an error. Check logs for details.")
            else:
                missing_details = ", ".join([key for key in ["endpoint", "key", "version"] if key not in selected_deployment])
                logger.error(f"Missing required deployment details ({missing_details}) for {deployment_name}. Please check the deployment configuration.")
                st.error(f"Missing required deployment details ({missing_details}) for {deployment_name}. Please check the deployment configuration.")

        if st.session_state[session_key]:
            st.success("Selected deployments added for evaluation.")
            if test not in st.session_state["settings_quality"]["benchmark_selection"]:
                st.session_state["settings_quality"]["benchmark_selection"].append(test)
        else:
            st.warning("No deployments were added. Please select at least one deployment and ensure all required details are provided.")
       
    if st.button("Remove Test", key=f"remove_test_button_{session_key}", use_container_width=True):
        try:
            st.session_state["settings_quality"]["benchmark_selection"].remove(test)
            st.success(f"Test '{test}' removed successfully.")
        except ValueError:
            st.error(f"Test '{test}' not found in the selection.")

def generate_responses(test: str):
    existing_df_key = f"{test}_df"
    input_df_key = f"{test}_input_df"

    if existing_df_key in st.session_state["settings_quality"]:
        st.markdown("### Loading the DataFrame from Cache:")  
        st.dataframe(st.session_state["settings_quality"][existing_df_key].head(), hide_index=True)
        st.success("✅ The DataFrame is ready for use.")

    if st.session_state.get(f"regenerate_{test}", False) or existing_df_key not in st.session_state["settings_quality"]:
        try:
            st.markdown("### Generating Responses...")
            with st.spinner('Please wait...'):
                df = asyncio.run(run_benchmark_quality(
                    df=st.session_state["settings_quality"].get(input_df_key, pd.DataFrame()),
                    max_tokens=512
                ))

                if df.empty:
                    st.warning("No data returned. Using the default dataset.")
                    df = load_default_dataset()
                
                st.session_state["settings_quality"][existing_df_key] = df
                st.session_state[f"regenerate_{test}"] = False  # Reset the flag after regeneration
                st.success("✅ New DataFrame generated successfully.")
                st.dataframe(df.head(), hide_index=True)
                st.session_state[f"activated_{test}"] = True
                st.experimental_rerun()
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state[f"regenerate_{test}"] = False  # Ensure to reset the flag in case of error

def load_default_dataset():
    default_eval_path = os.path.join("my_utils", "data", "evaluations", "dataframe", "golden_eval_dataset.csv")
    return pd.read_csv(default_eval_path)

def handle_byop_upload(session_key: str):
    byop_option = st.radio(
        "Bring Your Own Prompts (BYOP)?", 
        ["No", "Yes"], 
        index=0, 
        key=f"byop_{session_key}", 
        help="Select 'Yes' to provide a DataFrame with 'Question', 'Context', and 'Ground Truth' columns for custom prompts. Select 'No' to use the default dataset."
    )
    if byop_option == "Yes":
        uploaded_file = st.file_uploader("Upload CSV", type="csv", key=session_key)
        if uploaded_file:
            process_uploaded_file(uploaded_file, session_key)
        else:
            st.warning("Please upload a CSV file.")
    else:
        use_default_dataset(session_key)

def process_uploaded_file(uploaded_file, session_key):
    df = pd.read_csv(uploaded_file)
    required_columns = ['question', 'context', 'ground_truth']
    if all(column in df.columns for column in required_columns):
        st.session_state["settings_quality"][f"{session_key}_input_df"] = df
        st.markdown("### Uploaded CSV:")
        st.dataframe(df.head(), hide_index=True)
        if st.button("Generate Responses", key=f"generate_{session_key}"):
            generate_responses(session_key)
    else:
        st.error("Uploaded CSV is missing required columns: 'question', 'context', 'ground_truth'.")

def use_default_dataset(session_key: str):
    df = load_default_dataset()
    st.session_state["settings_quality"][f"{session_key}_input_df"] = df
    st.markdown("### Default Dataset Loaded:")
    st.markdown(f"*Sneak peek of the first five rows:*")
    st.dataframe(df.head(), hide_index=True)
    if st.button("Generate Responses", key=f"generate_{session_key}"):
        generate_responses(session_key)

def configure_retrieval_settings() -> None:
    """
    Configure settings for Retrieval benchmarks, including deployment selection and BYOP (Bring Your Own Prompts) option.
    """
    st.markdown("""
    Please select your deployments first. Then, decide if you are bringing your own prompts 
    or using the default evaluation dataset.
    """)
    st.markdown("""
    Connecting to Azure AI Studio is optional for these section but recommended for enhanced evaluation tracking.
    """)
    configure_azure_ai_studio(session_key="retrieval")
    
    tabs_1_retrieval, tabs_2_retrieval = st.tabs(["🔍 Select Deployments", "➕ BYOP (Bring Your Own Prompts)"])

    with tabs_1_retrieval:
        if "deployments" in st.session_state and st.session_state.deployments:
            handle_deployment_selection(list(st.session_state.deployments.keys()), "evaluation_clients_retrieval", "retrieval")
        else:
            st.info("No deployments available. Please add a deployment in the Deployment Center and select them here later.")

    with tabs_2_retrieval:
        handle_byop_upload(session_key="retrieval")

def configure_rai_settings() -> None:
    """
    Configure settings for Responsible AI (RAI) benchmarks, including deployment selection and BYOP (Bring Your Own Prompts) option.
    """
    st.markdown("""
    Please select your deployments first. Then, decide if you are bringing your own prompts 
    or using the default evaluation dataset.
    """)
    #FIXME: ADD LOGIC TO DISABLE BUTTONS IF NO AZUREAI STUDIO IS SELECTED
    st.warning("⚠️ Completing the Azure AI Studio connection details is mandatory for this section.")
    configure_azure_ai_studio(session_key="rai")
    
    tabs_1_rai, tabs_2_rai = st.tabs(["🔍 Select Deployments", "➕ BYOP (Bring Your Own Prompts)"])

    with tabs_1_rai:
        if "deployments" in st.session_state and st.session_state.deployments:
            handle_deployment_selection(list(st.session_state.deployments.keys()), "evaluation_clients_rai", "rai")
        else:
            st.info("No deployments available. Please add a deployment in the Deployment Center and select them here later.")

    with tabs_2_rai:
        handle_byop_upload(session_key="rai")

def understanding_configuration():
    """
    Configure settings for Understanding benchmarks, including the selection of benchmarks like MMLU, MedPub QA, and Truthful QA.
    """
    # Initialize 'settings_quality' and 'benchmark_selection' in session_state if they don't exist
    if "settings_quality" not in st.session_state:
        st.session_state["settings_quality"] = {}
    if "benchmark_selection_multiselect" not in st.session_state["settings_quality"]:
        st.session_state["settings_quality"]["benchmark_selection_multiselect"] = []

    benchmark_selection = st.multiselect(
        "Choose Benchmark(s) for SLM/LLM assessment:",
        ["MMLU", "MedPub QA", "Truthful QA"],
        help="""Select one or more benchmarks to configure:
                - 'MMLU' for a diverse set of questions across multiple domains.
                - 'MedPub QA' to evaluate on medical publication questions.
                - 'Truthful QA' for assessing the model's ability to provide truthful answers."""
    )

    st.session_state["settings_quality"]["benchmark_selection_multiselect"] = benchmark_selection

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
    Configure the MMLU benchmark settings, including the selection of subcategories and subsample percentage.
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

    st.session_state["settings_quality"]["mmlu_categories"] = mmlu_categories
    st.session_state["settings_quality"]["mmlu_subsample"] = mmlu_subsample

def configure_medpub_benchmark():
    """
    Configure the MedPub QA benchmark settings, including the selection of subsample percentage.
    """
    st.write("**MedPub QA Benchmark Settings**")
    medpub_subsample = st.slider(
        "Select MedPub QA benchmark subsample %. (1,000 total samples)",
        min_value=0,
        max_value=100,
    )

    st.session_state["settings_quality"]["medpub_subsample"] = medpub_subsample

def configure_truthful_benchmark():
    """
    Configure the Truthful QA benchmark settings, including the selection of subsample percentage.
    """
    st.write("**Truthful QA Benchmark Settings**")
    truthful_subsample = st.slider(
        "Select Truthful QA benchmark subsample %. (814 total samples)",
        min_value=0,
        max_value=100,
    )

    st.session_state["settings_quality"]["truthful_subsample"] = truthful_subsample

def configure_custom_benchmark():
    """
    Configure settings for a custom benchmark by uploading a CSV file.
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

    :param custom_df: DataFrame containing the custom benchmark data.
    """
    cols = custom_df.columns.tolist()
    cols.append("None")

    prompt_col = st.selectbox("Select 'prompt' column", options=cols, index=cols.index("None"))
    ground_truth_col = st.selectbox("Select 'ground_truth' column", options=cols, index=cols.index("None"))
    context_col = st.selectbox("Select 'context' column (optional)", options=cols, index=cols.index("None"), help="Select the context column if available. Otherwise leave as 'None'")

    st.session_state["settings_quality"]["custom_benchmark"] = {
        "prompt_col": prompt_col,
        "ground_truth_col": ground_truth_col,
        "context_col": context_col,
        "custom_df": custom_df
    }

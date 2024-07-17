import os
import streamlit as st
import pandas as pd
from datetime import datetime
from src.app.managers import create_eval_client
from my_utils.ml_logging import get_logger

# Set up logger
logger = get_logger()

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
        if (st.session_state['azure_ai_studio_subscription_id'] is not None) and (st.session_state['azure_ai_studio_resource_group_name'] is not None) and (st.session_state['azure_ai_studio_project_name'] is not None):
            st.info("Azure AI Studio is already set up. You can update the connection details below.")

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
    st.markdown("Please choose a deployment to test:")
    
    selected_deployment_names = [
        name for index, name in enumerate(deployment_names) 
        if st.checkbox(name, key=f"deployment_checkbox_{name}_{session_key}_{index}")
    ]
    
    # Layout buttons in parallel with a visually appealing format
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Add/Update Test", key=f"add_update_deployments_button_{session_key}", use_container_width=True):
            st.session_state[session_key] = []

            for deployment_name in selected_deployment_names:
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

    with col2:
        if st.button("Remove Test", key=f"remove_test_button_{session_key}", use_container_width=True):
            try:
                st.session_state["settings_quality"]["benchmark_selection"].remove(test)
                st.success(f"Test '{test}' removed successfully.")
            except ValueError:
                st.error(f"Test '{test}' not found in the selection.")

def handle_byop_upload(session_key: str) -> None:
    """
    Handle the BYOP (Bring Your Own Prompts) file upload and store the uploaded data in the session state.

    :param session_key: Key to store the BYOP data in the session state.
    """
    byop_option = st.radio("BYOP (Bring Your Own Prompts)", options=["No", "Yes"], index=0, help="Select 'Yes' to bring your own prompts or 'No' to use the default settings.", key=f"byop_option_{session_key}")
    if byop_option == "Yes":
        uploaded_file = st.file_uploader("Upload CSV", key=session_key, type="csv", help="Upload a CSV file with prompts for the benchmark tests. The CSV must include the columns: 'question', 'answer', 'context', 'ground_truth'.")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                required_columns = ['question', 'answer', 'context', 'ground_truth']
                if not all(column in df.columns for column in required_columns):
                    st.warning("The uploaded CSV is missing one or more required columns: 'question', 'answer', 'context', 'ground_truth'. Please upload a CSV that includes these columns.")
                else:
                    st.session_state["settings_quality"][session_key] = df
                    st.dataframe(st.session_state["settings_quality"][session_key].head())
                    st.session_state["settings_quality"][f"{session_key}_BYOP"] = True
            except Exception as e:
                st.error(f"An issue occurred while uploading the CSV. Please try again. Error: {e}")
        else:
            st.warning("Please upload a CSV file to proceed or select 'No' to use the default dataset.")
    else:
        try:
            default_csv_path = os.path.join("my_utils", "data", "evaluations", "dataframe", "CompositeChat.csv")
            df = pd.read_csv(default_csv_path)
            st.session_state["settings_quality"][session_key] = df
            st.session_state["settings_quality"][f"{session_key}_BYOP"] = False
        except Exception as e:
            st.error(f"Failed to load the default dataset. Please ensure the file exists. Error: {e}")

def configure_retrieval_settings() -> None:
    """
    Configure settings for Retrieval benchmarks, including deployment selection and BYOP (Bring Your Own Prompts) option.
    """
    st.markdown("""
    Please select your deployments first. Then, decide if you are bringing your own prompts 
    or using the default evaluation dataset.
    """)
    st.markdown("""
    Connecting to Azure AI Studio is optional but recommended for enhanced evaluation tracking.
    """)
    configure_azure_ai_studio(session_key="retrieval")
    
    tabs_1_retrieval, tabs_2_retrieval = st.tabs(["🔍 Select Deployments", "➕ BYOP (Bring Your Own Prompts)"])

    with tabs_1_retrieval:
        if "deployments" in st.session_state and st.session_state.deployments:
            handle_deployment_selection(list(st.session_state.deployments.keys()), "evaluation_clients_retrieval", "Retrieval")
        else:
            st.info("No deployments available. Please add a deployment in the Deployment Center and select them here later.")

    with tabs_2_retrieval:
        handle_byop_upload("evaluation_clients_retrieval_df")

def configure_rai_settings() -> None:
    """
    Configure settings for Responsible AI (RAI) benchmarks, including deployment selection and BYOP (Bring Your Own Prompts) option.
    """
    st.markdown("""
    Please select your deployments first. Then, decide if you are bringing your own prompts 
    or using the default evaluation dataset.
    """)
    st.markdown("""
    Connecting to Azure AI Studio is optional but recommended for enhanced evaluation tracking.
    """)

    configure_azure_ai_studio(session_key="rai")
    
    tabs_1_rai, tabs_2_rai = st.tabs(["🔍 Select Deployments", "➕ BYOP (Bring Your Own Prompts)"])

    with tabs_1_rai:
        if "deployments" in st.session_state and st.session_state.deployments:
            handle_deployment_selection(list(st.session_state.deployments.keys()), "evaluation_clients_rai", "RAI")
        else:
            st.info("No deployments available. Please add a deployment in the Deployment Center and select them here later.")

    with tabs_2_rai:
        handle_byop_upload("evaluation_clients_rai_df")

def understanding_configuration():
    """
    Configure settings for Understanding benchmarks, including the selection of benchmarks like MMLU, MedPub QA, and Truthful QA.
    """
    # Initialize 'settings_quality' and 'benchmark_selection' in session_state if they don't exist
    if "settings_quality" not in st.session_state:
        st.session_state["settings_quality"] = {}
    if "benchmark_selection" not in st.session_state["settings_quality"]:
        st.session_state["settings_quality"]["benchmark_selection"] = []

    benchmark_selection = st.multiselect(
        "Choose Benchmark(s) for SLM/LLM assessment:",
        ["MMLU", "MedPub QA", "Truthful QA"],
        help="""Select one or more benchmarks to configure:
                - 'MMLU' for a diverse set of questions across multiple domains.
                - 'MedPub QA' to evaluate on medical publication questions.
                - 'Truthful QA' for assessing the model's ability to provide truthful answers."""
    )

    if benchmark_selection:
        for benchmark in benchmark_selection:
            if benchmark not in st.session_state["settings_quality"]["benchmark_selection"]:
                st.session_state["settings_quality"]["benchmark_selection"].append(benchmark)

        for benchmark in ["MMLU", "MedPub QA", "Truthful QA"]:
            if benchmark in st.session_state["settings_quality"]["benchmark_selection"] and benchmark not in benchmark_selection:
                st.session_state["settings_quality"]["benchmark_selection"].remove(benchmark)
    else:
        for benchmark in ["MMLU", "MedPub QA", "Truthful QA"]:
            if benchmark in st.session_state["settings_quality"]["benchmark_selection"]:
                st.session_state["settings_quality"]["benchmark_selection"].remove(benchmark)

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



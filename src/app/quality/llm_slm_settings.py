import streamlit as st
import pandas as pd
from datetime import datetime
from src.app.managers import create_eval_client
from utils.ml_logging import get_logger
import os

# Set up logger
logger = get_logger()


def configure_azure_ai_studio():
    """Configure Azure AI Studio connection details."""
    unique_form_key = "azure_ai_studio_form_" + datetime.now().strftime("%Y%m%d%H%M%S%f")
    with st.expander("🔍 Azure Remote Tracing (Azure AI Studio)"):
        with st.form(unique_form_key):
            # Input fields within the form
            subscription_id = st.text_input("Subscription ID:", key=f"sub_id_{unique_form_key}")
            resource_group_name = st.text_input("Resource Group Name:", key=f"rg_name_{unique_form_key}")
            project_name = st.text_input("Project Name:", key=f"proj_name_{unique_form_key}")
            
            # Form submission button
            submitted = st.form_submit_button("Submit")
            
            # Actions to take upon form submission
            if submitted:
                # Ensure all required fields are filled
                if subscription_id and resource_group_name and project_name:
                    st.session_state['azure_ai_studio_subscription_id'] = subscription_id
                    st.session_state['azure_ai_studio_resource_group_name'] = resource_group_name
                    st.session_state['azure_ai_studio_project_name'] = project_name
                    st.success("✅ Azure AI Studio connection details added.")
                else:
                    st.error("❌ All fields are required to connect to Azure AI Studio.")

def configure_retrieval_settings() -> None:
    st.markdown("""
    Please select your deployments first. Then, decide if you are bringing your own prompts 
    or using the default evaluation dataset.
    """)
    
    st.markdown("""
    Connecting to Azure AI Studio is optional but recommended for enhanced evaluation tracking.
    """)
        
    configure_azure_ai_studio()
    
    tabs_1_retrieval, tabs_2_retrieval = st.tabs(["🔍 Select Deployments", "➕ BYOP (Bring Your Own Prompts)"])

    with tabs_1_retrieval:
        if "deployments" in st.session_state and st.session_state.deployments:
            deployment_names = list(st.session_state.deployments.keys())
            selected_deployment_names = []
            for name in deployment_names:
                if st.checkbox(name, key=f"deployment_checkbox_{name}"):
                    selected_deployment_names.append(name)

            submitted_deployments_retrieval = st.button("Add/Update Deployments", use_container_width=False, key="add_update_deployments_button_2")
            if submitted_deployments_retrieval:
                st.session_state["evaluation_clients_retrievals"] = []
                
                for deployment_name in selected_deployment_names:
                    # Retrieve the deployment details from the session state
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
                            st.session_state["evaluation_clients_retrievals"].append(client)
                        except Exception as e:
                            logger.error(f"Failed to create evaluation client for {deployment_name}. Error: {e}")
                    else:
                        # Notify the user if any required deployment detail is missing
                        logger.error(f"Missing required deployment details for {deployment_name}. Please check the deployment configuration.")
                
                # Notify the user that the selected deployments have been successfully added for evaluation
                st.success("Selected deployments added for evaluation.")
                # Ensure the "benchmark_selection" key exists in the "settings_quality" dictionary and is a list
                if "benchmark_selection" not in st.session_state["settings_quality"] or not isinstance(st.session_state["settings_quality"]["benchmark_selection"], list):
                    st.session_state["settings_quality"]["benchmark_selection"]
                st.session_state["settings_quality"]["benchmark_selection"].append("retrieval")

        else:
            st.info("No deployments available. Please add a deployment in the Deployment Center and select them here later.")

    with tabs_2_retrieval:
        byop_option_retrieval = st.radio(
            "BYOP (Bring Your Own Prompts)",
            options=["No", "Yes"],
            index=0,
            help="Select 'Yes' to bring your own prompts or 'No' to use the default settings.",
            key="byop_option_retrieval"  # Unique key added here
        )
        if byop_option_retrieval == "Yes":
            uploaded_file = st.file_uploader(
                "Upload CSV",
                type="csv",
                help="Upload a CSV file with prompts for the benchmark tests. The CSV must include the columns: 'question', 'answer', 'context', 'ground_truth'."
            ) 
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    # Check if the required columns are present in the uploaded CSV
                    required_columns = ['question', 'answer', 'context', 'ground_truth']
                    if not all(column in df.columns for column in required_columns):
                        st.warning("The uploaded CSV is missing one or more required columns: 'question', 'answer', 'context', 'ground_truth'. Please upload a CSV that includes these columns.")
                    else:
                        st.session_state["evaluation_clients_retrieval_df"] = df
                        st.session_state["settings_quality"]["retrieval_BYOP"] = True
                except Exception as e:
                    st.error(f"An issue occurred while uploading the CSV. Please try again. Error: {e}")
            else: 
                st.warning("Please upload a CSV file to proceed or select 'No' to use the default dataset.")
        else:
            try:
                default_csv_path = os.path.join("utils", "data", "evaluations", "dataframe", "CompositeChat.csv")
                df = pd.read_csv(default_csv_path)
                st.session_state["evaluation_clients_retrieval_df"] = df
                st.session_state["settings_quality"]["retrieval_BYOP"] = False
            except Exception as e:
                st.error(f"Failed to load the default dataset. Please ensure the file exists. Error: {e}")

def configure_rai_settings() -> None:
    st.markdown("""
    Please select your deployments first. Then, decide if you are bringing your own prompts 
    or using the default evaluation dataset.
    """)
    
    st.markdown("""
    Connecting to Azure AI Studio is optional but recommended for enhanced evaluation tracking.
    """)
        
    configure_azure_ai_studio()
    
    tabs_1_rai, tabs_2_rai = st.tabs(["🔍 Select Deployments", "➕ BYOP (Bring Your Own Prompts)"])

    with tabs_1_rai:
        if "deployments" in st.session_state and st.session_state.deployments:
            deployment_names = list(st.session_state.deployments.keys())
            selected_deployment_names = []
            for name in deployment_names:
                if st.checkbox(name, key=f"deployment_{name}"):
                    selected_deployment_names.append(name)

            submitted_deployments_rai = st.button("Add/Update Deployments", use_container_width=True, key="add_update_deployments_button")
            if submitted_deployments_rai:
                st.session_state["evaluation_clients_rai"] = []
                
                for deployment_name in selected_deployment_names:
                    # Retrieve the deployment details from the session state
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
                            st.session_state["evaluation_clients_rai"].append(client)
                        except Exception as e:
                            logger.error(f"Failed to create evaluation client for {deployment_name}. Error: {e}")
                    else:
                        # Notify the user if any required deployment detail is missing
                        logger.error(f"Missing required deployment details for {deployment_name}. Please check the deployment configuration.")
                
                # Notify the user that the selected deployments have been successfully added for evaluation
                st.success("Selected deployments added for evaluation.")
                st.session_state["settings_quality"]["benchmark_selection"] = "rai"

        else:
            st.info("No deployments available. Please add a deployment in the Deployment Center and select them here later.")

    with tabs_2_rai:
        byop_option_rai = st.radio(
            "BYOP (Bring Your Own Prompts)",
            options=["No", "Yes"],
            index=0,
            help="Select 'Yes' to bring your own prompts or 'No' to use the default settings.",
            key="byop_option_rai"
        )
        if byop_option_rai == "Yes":
            uploaded_file = st.file_uploader(
                "Upload CSV",
                type="csv",
                help="Upload a CSV file with prompts for the benchmark tests. The CSV must include the columns: 'question', 'answer', 'context', 'ground_truth'."
            ) 
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    # Check if the required columns are present in the uploaded CSV
                    required_columns = ['question', 'answer', 'context', 'ground_truth']
                    if not all(column in df.columns for column in required_columns):
                        st.warning("The uploaded CSV is missing one or more required columns: 'question', 'answer', 'context', 'ground_truth'. Please upload a CSV that includes these columns.")
                    else:
                        st.session_state["evaluation_clients_rai_df"] = df
                        st.session_state["settings_quality"]["rai_BYOP"] = True
                except Exception as e:
                    st.error(f"An issue occurred while uploading the CSV. Please try again. Error: {e}")
            else: 
                st.warning("Please upload a CSV file to proceed or select 'No' to use the default dataset.")
        else:
            try:
                default_csv_path = os.path.join("utils", "data", "evaluations", "dataframe", "CompositeChat.csv")
                df = pd.read_csv(default_csv_path)
                st.session_state["evaluation_clients_rai_df"] = df
                st.session_state["settings_quality"]["rai_BYOP"] = False
            except Exception as e:
                st.error(f"Failed to load the default dataset. Please ensure the file exists. Error: {e}")


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
        ["MMLU", "MedPub QA", "Truthful QA"],
        help="""Select one or more benchmarks to configure:
                - 'MMLU' for a diverse set of questions across multiple domains.
                - 'MedPub QA' to evaluate on medical publication questions.
                - 'Truthful QA' for assessing the model's ability to provide truthful answers."""
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



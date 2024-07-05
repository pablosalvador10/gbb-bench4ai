# Home.py sets up the home page for a Streamlit app that allows users to 
# manage Azure OpenAI deployments, including adding, viewing, and updating deployment configurations

import base64
import os
from typing import Any, Dict, Optional

# Load environment variables
import dotenv
import streamlit as st

# Load environment variables if not already loaded
dotenv.load_dotenv(".env")

FROM_EMAIL = "Pablosalvadorlopez@outlook.com"


def get_image_base64(image_path: str) -> str:
    """
    Convert an image file to a base64 string.

    This function reads an image from the specified path and encodes it into a base64 string.

    :param image_path: Path to the image file.
    :return: Base64 encoded string of the image.
    :raises FileNotFoundError: If the image file is not found.
    :raises IOError: If there is an error reading the image file.
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def initialize_session_state(defaults: Dict[str, Any]) -> None:
    """
    Initialize Streamlit session state with default values if not already set.

    This function ensures that the Streamlit session state contains the specified default values if they are not already present.

    :param defaults: Dictionary of default values.
    """
    for var, value in defaults.items():
        if var not in st.session_state:
            st.session_state[var] = value


@st.cache_data
def get_main_content() -> str:
    """
    Get the main content HTML for the app.

    This function generates the main content HTML for the Streamlit app.

    :return: The main content HTML.
    """
    return f"""
    <h1 style="text-align:center;">
        Welcome to upgrade your RAG 🚀
        <br>
        <span style="font-style:italic; font-size:0.4em;"> with the RAG Benchmark Factory app </span> 
        <img src="data:image/png;base64,{get_image_base64('./utils/images/azure_logo.png')}" alt="RAG logo" style="width:25px;height:25px;">        
        <br>
    </h1>
    """


@st.cache_data
def get_markdown_content() -> str:
    """
    Get the markdown content for the app.

    This function generates the markdown content for the Streamlit app, providing information about the app's capabilities and resources.

    :return: The markdown content.
    """
    return """
    Our app zeroes in on key performance areas like speed, response time, and accuracy 🤖. It's a one-stop shop for testing Azure OpenAI models, helping you make smarter, cost-effective choices for your AI projects and boosting your capabilities by embracing the latest AI tech with solid, real-world data.

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; border-radius: 10px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.15); width: 80%; margin: auto;">
        <iframe src="https://www.loom.com/embed/9c6592b16c5b4785805ce87393601dfd?sid=bcc2e170-9295-427c-ae11-b89489f3ab6b" 
        frameborder="0" 
        webkitallowfullscreen 
        mozallowfullscreen 
        allowfullscreen 
        style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

    The platform boasts a user-friendly interface, streamlining the process of benchmarking various models for both latency and throughput. 
    Additionally, it facilitates a comprehensive quality evaluation, ensuring a thorough assessment of model performance.

    ### Curious to Learn More?
    - Discover the power of [Azure OpenAI](https://azure.microsoft.com/en-us/services/openai/) and how it's changing the world of AI
    - Join the conversation in our [community forums]() where experts and enthusiasts discuss the latest trends and challenges in AI.
    - Explore the **MMLU** benchmark, a massive multitask test covering 57 tasks across various knowledge domains. [Learn more](https://huggingface.co/datasets/cais/mmlu).
    - Investigate **Truthful QA**, a benchmark designed to assess the truthfulness of language models. [Discover](https://huggingface.co/datasets/truthfulqa/truthful_qa).
    - Delve into **PubMedQA**, a unique challenge for models to answer research questions based on abstracts. [Explore](https://huggingface.co/datasets/qiaojin/PubMedQA).
    
    #### Getting Started
    Make sure you've got your keys ready. Check the sidebar under 'Add Required Environment Variables' for all the details.

    - **Easy Navigation:** 
        - You're on the main page right now. 
        - 👈 The navigation tool at the top right corner is your best friend here. It's super easy to use and lets you jump to different parts of the app in no time.
        - 📊 **Performance Insights:** Dive deep into model performance, exploring throughput and latency.
        - 🔍 **Quality Metrics:** Evaluate the precision and reliability of your AI models.
    """


@st.cache_data
def get_footer_content() -> str:
    """
    Get the footer content HTML for the app.

    This function generates the footer content HTML for the Streamlit app.

    :return: The footer content HTML.
    """
    return """
    <div style="text-align:center; font-size:30px; margin-top:10px;">
        ...
    </div>
    <div style="text-align:center; margin-top:20px;">
        <a href="https://github.com/pablosalvador10/gbb-ai-upgrade-llm" target="_blank" style="text-decoration:none; margin: 0 10px;">
            <img src="https://img.icons8.com/fluent/48/000000/github.png" alt="GitHub" style="width:40px; height:40px;">
        </a>
        <a href="https://www.linkedin.com/in/pablosalvadorlopez/?locale=en_US" target="_blank" style="text-decoration:none; margin: 0 10px;">
            <img src="https://img.icons8.com/fluent/48/000000/linkedin.png" alt="LinkedIn" style="width:40px; height:40px;">
        </a>
        <!-- TODO: Update this link to the correct URL in the future -->
        <a href="#" target="_blank" style="text-decoration:none; margin: 0 10px;">
            <img src="https://img.icons8.com/?size=100&id=23438&format=png&color=000000" alt="Blog" style="width:40px; height:40px;">
        </a>
    </div>
    """


def load_default_deployment(
    name: Optional[str] = None,
    key: Optional[str] = None,
    endpoint: Optional[str] = None,
    version: Optional[str] = None,
) -> None:
    """
    Load default deployment settings, optionally from provided parameters.

    Ensures that a deployment with the same name does not already exist.

    :param name: (Optional) Name of the deployment.
    :param key: (Optional) Azure OpenAI key.
    :param endpoint: (Optional) API endpoint for Azure OpenAI.
    :param version: (Optional) API version for Azure OpenAI.
    :raises ValueError: If required deployment settings are missing.
    """
    # Ensure deployments is a dictionary
    if "deployments" not in st.session_state or not isinstance(
        st.session_state.deployments, dict
    ):
        st.session_state.deployments = {}

    # Check if the deployment name already exists
    deployment_name = (
        name if name else os.getenv("AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID")
    )
    if deployment_name in st.session_state.deployments:
        return  # Exit the function if deployment already exists

    default_deployment = {
        "name": deployment_name,
        "key": key if key else os.getenv("AZURE_OPENAI_KEY"),
        "endpoint": endpoint if endpoint else os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        "version": version if version else os.getenv("AZURE_OPENAI_API_VERSION"),
        "stream": False,
    }

    if all(
        value is not None for value in default_deployment.values() if value != False
    ):
        st.session_state.deployments[default_deployment["name"]] = default_deployment
    else:
        st.error("Default deployment settings are missing.")


def add_deployment_form() -> None:
    """
    Render the form to add a new Azure OpenAI deployment.

    This function provides a form in the Streamlit sidebar to add a new deployment, allowing users to specify deployment details.
    """
    with st.form("add_deployment_form"):
        deployment_name = st.text_input(
            "Deployment id",
            help="Enter the deployment ID for Azure OpenAI.",
            placeholder="e.g., chat-gpt-1234abcd",
        )
        deployment_key = st.text_input(
            "Azure OpenAI Key",
            help="Enter your Azure OpenAI key.",
            type="password",
            placeholder="e.g., sk-ab*****..",
        )
        deployment_endpoint = st.text_input(
            "API Endpoint",
            help="Enter the API endpoint for Azure OpenAI.",
            placeholder="e.g., https://api.openai.com/v1",
        )
        deployment_version = st.text_input(
            "API Version",
            help="Enter the API version for Azure OpenAI.",
            placeholder="e.g., 2024-02-15-preview",
        )
        is_streaming = st.radio(
            "Streaming",
            (True, False),
            index=1,
            format_func=lambda x: "Yes" if x else "No",
            help="Select 'Yes' if the model will be tested with output in streaming mode.",
        )
        submitted = st.form_submit_button("Add Deployment")

        if submitted:
            if (
                deployment_name
                and deployment_key
                and deployment_endpoint
                and deployment_version
            ):
                if deployment_name not in st.session_state.deployments:
                    st.session_state.deployments[deployment_name] = {
                        "key": deployment_key,
                        "endpoint": deployment_endpoint,
                        "version": deployment_version,
                        "stream": is_streaming,
                    }
                    st.success(f"Deployment '{deployment_name}' added successfully.")
                else:
                    st.error(
                        f"A deployment with the name '{deployment_name}' already exists."
                    )
            else:
                st.error("Please fill in all fields.")


def display_deployments() -> None:
    """
    Display and manage existing Azure OpenAI deployments.

    This function renders the existing deployments in the Streamlit sidebar, allowing users to view, update, or remove deployments.
    """
    if "deployments" in st.session_state:
        st.markdown("##### Loaded Deployments")
        for deployment_name, deployment in st.session_state.deployments.items():
            with st.expander(deployment_name):
                updated_name = st.text_input(
                    "Name", value=deployment_name, key=f"name_{deployment_name}"
                )
                updated_key = st.text_input(
                    "Key",
                    value=deployment.get("key", ""),
                    type="password",
                    key=f"key_{deployment_name}",
                )
                updated_endpoint = st.text_input(
                    "Endpoint",
                    value=deployment.get("endpoint", ""),
                    key=f"endpoint_{deployment_name}",
                )
                updated_version = st.text_input(
                    "Version",
                    value=deployment.get("version", ""),
                    key=f"version_{deployment_name}",
                )
                updated_stream = st.radio(
                    "Streaming",
                    (True, False),
                    format_func=lambda x: "Yes" if x else "No",
                    index=0 if deployment.get("stream", False) else 1,
                    key=f"stream_{deployment_name}",
                    help="Select 'Yes' if the model will be tested with output in streaming mode.",
                )

                if st.button("Update Deployment", key=f"update_{deployment_name}"):
                    st.session_state.deployments[deployment_name] = {
                        "key": updated_key,
                        "endpoint": updated_endpoint,
                        "version": updated_version,
                        "stream": updated_stream,
                    }
                    st.experimental_rerun()

                if st.button("Remove Deployment", key=f"remove_{deployment_name}"):
                    del st.session_state.deployments[deployment_name]
                    st.experimental_rerun()
    else:
        st.error("No deployments found. Please add a deployment in the sidebar.")


def main() -> None:
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(
        page_title="Home",
        page_icon="👋",
    )

    env_vars = {
        "AZURE_OPENAI_KEY": "",
        "AZURE_OPENAI_API_ENDPOINT": "",
        "AZURE_OPENAI_API_VERSION": "",
        "AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID": "",
        "deployments": {},
    }
    initialize_session_state(env_vars)

    st.toast(
        "Welcome to the RAG Benchmark Factory! Navigate through the 'Tool Help' guide or watch the video and start benchmarking your MaaS solutions!",
        icon="🚀",
    )

    with st.sidebar.expander("📘 Tool Help Guide", expanded=False):
        st.markdown(
            """
            Adding new deployments allows you to compare performance across multiple Azure OpenAI deployments in different regions. Here's a step-by-step guide on how to add and manage your deployments:

            ### Step 1: Enable Multi-Deployment
            - Check the **Add New Deployment** box at the top of the sidebar. This enables the option to add multiple deployments.

            ### Step 2: Add Deployment Details
            - Fill in the form with the following details:
                - **Azure OpenAI Key**: Your Azure OpenAI key. This is sensitive information, so it's treated as a password.
                - **API Endpoint**: The endpoint URL for Azure OpenAI.
                - **API Version**: The version of the Azure OpenAI API you're using.
                - **Chat Model Name Deployment ID**: The specific ID for your chat model deployment.
                - **Streaming**: Choose 'Yes' if you want the model to output in streaming mode.
            - Click **Add Deployment** to save your deployment to the session state.

            ### Step 3: View and Manage Deployments
            - Once added, your deployments are listed under **Loaded Deployments**.
            - Click on a deployment to expand and view its details.
            - You can update any of the deployment details here and click **Update Deployment** to save changes.

            ### How Deployments are Managed
            - Deployments are stored in the Streamlit `session_state`, allowing them to persist across page reloads and be accessible across different pages of the app.
            - You can add multiple deployments and manage them individually from the sidebar.
            - This flexibility allows you to easily compare the performance of different deployments and make adjustments as needed.

            ### Updating Deployments Across Pages
            - Since deployments are stored in the `session_state`, any updates made to a deployment from one page are reflected across the entire app.
            - This means you can seamlessly switch between different deployments or update their configurations without losing context.

            Follow these steps to efficiently manage your Azure OpenAI deployments and leverage the power of multi-deployment benchmarking.
            """,
            unsafe_allow_html=True,
        )

    with st.sidebar:
        st.markdown("## 🤖 Deployment Center ")

        load_default_deployment()

        with st.expander("Add Your MaaS Deployment", expanded=False):
            operation = st.selectbox(
                "Choose Model Family:",
                ("AOAI", "Other"),
                index=0,
                help="Select the benchmark you want to perform to evaluate AI model performance.",
                placeholder="Select a Benchmark",
            )
            if operation == "AOAI":
                add_deployment_form()
            else:
                st.info("Other deployment options will be available soon.")

        display_deployments()

        st.markdown("---")

        with st.expander("We Value Your Feedback! 🌟", expanded=False):
            st.markdown(
                """
                🐞 **Encountered a bug?** Or 🚀 have a **feature request**? We're all ears!

                Your feedback is crucial in helping us make our service better. If you've stumbled upon an issue or have an idea to enhance our platform, don't hesitate to let us know.

                📝 **Here's how you can help:**
                - Click on the link below to open a new issue on our GitHub repository.
                - Provide a detailed description of the bug or the feature you envision. The more details, the better!
                - Submit your issue. We'll review it as part of our ongoing effort to improve.

                [🔗 Open an Issue on GitHub](https://github.com/pablosalvador10/gbb-ai-upgrade-llm/issues)

                🙏 **Thank you for contributing!** Your insights are invaluable to us. Together, let's make our service the best it can be!
                """,
                unsafe_allow_html=True,
            )

    st.write(get_main_content(), unsafe_allow_html=True)
    st.markdown(get_markdown_content(), unsafe_allow_html=True)
    st.write(get_footer_content(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()

import base64
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import streamlit as st

from src.aoai.azure_openai import AzureOpenAIManager
from src.extractors.blob_data_extractor import AzureBlobDataExtractor
from src.app.utilsapp import send_email
from src.ocr.document_intelligence import AzureDocumentIntelligenceManager

FROM_EMAIL = "Pablosalvadorlopez@outlook.com"

# Function to convert image to base64
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

# Only set 'env_vars_loaded' to False if it hasn't been set to True
if not st.session_state.get("env_vars_loaded", False):
    st.session_state["env_vars_loaded"] = False
# Set 'env_vars_load_count_free' to 0 if it hasn't been set yet
if st.session_state.get("env_vars_load_count_free") is None:
    st.session_state["env_vars_load_count_free"] = 0

# Initialize environment variables in session state
env_vars = {
    "AZURE_OPENAI_KEY": "",
    "AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID": "",
    "AZURE_OPENAI_API_ENDPOINT": "",
    "AZURE_OPENAI_API_VERSION": "",
    "AZURE_BLOB_CONTAINER_NAME": "",
    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT": "",
    "AZURE_DOCUMENT_INTELLIGENCE_KEY": "",
    "AZURE_STORAGE_CONNECTION_STRING": "",
}

for var in env_vars:
    if var not in st.session_state:
        st.session_state[var] = env_vars[var]

# Add Feedback button
with st.sidebar.expander("We value your feedback! 😊", expanded=False):
    with st.form("feedback_form"):
        feedback_subject = st.text_input(
            "Subject:", value="", help="Enter the subject of your feedback."
        )
        feedback_text = st.text_area(
            "Please enter your feedback:",
            value="",
            help="Your feedback helps us improve our services.",
        )
        submitted = st.form_submit_button("Submit")
        if submitted:
            if feedback_subject and feedback_text:  # Check if both subject and feedback are provided
                to_emails = ["pablosal@microsoft.com"]
                subject = feedback_subject
                response = "Feedback: " + feedback_text

                with st.spinner("Sending feedback to the team..."):
                    send_email(
                        response=response,
                        from_email=FROM_EMAIL,
                        to_emails=[to_emails],  # Adjusted to match expected List[str] type
                        subject=subject
                    )

                st.success("Thank you for your feedback!")
            else:
                st.error("Please provide both a subject and feedback before submitting.")

with st.sidebar.expander("Add Required Environment Variables ⚙️", expanded=False):
    st.markdown(
        """
        Please provide the following Azure environment variables to configure the application. You can find these details in the respective Azure services.

        - **Azure OpenAI Key**: Get your key from [Azure OpenAI](https://azure.microsoft.com/en-us/services/openai/)
        - **Chat Model Deployment ID**: Your deployment ID for the chat model in Azure OpenAI.
        - **Completion Model Deployment ID**: Your deployment ID for the completion model in Azure OpenAI.
        - **Embedding Deployment ID**: Your deployment ID for the embedding model in Azure OpenAI.
        - **API Endpoint**: The API endpoint for your Azure OpenAI service.
        - **API Version**: The version of the Azure OpenAI API you are using.
        - **Document Intelligence Endpoint**: The endpoint for your Azure Document Intelligence API.
        - **Document Intelligence Key**: Your key for the Azure Document Intelligence API.
        - **Storage Connection String**: Your Azure Storage connection string.
        """
    )

    st.session_state["AZURE_OPENAI_KEY"] = st.text_input(
        "Azure OpenAI Key",
        value=st.session_state["AZURE_OPENAI_KEY"],
        help="Enter your Azure OpenAI key.",
        type="password",
        placeholder="Enter your Azure OpenAI Key",
        label_visibility="visible",
    )
    st.session_state["AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID"] = st.text_input(
        "Chat Model Deployment ID",
        value=st.session_state["AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID"],
        help="Enter your chat model deployment ID for Azure OpenAI.",
        placeholder="Enter your Chat Model Deployment ID",
        label_visibility="visible",
    )

    st.session_state["AZURE_OPENAI_API_ENDPOINT"] = st.text_input(
        "API Endpoint",
        value=st.session_state["AZURE_OPENAI_API_ENDPOINT"],
        help="Enter the API endpoint for Azure OpenAI.",
        placeholder="Enter your API Endpoint",
        label_visibility="visible",
    )
    st.session_state["AZURE_OPENAI_API_VERSION"] = st.text_input(
        "API Version",
        value=st.session_state["AZURE_OPENAI_API_VERSION"],
        help="Enter the API version for Azure OpenAI.",
        placeholder="Enter your API Version",
        label_visibility="visible",
    )
    st.session_state["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"] = st.text_input(
        "Document Intelligence Endpoint",
        value=st.session_state["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"],
        help="Enter the endpoint for Azure Document Intelligence API.",
        placeholder="Enter your Document Intelligence Endpoint",
        label_visibility="visible",
    )
    st.session_state["AZURE_DOCUMENT_INTELLIGENCE_KEY"] = st.text_input(
        "Document Intelligence Key",
        value=st.session_state["AZURE_DOCUMENT_INTELLIGENCE_KEY"],
        help="Enter your Azure Document Intelligence API key.",
        type="password",
        placeholder="Enter your Document Intelligence Key",
        label_visibility="visible",
    )
    st.session_state["AZURE_BLOB_CONTAINER_NAME"] = st.text_input(
        "Blob Container Name",
        value=st.session_state.get("CONTAINER_NAME", ""),
        help="Enter your Azure Blob Storage container name.",
        placeholder="Enter your Blob Container Name",
        label_visibility="visible",
    )
    st.session_state["AZURE_STORAGE_CONNECTION_STRING"] = st.text_input(
        "Storage Connection String",
        value=st.session_state.get("AZURE_STORAGE_CONNECTION_STRING", ""),
        help="Enter your Azure Storage connection string.",
        type="password",
        placeholder="Enter your Storage Connection String",
        label_visibility="visible",
    )

    if st.button("Validate Environment Variables"):
        try:
            # Initialize managers if they don't exist in session state
            for manager_name, manager in [
                (
                    "document_intelligence_manager",
                    AzureDocumentIntelligenceManager(
                        azure_endpoint=st.session_state[
                            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"
                        ],
                        azure_key=st.session_state["AZURE_DOCUMENT_INTELLIGENCE_KEY"],
                    ),
                ),
                (
                    "blob_data_extractor_manager",
                    AzureBlobDataExtractor(
                        connect_str=st.session_state["AZURE_STORAGE_CONNECTION_STRING"],
                        container_name=st.session_state["AZURE_BLOB_CONTAINER_NAME"],
                    ),
                ),
                (
                    "azure_openai_manager",
                    AzureOpenAIManager(
                        api_key=st.session_state["AZURE_OPENAI_KEY"],
                        azure_endpoint=st.session_state["AZURE_OPENAI_API_ENDPOINT"],
                        api_version=st.session_state["AZURE_OPENAI_API_VERSION"],
                        chat_model_name=st.session_state[
                            "AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID"
                        ],
                    ),
                ),
            ]:
                if manager_name not in st.session_state:
                    st.session_state[manager_name] = manager
            st.session_state["env_vars_loaded"] = True
            st.sidebar.success(
                "All environment variables and managers have been successfully validated and initialized."
            )
        except Exception as e:
            st.sidebar.error(
                f"An error occurred: {e}. Please check your environment variables."
            )

# Web user interface
st.write(
    f"""
    <h1 style="text-align:center;">
        Welcome to Azure AIFactory! 💡
        <br>
        <span style="font-style:italic; font-size:0.4em;"> Powered by <a href="https://pablosalvador10.github.io/GenAI-Engineering-Diary/" target="_blank">Microsoft AI Global Black Belt</a> </span> 
        <img src="data:image/png;base64,{get_image_base64('./utils/images/azure_logo.png')}" alt="logo" style="width:30px;height:30px;">        
        <br>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    Your one-stop solution for cultivating new ideas and possibilities with Azure. This is a factory of ideas, demonstrating what's possible and inspiring you to create your own AI applications. The sky is the limit with Azure!

    Our goal is to streamline the AI application creation process, reducing both the time and cost involved, while maintaining the highest quality standards. With Azure AIFactory, you can focus on what truly matters - delivering exceptional AI experiences.

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; border-radius: 10px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.15); width: 80%; margin: auto;">
        <iframe src="https://www.loom.com/embed/bd2b310ac6334a0a8da33b59986b2283?sid=f09a505b-e63e-400b-9460-ba039035d8ed" 
        frameborder="0" 
        webkitallowfullscreen 
        mozallowfullscreen 
        allowfullscreen 
        style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

    ### Curious to Learn More?
    - Discover the power of [Azure OpenAI](https://azure.microsoft.com/en-us/services/openai/) and how it's changing the world of AI.
    - Dive into our comprehensive [documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/openai-service/) to understand how our service works.
    - Join the conversation in our [community forums](https://discuss.streamlit.io) where experts and enthusiasts discuss the latest trends and challenges in AI.

    #### Getting Started
    To get the most out of Azure AIFactory, you'll need a few keys. Here's how you can obtain them:
    - **Azure OpenAI Key**: Sign up for Azure OpenAI and [get your key here](https://azure.microsoft.com/en-us/services/openai/).
    - **Azure Storage Connection String**: Follow the [Azure Storage documentation](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage) to set up your storage and obtain your connection string.
    - **Document Intelligence API**: Get your API key and endpoint by following the [Azure Document Intelligence documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/form-recognizer/).

    Embark on your journey with Azure AIFactory and transform the way you create AI applications!
    """,
    unsafe_allow_html=True,
)

st.write(
    """
    <div style="text-align:center; font-size:30px; margin-top:10px;">
        ...
    </div>
    <div style="text-align:center; margin-top:20px;">
        <a href="https://github.com/pablosalvador10" target="_blank" style="text-decoration:none; margin: 0 10px;">
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
    """,
    unsafe_allow_html=True,
)

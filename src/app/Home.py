import base64

import streamlit as st

from src.aoai.azure_openai import AzureOpenAIManager
from src.app.utilsapp import send_email
from src.performance.latencytest import (AzureOpenAIBenchmarkNonStreaming)

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

# Initialize environment variables in session state
env_vars = {
    "AZURE_OPENAI_KEY": "",
    "AZURE_OPENAI_API_ENDPOINT": "",
    "AZURE_OPENAI_API_VERSION": "",
    "AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID": "",
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
            if (
                feedback_subject and feedback_text
            ):  # Check if both subject and feedback are provided
                to_emails = ["pablosal@microsoft.com"]
                subject = feedback_subject
                response = "Feedback: " + feedback_text

                with st.spinner("Sending feedback to the team..."):
                    send_email(
                        response=response,
                        from_email=FROM_EMAIL,
                        to_emails=[
                            to_emails
                        ],  # Adjusted to match expected List[str] type
                        subject=subject,
                    )

                st.success("Thank you for your feedback!")
            else:
                st.error(
                    "Please provide both a subject and feedback before submitting."
                )

with st.sidebar.expander("Add Required Environment Variables ⚙️", expanded=False):
    st.markdown(
        """
        Please provide the following Azure environment variables to configure the application. You can find these details in the respective Azure services.

        - **Azure OpenAI Key**: Obtain your key from the [Azure OpenAI Service](https://azure.microsoft.com/en-us/services/openai/). This key is essential for authenticating your requests.
        - **Azure API Endpoint**: Find your specific API endpoint in the [Azure Portal](https://portal.azure.com/).
        - **Azure API Version**: Use the appropriate version of the Azure OpenAI API for your application. Refer to the [Azure OpenAI documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/) for details on different versions and their features.
        - **Azure OpenAI Chat Model Name Deployment ID**: This is the unique deployment ID for the chat model you intend to use, for accessing app capabilities and models. You will specify the models to test on subsequent pages. For more information on deployment IDs and setting up chat models, visit the [Azure OpenAI chat models documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/concept-chat).
        """
    )

    st.session_state["AZURE_OPENAI_KEY"] = st.text_input(
        "Azure OpenAI Key",
        value=st.session_state["AZURE_OPENAI_KEY"],
        help="Enter your Azure OpenAI key.",
        type="password",
        placeholder="e.g., sk-ab*****..",
        label_visibility="visible",
    )
    st.session_state["AZURE_OPENAI_API_ENDPOINT"] = st.text_input(
        "API Endpoint",
        value=st.session_state["AZURE_OPENAI_API_ENDPOINT"],
        help="Enter the API endpoint for Azure OpenAI.",
        placeholder="e.g., https://api.openai.com/v1",
        label_visibility="visible",
    )
    st.session_state["AZURE_OPENAI_API_VERSION"] = st.text_input(
        "API Version",
        value=st.session_state["AZURE_OPENAI_API_VERSION"],
        help="Enter the API version for Azure OpenAI.",
        placeholder="e.g., 2024-02-15-preview",
        label_visibility="visible",
    )
    st.session_state["AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID"] = st.text_input(
        "Chat Model Name Deployment ID",
        value=st.session_state["AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID"],
        help="Enter the chat model name deployment ID for Azure OpenAI.",
        placeholder="e.g., chat-gpt-1234abcd",
        label_visibility="visible",
    )

    if st.button("Validate Environment Variables"):
        try:
            # Initialize managers if they don't exist in session state
            managers_to_initialize = [
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
                # Create an instance of the benchmarking class
                (
                    "client_non_streaming",
                    AzureOpenAIBenchmarkNonStreaming(
                        api_key=st.session_state["AZURE_OPENAI_KEY"],
                        azure_endpoint=st.session_state["AZURE_OPENAI_API_ENDPOINT"],
                        api_version=st.session_state["AZURE_OPENAI_API_VERSION"],
                    ),
                ),
            ]

            for manager_name, manager in managers_to_initialize:
                if manager_name not in st.session_state:
                    st.session_state[manager_name] = manager

            st.session_state["env_vars_loaded"] = True
            st.sidebar.success(
                "Environment variables and managers initialized successfully."
            )
        except Exception as e:
            st.sidebar.error(
                f"Error initializing environment: {e}. Check your variables."
            )

st.write(
    f"""
    <h1 style="text-align:center;">
        Welcome to upgrade your LLM 🚀
        <br>
        <span style="font-style:italic; font-size:0.4em;"> Simplifying the adoption of the latest Azure OpenAI models  </span> 
        <img src="data:image/png;base64,{get_image_base64('./utils/images/azure_logo.png')}" alt="Azure OpenAI logo" style="width:25px;height:25px;">        
        <br>
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
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
    
    """,
    unsafe_allow_html=True,
)

st.write(
    """
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
    """,
    unsafe_allow_html=True,
)

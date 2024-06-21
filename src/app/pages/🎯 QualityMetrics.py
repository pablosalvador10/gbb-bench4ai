import asyncio
import json
import os

import dotenv
import streamlit as st

from src.aoai.azure_openai import AzureOpenAIManager
from src.app.outputformatting import markdown_to_docx
from src.performance.latencytest import (AzureOpenAIBenchmarkNonStreaming)
from utils.ml_logging import get_logger

FROM_EMAIL = "Pablosalvadorlopez@outlook.com"

# Load environment variables
dotenv.load_dotenv(".env")

# Set up logger
logger = get_logger()

# Initialize session state variables if they don't exist
session_vars = [
    "conversation_history",
    "ai_response",
    "chat_history",
    "messages",
    "log_messages",
    "benchmark_results",
]
initial_values = {
    "conversation_history": [],
    "ai_response": "",
    "chat_history": [],
    "messages": [
        {
            "role": "assistant",
            "content": "Hey, this is your AI assistant. Please look at the AI request submit and let's work together to make your content shine!",
        }
    ],
    "log_messages": [],
    "benchmark_results": [],
}
for var in session_vars:
    if var not in st.session_state:
        st.session_state[var] = initial_values.get(var, None)

st.set_page_config(
    page_title="Quality Metrics AI Assistant",
    page_icon="🎯",
)

# Check if environment variables have been loaded
if not st.session_state.get("env_vars_loaded", False):
    st.session_state.update(
        {
            "azure_openai_manager": None,
            "document_intelligence_manager": None,
            "blob_data_extractor_manager": None,
            "client_non_streaming": None,  # Initialize the key for the non-streaming client
        }
    )

    env_vars = {
        "AZURE_OPENAI_KEY": os.getenv("AZURE_OPENAI_KEY"),
        "AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID": os.getenv(
            "AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID"
        ),
        "AZURE_OPENAI_API_ENDPOINT": os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
    }

    st.session_state.update(env_vars)

    # Initialize the AzureOpenAIManager
    st.session_state["azure_openai_manager"] = AzureOpenAIManager(
        api_key=st.session_state["AZURE_OPENAI_KEY"],
        azure_endpoint=st.session_state["AZURE_OPENAI_API_ENDPOINT"],
        api_version=st.session_state["AZURE_OPENAI_API_VERSION"],
        chat_model_name=st.session_state["AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID"],
    )

    # Activate the AzureOpenAIBenchmarkNonStreaming manager
    st.session_state["client_non_streaming"] = AzureOpenAIBenchmarkNonStreaming(
        api_key=st.session_state["AZURE_OPENAI_KEY"],
        azure_endpoint=st.session_state["AZURE_OPENAI_API_ENDPOINT"],
        api_version=st.session_state["AZURE_OPENAI_API_VERSION"],
    )

# Main layout for initial submission #TODO
with st.expander("What Can I Do? 🤔", expanded=False):
    st.markdown(
        """
        Dive into the capabilities of our application:

        - **Multi-Region Latency Benchmark**: Test the response time of various models across different regions. This feature helps you identify the fastest model for your needs, ensuring efficient performance no matter where you are.
        - **Throughput Test by Model**: Evaluate how many requests a model can handle over a set period. This is crucial for understanding a model's capacity and ensuring it can handle your workload without slowing down.

        Our tool is designed to give you a comprehensive understanding of model performance, helping you make informed decisions. To begin, simply select an option from the sidebar. Let's optimize your AI model selection together! 👍
        """
    )

# Sidebar layout for initial submission # TODO
with st.sidebar:
    st.markdown("### 🚀 Let's Get Started!")
    operation = st.selectbox(
        "🎯 Choose Your Benchmark:",
        (
            "Latency Benchmark",
            "Throughput Benchmark by Model",
        ),
        help="Select the benchmark you want to perform to evaluate AI model performance.",
        placeholder="Select a Benchmark",
    )

    if operation == "Latency Benchmark":
        with st.expander("Benchmark Guide 📊", expanded=False):
            st.markdown(
                """
                Ready to test the performance and quality of your LLMs? Our benchmarking tool makes it easy! 📊✨

                Here's how it works:
                1. **Select your model settings**: Choose the models, maximum tokens, and the number of iterations for your benchmark tests.
                2. **Run the benchmark**: Hit the 'Run Benchmark' button and watch the real-time logs for progress updates.
                3. **Review the results**: Once the benchmark is complete, view detailed results and performance metrics.

                Let's get started and optimize your LLM experience! 🚀
                """
            )

        deployment_names = st.text_area(
            "Enter Deployment Names",
            help="Enter the deployment names you want to benchmark, one per line.",
        )

        max_tokens_list = st.multiselect(
            "Select Max Tokens",
            options=[100, 500, 700, 800],
            default=[100, 500, 700, 800],
            help="Select the maximum tokens for each run.",
        )

        num_iterations = st.slider(
            "Number of Iterations",
            min_value=1,
            max_value=100,
            value=50,
            help="Select the number of iterations for each benchmark test.",
        )

        context_tokens = st.slider(
            "Context Tokens",
            min_value=100,
            max_value=2000,
            value=1000,
            help="Select the number of context tokens for each run.",
        )

        temperature = st.slider(
            "Temperature",
            min_value=0,
            max_value=1,
            value=0,
            help="Select the temperature setting for the benchmark tests.",
        )

        multiregion = st.checkbox(
            "Enable Multi-region",
            value=False,
            help="Enable this option to run the benchmark tests across multiple regions.",
        )

        prevent_server_caching = st.checkbox(
            "Prevent Server Caching",
            value=True,
            help="Enable this option to prevent server caching during the benchmark tests.",
        )

        run_benchmark = st.button("Run Benchmark 🚀")


# Function to display statistics in a formatted manner # TODO
def display_statistics(stats):
    st.markdown("## Benchmark Results")
    st.table(stats)


# Define an asynchronous function to run benchmark tests and log progress # TODO
async def run_benchmark_tests():
    try:
        client_non_streaming = st.session_state["client_non_streaming"]

        deployment_names_list = [name.strip() for name in deployment_names.split(",")]

        await client_non_streaming.run_latency_benchmark_bulk(
            deployment_names=deployment_names_list,
            max_tokens_list=max_tokens_list,
            iterations=num_iterations,
            context_tokens=context_tokens,
            multiregion=multiregion,
            prevent_server_caching=prevent_server_caching,
        )

        stats = client_non_streaming.calculate_and_show_statistics()
        st.session_state["benchmark_results"] = stats

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


# Button to start the benchmark tests # TODO
if run_benchmark:
    with st.spinner("Running benchmark tests..."):
        asyncio.run(run_benchmark_tests())

    if st.session_state["benchmark_results"]:
        display_statistics(st.session_state["benchmark_results"])


def download_chat_history():
    chat_history_json = json.dumps(st.session_state.messages, indent=2)
    st.download_button(
        label="📜 Download Chat",
        data=chat_history_json,
        file_name="chat_history.json",
        mime="application/json",
        key="download-chat-history",
    )


def download_ai_response_as_docx_or_pdf():
    try:
        doc_io = markdown_to_docx(st.session_state.ai_response)
        file_format = st.selectbox("Select file format", ["DOCX", "PDF"])

        if file_format == "DOCX":
            st.download_button(
                label="📁 Download .docx",
                data=doc_io,
                file_name="AI_Generated_Guide.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="download-docx",
            )
        elif file_format == "PDF":
            st.download_button(
                label="📁 Download .pdf",
                data=doc_io,
                file_name="AI_Generated_Guide.pdf",
                mime="application/pdf",
                key="download-pdf",
            )
    except Exception as e:
        logger.error(f"Error generating {file_format} file: {e}")
        st.error(
            f"❌ Error generating {file_format} file. Please check the logs for more details."
        )


if st.session_state.ai_response:
    with st.sidebar:
        st.markdown("<hr/>", unsafe_allow_html=True)
        with st.expander("📥 Download Center", expanded=False):
            download_ai_response_as_docx_or_pdf()
            download_chat_history()

    # Enhanced Feedback and Contact Section
    st.sidebar.write(
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

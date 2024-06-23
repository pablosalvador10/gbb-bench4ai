import asyncio
import json
import os

import dotenv
import streamlit as st
import pandas as pd

from src.aoai.azure_openai import AzureOpenAIManager
from src.app.outputformatting import markdown_to_docx
from src.performance.latencytest import AzureOpenAIBenchmarkNonStreaming
from src.performance.aoaihelpers.stats import ModelPerformanceVisualizer
from utils.ml_logging import get_logger

# Load environment variables if not already loaded
dotenv.load_dotenv(".env")

# Set up logger
logger = get_logger()

st.set_option('deprecation.showPyplotGlobalUse', False)

# Initialize session state variables if they don't exist
session_vars = [
    "conversation_history",
    "ai_response",
    "chat_history",
    "messages",
    "log_messages",
    "benchmark_results",
    "deployments",
    "env_vars_loaded"
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
    "deployments": [],
    "env_vars_loaded": False
}
for var in session_vars:
    if var not in st.session_state:
        st.session_state[var] = initial_values.get(var, None)

st.set_page_config(
    page_title="Performance Insights AI Assistant",
    page_icon="📊",
)

# Load default deployment if multi-environment is not selected
def load_default_deployment():
    default_deployment = {
        "name": os.getenv("AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID"),
        "key": os.getenv("AZURE_OPENAI_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        "version": os.getenv("AZURE_OPENAI_API_VERSION"),
    }
    if all(default_deployment.values()):
        st.session_state.deployments = [default_deployment]
    else:
        st.error("Default deployment settings are missing in environment variables.")

# Function to create a new deployment manager
def create_azure_openai_manager(api_key, endpoint, api_version, deployment_id):
    return AzureOpenAIManager(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        chat_model_name=deployment_id,
    )

# Function to create a new benchmark client
def create_benchmark_client(api_key, endpoint, api_version):
    return AzureOpenAIBenchmarkNonStreaming(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )

# Sidebar logic
with st.sidebar:
    # Benchmark Selection
    st.markdown("## 🎯 Benchmark Selection")
    operation = st.selectbox(
        "Choose Your Benchmark:",
        ("Latency Benchmark", "Throughput Benchmark by Model"),
        help="Select the benchmark you want to perform to evaluate AI model performance.",
        placeholder="Select a Benchmark",
    )

    # Benchmark Guide
    if operation == "Latency Benchmark":
        with st.expander("Latency Benchmark Guide 📊", expanded=False):
            st.markdown(
                """
                Ready to test the performance and quality of your LLMs? Our benchmarking tool makes it easy!
    
                **Here's how it works:**
                1. **Select your model settings**: Choose the models, maximum tokens, and the number of iterations for your benchmark tests.
                2. **Run the benchmark**: Hit the 'Run Benchmark' button and watch the real-time logs for progress updates.
                3. **Review the results**: Once the benchmark is complete, view detailed results and performance metrics.
    
                Let's get started and optimize your LLM experience!
                """
            )

        st.markdown("---")  # Visual separator

        # Configuration Section
        st.markdown("## ⚙️ Configuration Settings")
        enable_multi_region = st.checkbox("Enable Multi Deployment")
        if not enable_multi_region:
            if not st.session_state.deployments:
                load_default_deployment()
        if enable_multi_region:
            st.markdown("### Add New Deployment")
            with st.form("add_deployment_form"):
                deployment_name = st.text_input("Deployment Name")
                deployment_key = st.text_input("API Key", type="password")
                deployment_endpoint = st.text_input("API Endpoint")
                deployment_version = st.text_input("API Version")
                submitted = st.form_submit_button("Add Deployment")

                if submitted:
                    if deployment_name and deployment_key and deployment_endpoint and deployment_version:
                        new_deployment = {
                            "name": deployment_name,
                            "key": deployment_key,
                            "endpoint": deployment_endpoint,
                            "version": deployment_version,
                        }
                        if 'deployments' not in st.session_state:
                            st.session_state.deployments = []
                        st.session_state.deployments.append(new_deployment)
                        st.success(f"Deployment '{deployment_name}' added successfully.")
                    else:
                        st.error("Please fill in all fields.")

        # Display the added deployments
        if 'deployments' in st.session_state:
            st.markdown("##### Current Deployments")
            for deployment in st.session_state.deployments:
                with st.expander(f"{deployment.get('name', 'Unnamed')}"):
                    # Make a copy of the deployment to avoid modifying the session state directly
                    deployment_copy = deployment.copy()
                    # Mask the 'key' field
                    if 'key' in deployment_copy:
                        deployment_copy['key'] = '*****'
                    # Display the modified deployment as JSON
                    st.json(deployment_copy)
        else: 
            st.error("No deployments found. Please add a deployment in the sidebar.")

        # Benchmark Configuration
        if operation == "Latency Benchmark":
            context_tokens = st.slider(
                "Context Tokens (Input)",
                min_value=100,
                max_value=5000,
                value=1000,
                help="Select the number of context tokens for each run.",
            )

        options = [100, 500, 800, 1000, 1500, 2000]
        max_tokens_list = st.multiselect(
            "Select Max Output Tokens (Generation)",
            options=options,
            default=[500],
            help="Select the maximum tokens for each run.",
        )

        custom_tokens = st.checkbox("Custom Output Tokens")
        if custom_tokens:
            custom_tokens_input = st.text_input("Type your own max tokens (separate multiple values with commas):")
            if custom_tokens_input:
                try:
                    custom_token_list = [int(token.strip()) for token in custom_tokens_input.split(',')]
                    max_tokens_list = list(set(max_tokens_list + custom_token_list))
                except ValueError:
                    st.error("Please enter valid integers separated by commas for max tokens.")

        num_iterations = st.slider(
            "Number of Iterations",
            min_value=1,
            max_value=100,
            value=50,
            help="Select the number of iterations for each benchmark test.",
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Select the temperature setting for the benchmark tests.",
        )

        prevent_server_caching = st.checkbox(
            "Prevent Server Caching",
            value=True,
            help="Enable this option to prevent server caching during the benchmark tests.",
        )

        st.markdown("---")  # Visual separator

    # Enhanced Run Benchmark Section
    st.markdown("## 🚀 Run Benchmark")
    st.markdown(
        """
        Ready to start the benchmark? Click the "Run Benchmark" button below. 
        You'll see real-time updates on the progress and be notified once the benchmark completes.
        """
    )
    
    run_benchmark = st.button("Run Benchmark")

# Function to display statistics in a formatted manner with updated key column name and row coloring
def display_statistics(stats):
    st.markdown("## Benchmark Results")
    table = []
    headers = [
        "Deployment_MaxTokens", "Iterations", "Regions", "Average Time", "Median Time",
        "IQR Time", "95th Percentile Time", "99th Percentile Time", "CV Time",
        "Median Prompt Tokens", "IQR Prompt Tokens", "Median Completion Tokens",
        "IQR Completion Tokens", "95th Percentile Completion Tokens",
        "99th Percentile Completion Tokens", "CV Completion Tokens", "Error Rate",
        "Error Types", "Successful Runs", "Unsuccessful Runs", "Throttle Count",
        "Throttle Rate", "Best Run", "Worst Run"
    ]

    # def format_run_info(run_data):
    #     if not run_data:
    #         return "N/A"
    #     # Example keys: 'time', 'tokens', 'error_rate'. Adjust based on actual data structure.
    #     formatted_info = f"Time: {run_data.get('time', 'N/A')}, Completion Tokens: {run_data.get('completion_tokens', 'N/A')}, Prompt Tokens: {run_data.get('prompt_tokens', 'N/A')}"
    #     return formatted_info

    for stat in stats: 
        for key, data in stat.items():
            regions = data.get("regions", [])
            regions = [r for r in regions if r is not None]  # Remove None values
            region_string = ", ".join(set(regions)) if regions else "N/A"
            # best_run_info = format_run_info(data.get("best_run"))
            # worst_run_info = format_run_info(data.get("worst_run"))
            row = [
                key,  # Changed to "Deployment MaxTokens"
                data.get("number_of_iterations", "N/A"),
                region_string,
                data.get("average_time", "N/A"),
                data.get("median_time", "N/A"),
                data.get("iqr_time", "N/A"),
                data.get("percentile_95_time", "N/A"),
                data.get("percentile_99_time", "N/A"),
                data.get("cv_time", "N/A"),
                data.get("median_prompt_tokens", "N/A"),
                data.get("iqr_prompt_tokens", "N/A"),
                data.get("median_completion_tokens", "N/A"),
                data.get("iqr_completion_tokens", "N/A"),
                data.get("percentile_95_completion_tokens", "N/A"),
                data.get("percentile_99_completion_tokens", "N/A"),
                data.get("cv_completion_tokens", "N/A"),
                data.get("error_rate", "N/A"),
                data.get("errors_types", "N/A"),
                data.get("successful_runs", "N/A"),
                data.get("unsuccessful_runs", "N/A"),
                data.get("throttle_count", "N/A"),
                data.get("throttle_rate", "N/A"),
                json.dumps(data.get("best_run", {})) if data.get("best_run") else "N/A",
                json.dumps(data.get("worst_run", {}))
            ]
            table.append(row)

    # Sort the table by the "Average Time" column (index 3) in ascending order
    table.sort(key=lambda x: x[3])

    # Convert the sorted table list into a DataFrame
    df = pd.DataFrame(table, columns=headers)
    
    # Color the rows based on the lowest value in the 'Median Time' column
    def color_lowest_median_time(s):
        is_lowest = s == s.min()
        return ['background-color: green' if v else '' for v in is_lowest]
    
    # Apply the coloring function to the DataFrame
    styled_df = df.style.apply(color_lowest_median_time, subset=['Median Time'])
    
    st.markdown("""
        ### Comprehensive Data Overview

        Below is the complete dataset showcasing all relevant metrics. This table includes detailed numerical data for each entry, providing insights into various performance indicators. Please review the table to understand the distribution and range of values across different metrics.
        """)
    # Display the styled DataFrame in Streamlit
    st.dataframe(styled_df)

    # Combine stats into a single dictionary for visualization
    combined_stats = {}
    for stat in stats:
        combined_stats.update(stat)
    
    st.markdown("""
        ### Visual Data Insights

        Following the table, you will find visualizations of the data. These visualizations are designed to provide a clear and intuitive understanding of the data, presented in a visually appealing format. We have arranged these visualizations in boxes, side by side, for a comparative and comprehensive view.
        """)
    # Assuming each plot method in ModelPerformanceVisualizer has been adjusted to return a plot object
    # Initialize the visualizer with the combined statistics data
    visualizer = ModelPerformanceVisualizer(data=combined_stats)
    visualizer.parse_data()  # Parse data before generating plots

    # Adjust the layout to create three columns for the visualizations using Streamlit's layout feature
    col1, col2, col3 = st.columns(3)

    # First column for Time Analysis
    with col1:
        st.markdown("##### Time Analysis")
        try:
            fig_time = visualizer.plot_times()  # Generate the Time Analysis plot
            st.pyplot(fig_time)
        except Exception as e:
            st.error(f"Error generating Time Analysis plot: {e}")

    # Second column for Token Analysis
    with col2:
        st.markdown("##### Token Analysis")
        try:
            fig_token = visualizer.plot_tokens()  # Generate the Token Analysis plot
            st.pyplot(fig_token)
        except Exception as e:
            st.error(f"Error generating Token Analysis plot: {e}")

    # Third column for Best and Worst Runs
    with col3:
        st.markdown("##### Best vs Worst")
        try:
            fig_best_worst = visualizer.plot_best_worst_runs()  # Generate the Best and Worst Runs plot
            st.pyplot(fig_best_worst)
        except Exception as e:
            st.error(f"Error generating Best and Worst Runs plot: {e}")
# Define an asynchronous function to run benchmark tests
async def run_benchmark_tests():
    try:
        deployment_clients = []
        for deployment in st.session_state.deployments:
            client = create_benchmark_client(
                api_key=deployment["key"],
                endpoint=deployment["endpoint"],
                api_version=deployment["version"],
            )
            deployment_clients.append(client)

        tasks = [
            client.run_latency_benchmark_bulk(
                deployment_names=[deployment["name"]],
                max_tokens_list=max_tokens_list,
                iterations=num_iterations,
                context_tokens=context_tokens,
                temperature=temperature,
                prevent_server_caching=prevent_server_caching,
            )
            for client in deployment_clients
        ]

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        stats = [
            client.calculate_and_show_statistics() for client in deployment_clients
        ]
        st.session_state["benchmark_results"] = stats

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Button to start the benchmark tests
if run_benchmark:
    if not st.session_state.deployments:
        st.error("No deployments found. Please add a deployment in the sidebar.")
    else: 
        deployment_names = [deployment["name"] for deployment in st.session_state.deployments]
        st.info(
            f"""
            ### Benchmark Configuration Summary
            - **Benchmark Type:** {operation}
            - **Max Tokens:** {max_tokens_list}
            - **Number of Iterations:** {num_iterations}
            - **Context Tokens:** {context_tokens}
            - **Temperature:** {temperature}
            - **Prevent Server Caching:** {prevent_server_caching}
            - **Deployments:** {', '.join(deployment_names)}
            """
        )
    
    with st.spinner("Running benchmark tests..."):
        asyncio.run(run_benchmark_tests())

    if st.session_state["benchmark_results"]:
        display_statistics(st.session_state["benchmark_results"])

# Download functions
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


import streamlit as st

from src.app.managers import create_azure_openai_manager


def configure_benchmarkbudyy_model_settings() -> dict:
    """
    Configure AOAI model settings and return the values from each input.

    :return: A dictionary containing the settings values.
    """
    with st.expander("BenchmarkBuddy Settings", expanded=False):
        # Ensure 'settings' exists in 'session_state'
        if "settings_buddy" not in st.session_state:
            st.session_state["settings_buddy"] = {}

        st.session_state["settings_buddy"]["temperature"] = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Think of 'Temperature' as the chatbot's creativity control. A higher value makes the chatbot more adventurous, giving you more varied and surprising responses.",
        )
        st.session_state["settings_buddy"]["top_p"] = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
            help="The 'Top P' setting helps fine-tune how the chatbot picks its words. Lowering this makes the chatbot's choices more predictable and focused, while a higher setting encourages diversity in responses.",
        )

        st.session_state["settings_buddy"]["max_tokens"] = st.slider(
            "Max Generation Tokens (Input)",
            min_value=100,
            max_value=3000,
            value=1000,
            help="This slider controls how much the chatbot talks. Slide to the right for longer stories or explanations, and to the left for shorter, more concise answers.",
        )

        st.session_state["settings_buddy"]["presence_penalty"] = st.slider(
            "Presence Penalty",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="Use 'Presence Penalty' to manage repetition. Increase it to make the chatbot avoid repeating itself, making each response fresh and unique.",
        )

        st.session_state["settings_buddy"]["frequency_penalty"] = st.slider(
            "Frequency Penalty",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="The 'Frequency Penalty' discourages the chatbot from using the same words too often. A higher penalty encourages a richer vocabulary in responses.",
        )

    return st.session_state["settings_buddy"]


def configure_chatbot() -> None:
    error_client_buddy = st.empty()
    if "azure_openai_manager" not in st.session_state:
        error_client_buddy.error(
            "Chatbot capabilities are currently disabled. To activate and fully utilize BenchmarkAI buddy knowledge, please configure the AOAI model."
        )

    with st.expander("Configure Buddy's Brain (AOAI)", expanded=False):
        st.write(
            "Add the AOAI-model to empower Buddy with advanced cognitive capabilities."
        )
        with st.form("add_deployment_chatbot"):
            st.session_state[
                "AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID_CHATBOT"
            ] = st.text_input(
                "Deployment id",
                help="Enter the deployment ID for Azure OpenAI.",
                placeholder="e.g., chat-gpt-1234abcd",
            )
            st.session_state["AZURE_OPENAI_KEY_CHATBOT"] = st.text_input(
                "Azure OpenAI Key",
                help="Enter your Azure OpenAI key.",
                type="password",
                placeholder="e.g., sk-ab*****..",
            )
            st.session_state["AZURE_OPENAI_API_ENDPOINT_CHATBOT"] = st.text_input(
                "API Endpoint",
                help="Enter the API endpoint for Azure OpenAI.",
                placeholder="e.g., https://api.openai.com/v1",
            )
            st.session_state["AZURE_OPENAI_API_VERSION_CHATBOT"] = st.text_input(
                "API Version",
                help="Enter the API version for Azure OpenAI.",
                placeholder="e.g., 2024-02-15-preview",
            )
            submitted_buddy = st.form_submit_button("Add Deployment")

    if submitted_buddy:
        try:
            st.session_state["azure_openai_manager"] = create_azure_openai_manager(
                api_key=st.session_state["AZURE_OPENAI_KEY_CHATBOT"],
                azure_endpoint=st.session_state["AZURE_OPENAI_API_ENDPOINT_CHATBOT"],
                api_version=st.session_state["AZURE_OPENAI_API_VERSION_CHATBOT"],
                deployment_id=st.session_state[
                    "AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID_CHATBOT"
                ],
            )
            stream = st.session_state.azure_openai_manager.openai_client.chat.completions.create(
                model=st.session_state.azure_openai_manager.chat_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Test: Verify setup.",
                    }
                ]
                + [{"role": "user", "content": "test"}],
                max_tokens=2,
                seed=555,
                stream=True,
            )
            st.toast("BenchmarkAI Buddy 🤖 successfully configured.")
            error_client_buddy.empty()
            st.session_state["disable_chatbot"] = False
        except Exception as e:
            st.warning(
                f"An issue occurred while initializing the Azure OpenAI manager. {e} Please try again. If the issue persists, verify your configuration."
            )

    configure_benchmarkbudyy_model_settings()

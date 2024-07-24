import streamlit as st 

def display_resources():
    """
    This function uses Streamlit to create a multi-expander interface for displaying various resources and information.
    Each expander contains detailed instructions or information on a specific topic related to benchmarking LLMs, adding new deployments,
    understanding the motivation behind the tool, learning about public benchmarks, metrics for custom evaluation, and additional
    capabilities of the application.

    Expanders:
    - Learn More About Public Benchmarks: Provides information on various public benchmarks for evaluating LLMs.
    - Learn More About Metrics for Custom Eval: Describes the metrics used for custom evaluations.
    """
    # Learn More About Public Benchmarks
    with st.expander("Learn More about LLM Benchmarks"):
        st.markdown(
            """
            - **MMLU (Massive Multitask Language Understanding)**
                - A comprehensive test covering 57 tasks across various domains: STEM, Medical, Business, Social Sciences, Humanities, and more.
                - Aims to evaluate extensive world knowledge and problem-solving abilities.
                - Resources: [Paper](https://arxiv.org/pdf/2009.03300), [HuggingFace Dataset](https://huggingface.co/datasets/cais/mmlu)

            - **Truthful QA**
                - Benchmarks language models on their ability to generate truthful answers across 38 categories, including health, law, finance, and politics.
                - Contains 817 questions designed to challenge common misconceptions.
                - Resources: [Paper](https://arxiv.org/pdf/2109.07958), [HuggingFace Dataset](https://huggingface.co/datasets/truthfulqa/truthful_qa), [GitHub](https://github.com/sylinrl/TruthfulQA)

            - **PubMedQA**
                - Focuses on answering yes/no/maybe research questions using abstracts from the PubMed database.
                - Features 1k expert-labeled instances for evaluation.
                - Resources: [Paper](https://arxiv.org/pdf/1909.06146), [HuggingFace Dataset](https://huggingface.co/datasets/qiaojin/PubMedQA), [Website](https://pubmedqa.github.io/), [GitHub](https://github.com/pubmedqa/pubmedqa)
            """
        )

    # Learn More About Metrics for Custom Eval
    with st.expander("Learn More About Systems Benchmarks"):
        st.markdown("""
            - **Accuracy**
                - Defined as the ratio of correct predictions to total predictions.
                - Requires exact matches between model outputs and ground truth.

            - **Answer Similarity**
                - Measures how closely the generated answer matches the ground truth.
                - Utilizes Sentence Transformers for computing sentence embeddings and cosine similarity.

            - **Context Similarity**
                - Assesses the relevance of the generated answer to the provided context.
                - Also uses Sentence Transformers for sentence embeddings and similarity calculation.
            """
        )
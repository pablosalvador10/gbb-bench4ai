import streamlit as st
from src.quality.evals import MMLU, CustomEval, PubMedQA, TruthfulQA
import pandas as pd
import asyncio
import plotly.express as px

from utils.ml_logging import get_logger

# Set up logger
logger = get_logger()

def display_results(results_c: st.container):
    # Access the results from session state
    if "results_quality" in st.session_state:
        results_df = st.session_state["results_quality"]
        
        results_c.markdown("## Benchmark Results")
        
        # Check if results_df is not empty
        if not results_df.empty:
            fig = px.bar(
                results_df,
                x="overall_score",
                y="test",
                color="deployment",
                barmode="group",
                orientation="h",
                title="Benchmark Results Overview"
            )
            fig.update_layout(
                xaxis_title="Overall Score",
                yaxis_title="Test",
                legend_title="Deployment",
                barmode='group'
            )
            results_c.plotly_chart(fig, use_container_width=True)
            top_bar = st.empty()  # Assuming top_bar is defined elsewhere as a placeholder
            top_bar.success("Benchmark tests completed successfully! 🎉")
        else:
            results_c.warning("No results to display.")
    else:
        st.error("No results found in session state.")
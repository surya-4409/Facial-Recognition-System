import streamlit as st
import json

st.title("Facial Verification Fairness Audit")
st.write("Live dashboard for demographic bias measurement and mitigation.")

# Load and display the initial audit results
st.header("Initial Audit Results")
try:
    with open("results/initial_audit.json", "r") as f:
        initial_audit = json.load(f)
    st.json(initial_audit)
except FileNotFoundError:
    st.error("Audit results not found. Please run the pipeline first.")

# Load and display the overall metrics
st.header("Overall Metrics")
try:
    with open("results/overall_metrics.json", "r") as f:
        metrics = json.load(f)
    st.json(metrics)
except FileNotFoundError:
    st.error("Metrics not found.")
import streamlit as st
import json
import pandas as pd

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Facial Verification Fairness Audit",
    page_icon="⚖️",
    layout="wide"
)

# ──────────────────────────────────────────────
# Sidebar: Author Information
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100) # Placeholder avatar
    st.header("Author Details")
    st.markdown("**Name:** BILLAKURTI VENKATA SURYANARAYANA")
    st.markdown("**Roll Number:23MH1A4409") 
    st.markdown("---")
    st.markdown("**Project Domain:** Responsible AI & Computer Vision")
    st.markdown("**Tech Stack:** PyTorch, OpenCV, Streamlit, Docker")

# ──────────────────────────────────────────────
# Main Header
# ──────────────────────────────────────────────
st.title("⚖️ Facial Verification Fairness Audit")
st.markdown("*Measuring and Mitigating Demographic Bias in Biometric Systems*")
st.markdown("---")

# ──────────────────────────────────────────────
# Create Tabs for clean organization
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📖 Project Context", "📊 Audit Results", "🎯 Final Outcome"])

# ==========================================
# TAB 1: Project Context
# ==========================================
with tab1:
    st.header("The Problem")
    st.write("""
    Facial recognition technology is increasingly utilized in high-stakes domains, from airport security to law enforcement. 
    However, numerous studies have revealed significant performance disparities across demographic groups. Models often perform 
    less accurately for women, people of color, and older individuals, leading to harmful outcomes like false arrests or unequal access to services.
    
    This project audits a facial verification system to quantify its biases and apply algorithmic fairness techniques to mitigate them, 
    aligning with emerging regulations like the EU AI Act.
    """)
    
    st.header("Methodology")
    st.write("""
    1. **Model Architecture:** Fine-tuned an `InceptionResnetV1` (VGGFace2) deep learning backbone to generate face embeddings.
    2. **Evaluation Metrics:** Calculated False Accept Rate (FAR) and False Reject Rate (FRR) across disaggregated demographic groups (Age, Gender, Skin Tone).
    3. **Bias Mitigation:** Implemented a **post-processing threshold calibration** strategy. By calculating the Equal Error Rate (EER) and establishing demographic-specific similarity thresholds, the system attempts to balance FRR disparities while enforcing a strict FAR tolerance.
    """)

# ==========================================
# TAB 2: Audit Results
# ==========================================
with tab2:
    st.header("Performance Trade-offs")
    
    # Load Overall Metrics
    try:
        with open("results/overall_metrics.json", "r") as f:
            metrics = json.load(f)
            
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Initial Model Accuracy", 
                value=f"{metrics.get('initial_model', {}).get('accuracy', 0):.4f}"
            )
        with col2:
            st.metric(
                label="Mitigated Model Accuracy", 
                value=f"{metrics.get('mitigated_model', {}).get('accuracy', 0):.4f}",
                delta="Post-Calibration Shift",
                delta_color="off"
            )
    except FileNotFoundError:
        st.warning("Overall metrics data not found.")

    st.markdown("---")
    
    # Load Disaggregated Audits
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Initial Disparities")
        try:
            with open("results/initial_audit.json", "r") as f:
                initial_audit = json.load(f)
            st.dataframe(pd.DataFrame.from_dict(initial_audit, orient='index'), use_container_width=True)
        except FileNotFoundError:
            st.error("Initial audit results not found.")

    with col4:
        st.subheader("Mitigated Disparities")
        try:
            with open("results/mitigated_audit.json", "r") as f:
                mitigated_audit = json.load(f)
            st.dataframe(pd.DataFrame.from_dict(mitigated_audit, orient='index'), use_container_width=True)
        except FileNotFoundError:
            st.error("Mitigated audit results not found.")

# ==========================================
# TAB 3: Final Outcome
# ==========================================
with tab3:
    st.header("Conclusion & Deployment Readiness")
    st.write("""
    ### Mitigation Effectiveness
    The post-processing threshold calibration successfully reduced the False Reject Rate (FRR) disparity among the most marginalized groups. 
    By allowing a localized threshold adjustment, the system became more equitable in authenticating underrepresented demographics.
    
    ### Remaining Ethical Risks
    Despite mathematical mitigation, significant ethical risks remain:
    * **Proxy Limitations:** Broad categorical labels (like race/ethnicity) are imperfect proxies for continuous physical traits, meaning micro-biases still exist within these aggregated bins.
    * **FAR Security Trade-off:** Accommodating higher FRR in underrepresented groups inherently required relaxing the False Accept Rate (FAR) tolerance, slightly increasing the risk of false matches.
    
    ### Final Recommendation
    **NOT RECOMMENDED FOR HIGH-STAKES DEPLOYMENT.** While the algorithmic fairness audit successfully reduced measured disparities, the baseline variance in subgroup performance and the reliance on socially constructed demographic proxies pose an unacceptable risk for environments like airport security or law enforcement. Deployment should only proceed with a strict "human-in-the-loop" oversight protocol.
    """)
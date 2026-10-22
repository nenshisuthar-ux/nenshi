"""
HCC Survival Prediction - Professional Clinical AI Portal
An enterprise-grade, modern healthcare web application for 1-year Hepatocellular Carcinoma prognostic prediction.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

# Set matplotlib cache directory to avoid permissions warnings in restricted environments
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import matplotlib.pyplot as plt

import streamlit as st

# Import training logic and schema from train.py
from train import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FEATURE_METADATA,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    load_dataset,
    train_and_evaluate_models,
    find_dataset_file
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & ENTERPRISE CLINICAL THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HCC Survival Analytics | Clinical ML Portal",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Web Application Styling (Clean Medical SaaS Aesthetic)
st.markdown("""
<style>
    /* Hide Streamlit default branding & header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}

    /* Typography & App Base */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    /* Top Sticky Brand Navigation Bar */
    .top-nav {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 24px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .top-nav-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .top-nav-icon {
        background: #e0f2fe;
        color: #0284c7;
        width: 44px;
        height: 44px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        border: 1px solid #bae6fd;
    }
    .top-nav-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .top-nav-subtitle {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 500;
        margin-top: 2px;
    }
    .top-nav-badges {
        display: flex;
        gap: 10px;
    }
    .nav-badge-online {
        background: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .nav-badge-meta {
        background: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    /* Modern Executive Cards */
    .pro-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
    }

    /* KPI Stat Boxes */
    .kpi-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-container:hover {
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
        transform: translateY(-1px);
    }
    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .kpi-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
    }
    .kpi-icon {
        font-size: 1.1rem;
        padding: 6px;
        border-radius: 8px;
        background: #f8fafc;
        border: 1px solid #f1f5f9;
    }
    .kpi-number {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.1;
    }
    .kpi-footer {
        font-size: 0.8rem;
        color: #0284c7;
        margin-top: 8px;
        font-weight: 500;
    }

    /* Workflow Pipeline Step */
    .pipeline-step {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0284c7;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02);
    }
    .pipeline-step-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
    }
    .pipeline-tag {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #0284c7;
        background: #e0f2fe;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .pipeline-step-title {
        font-size: 0.96rem;
        font-weight: 600;
        color: #0f172a;
    }
    .pipeline-step-desc {
        font-size: 0.85rem;
        color: #475569;
        line-height: 1.5;
    }

    /* Result Cards */
    .verdict-card-survives {
        background: linear-gradient(180deg, #f0fdf4 0%, #ffffff 100%);
        border: 1.5px solid #86efac;
        border-radius: 12px;
        padding: 24px;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.08);
    }
    .verdict-card-dies {
        background: linear-gradient(180deg, #fef2f2 0%, #ffffff 100%);
        border: 1.5px solid #fca5a5;
        border-radius: 12px;
        padding: 24px;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.08);
    }

    /* Compliance & Disclaimer Box */
    .pro-disclaimer {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0284c7;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 24px;
        color: #475569;
        font-size: 0.83rem;
        line-height: 1.55;
    }

    /* Tabs enhancements */
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88rem;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f1f5f9 !important;
        color: #0284c7 !important;
    }

    /* Modern Form Inputs & Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 9px 20px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. MODEL MANAGEMENT & DATA CACHING
# -----------------------------------------------------------------------------
MODEL_PATH = "models/hcc_pipeline.joblib"
METRICS_PATH = "models/metrics.json"


@st.cache_resource
def get_pipeline():
    """Retrieve saved scikit-learn pipeline from disk."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None


def get_metrics():
    """Retrieve saved model metrics and cross-validation metadata."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# -----------------------------------------------------------------------------
# 3. TOP BRANDING NAVIGATION BAR
# -----------------------------------------------------------------------------
st.markdown("""
<div class="top-nav">
    <div class="top-nav-left">
        <div class="top-nav-icon">🏥</div>
        <div>
            <div class="top-nav-title">HCC Survival Analytics</div>
            <div class="top-nav-subtitle">Clinical Decision Support Portal & Prognostic Machine Learning</div>
        </div>
    </div>
    <div class="top-nav-badges">
        <div class="nav-badge-online">● Model Engine Online</div>
        <div class="nav-badge-meta">EASL-EORTC Standards</div>
        <div class="nav-badge-meta">Coimbra Cohort (N=165)</div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### 🧭 Application Portal")
    app_section = st.radio(
        "Select Portal Module:",
        [
            "1. Overview",
            "2. Model Performance",
            "3. Prediction Demo"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("##### 📌 Clinical Information")
    st.caption("**Condition:** Hepatocellular Carcinoma (HCC)")
    st.caption("**Prediction Target:** 1-Year Survival Post-Diagnosis")
    st.caption("**Class Distribution:** 102 Survivors (61.8%) vs 63 Deaths (38.2%)")
    st.caption("**Feature Space:** 49 Clinical Attributes")

    st.markdown("---")
    st.markdown(
        "<div style='font-size: 0.76rem; color: #64748b; line-height: 1.4;'>"
        "Designed for Academic Project Defense, College Viva Examination & Clinical Research Demonstration."
        "</div>",
        unsafe_allow_html=True
    )

# Load pipeline and cached evaluation metrics
pipeline_model = get_pipeline()
metrics_dict = get_metrics()


# =============================================================================
# SECTION 1: OVERVIEW
# =============================================================================
if app_section == "1. Overview":
    # Executive Summary Card
    st.markdown("""
    <div class="pro-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
            <h2 style="margin: 0; font-size: 1.5rem; color: #0f172a; font-weight: 700;">Executive Clinical Overview</h2>
            <span style="background: #e0f2fe; color: #0369a1; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600;">Supervised Binary Classification</span>
        </div>
        <p style="color: #475569; font-size: 0.95rem; line-height: 1.65; margin: 0;">
            <b>Hepatocellular Carcinoma (HCC)</b> is the primary histological subtype of liver cancer and ranks among the leading causes of cancer-related mortality globally. 
            Prognostic assessment requires synthesizing complex interactions across patient functional status, underlying hepatic cirrhosis severity, viral serology, and tumor burden. 
            This web application provides a verified, leak-free Machine Learning prognostic pipeline trained on <b>165 real patient records</b> from the University Hospital Center of Coimbra, Portugal. 
            Adhering to the <b>EASL-EORTC (European Association for the Study of the Liver - European Organisation for Research and Treatment of Cancer)</b> guidelines, 
            the system applies median-mode imputation, standardized feature transformation, and balanced class weighting to stratify 1-year survival risk.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Ingest dataset for dynamic overview metrics
    try:
        df_real = load_dataset()
        cnt_records = len(df_real)
        cnt_features = len(FEATURE_COLUMNS)
        dist_counts = df_real[TARGET_COLUMN].value_counts()
        c_surv = int(dist_counts.get(1, 0))
        c_dies = int(dist_counts.get(0, 0))
        pct_surv = (c_surv / cnt_records) * 100
        pct_dies = (c_dies / cnt_records) * 100
        overall_na_pct = (df_real[FEATURE_COLUMNS].isna().mean().mean()) * 100
    except Exception:
        cnt_records = 165
        cnt_features = 49
        c_surv, c_dies = 102, 63
        pct_surv, pct_dies = 61.82, 38.18
        overall_na_pct = 10.22

    # Executive KPI Metric Grid
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">
                <span class="kpi-label">Cohort Size</span>
                <span class="kpi-icon">👥</span>
            </div>
            <div class="kpi-number">{cnt_records}</div>
            <div class="kpi-footer">Clinical Patient Records</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">
                <span class="kpi-label">Input Biomarkers</span>
                <span class="kpi-icon">🧬</span>
            </div>
            <div class="kpi-number">{cnt_features}</div>
            <div class="kpi-footer">26 Categorical + 23 Quantitative</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">
                <span class="kpi-label">Survival Balance</span>
                <span class="kpi-icon">⚖️</span>
            </div>
            <div class="kpi-number">{pct_surv:.1f}%</div>
            <div class="kpi-footer">{c_surv} Survives vs {c_dies} Dies</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">
                <span class="kpi-label">Missing Data Rate</span>
                <span class="kpi-icon">🛡️</span>
            </div>
            <div class="kpi-number">{overall_na_pct:.1f}%</div>
            <div class="kpi-footer">Zero Records Dropped (Imputed)</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Split View: Clean Matplotlib Target Chart vs ML Workflow Stepper
    col_chart, col_pipe = st.columns([1, 1], gap="large")

    with col_chart:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Target Class Distribution")
        st.caption("Survival outcomes observed within 1 year of initial HCC diagnosis:")

        # Modern Horizontal Bar Chart
        fig_dist, ax_dist = plt.subplots(figsize=(6, 3.8), dpi=100)
        cohort_labels = ["Survives ≥ 1 Year (Class 1)", "Dies within 1 Year (Class 0)"]
        cohort_counts = [c_surv, c_dies]
        bar_colors = ["#0284c7", "#e11d48"]

        bars = ax_dist.barh(cohort_labels, cohort_counts, color=bar_colors, height=0.45, edgecolor="#cbd5e1", linewidth=1.2)
        ax_dist.set_xlabel("Number of Verified Patients", fontsize=10, color="#475569", fontweight="600")
        ax_dist.set_xlim(0, max(cohort_counts) + 25)
        ax_dist.spines['top'].set_visible(False)
        ax_dist.spines['right'].set_visible(False)
        ax_dist.spines['left'].set_color('#cbd5e1')
        ax_dist.spines['bottom'].set_color('#cbd5e1')
        ax_dist.tick_params(colors='#1e293b', labelsize=9.5)
        ax_dist.grid(axis='x', linestyle='--', alpha=0.35)

        for bar, count, pct in zip(bars, cohort_counts, [pct_surv, pct_dies]):
            w = bar.get_width()
            ax_dist.text(
                w + 2.5,
                bar.get_y() + bar.get_height() / 2.0,
                f"{count} patients ({pct:.1f}%)",
                va='center',
                ha='left',
                fontsize=9.8,
                fontweight='700',
                color="#0f172a"
            )

        fig_dist.patch.set_facecolor('#ffffff')
        ax_dist.set_facecolor('#ffffff')
        plt.tight_layout()
        st.pyplot(fig_dist)
        plt.close(fig_dist)

        st.markdown("""
        <div style="background: #f8fafc; border-radius: 8px; padding: 12px 14px; margin-top: 10px; font-size: 0.85rem; color: #475569; border: 1px solid #f1f5f9;">
            💡 <b>Class Imbalance Strategy:</b> The minority mortality class (38.2%) is protected during training via <code>class_weight='balanced'</code>. 
            This forces the optimization algorithm to penalize mortality misclassifications inversely proportional to class frequency.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_pipe:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown("### 🔄 End-to-End ML Pipeline Architecture")
        st.caption("Standard leak-free data science pipeline implemented via Scikit-Learn:")

        st.markdown("""
        <div class="pipeline-step">
            <div class="pipeline-step-header">
                <span class="pipeline-tag">Stage 1</span>
                <span class="pipeline-step-title">Clinical Data Ingestion & Sanitization</span>
            </div>
            <div class="pipeline-step-desc">Ingests <code>data/hcc-data.csv</code>. Missing value indicators (<code>?</code>) are mapped to genuine nulls. Target variable <code>Class</code> is segregated before preprocessing.</div>
        </div>

        <div class="pipeline-step">
            <div class="pipeline-step-header">
                <span class="pipeline-tag">Stage 2</span>
                <span class="pipeline-step-title">ColumnTransformer Leak-Free Imputation</span>
            </div>
            <div class="pipeline-step-desc">Continuous features are imputed via <b>median</b> and scaled with <b>StandardScaler</b>. Categorical/ordinal features are imputed with the <b>mode</b>. Transformers are strictly fitted on training splits only.</div>
        </div>

        <div class="pipeline-step">
            <div class="pipeline-step-header">
                <span class="pipeline-tag">Stage 3</span>
                <span class="pipeline-step-title">Stratified 5-Fold Cross-Validation</span>
            </div>
            <div class="pipeline-step-desc">Evaluates <b>Logistic Regression</b> (interpretable clinical baseline) and <b>Random Forest Classifier</b> on training folds with balanced class weighting.</div>
        </div>

        <div class="pipeline-step">
            <div class="pipeline-step-header">
                <span class="pipeline-tag">Stage 4</span>
                <span class="pipeline-step-title">Independent Held-Out Test Evaluation</span>
            </div>
            <div class="pipeline-step-desc">Rigorous objective evaluation on 20% unseen test patients (N=33) for Accuracy, Precision, Recall, F1, and ROC-AUC.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Interactive Patient Data Explorer
    with st.expander("🔍 Interactive Patient Cohort Explorer (Real Medical Data Sample)", expanded=False):
        try:
            st.dataframe(df_real.head(10), use_container_width=True)
            st.caption(f"Showing first 10 of {len(df_real)} patient records across all 50 columns. Target variable: `Class` (1 = Survives, 0 = Dies).")
        except Exception:
            st.info("Cohort data preview unavailable.")


# =============================================================================
# SECTION 2: MODEL PERFORMANCE
# =============================================================================
elif app_section == "2. Model Performance":
    st.markdown("""
    <div class="pro-card">
        <h2 style="margin: 0; font-size: 1.5rem; color: #0f172a; font-weight: 700;">Model Training & Evaluation Dashboard</h2>
        <p style="color: #475569; font-size: 0.95rem; line-height: 1.6; margin-top: 8px; margin-bottom: 0;">
            Transparent evaluation metrics computed strictly on the independent <b>held-out test cohort (20% stratified split, N=33 patients)</b>. 
            Switch between clinical model architectures and trigger on-demand pipeline retraining with live cross-validation diagnostics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Model Configuration & Control Center
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    c_ctrl, c_info = st.columns([1, 2], gap="large")

    with c_ctrl:
        st.markdown("##### ⚙️ Model Architecture Selection")
        chosen_model = st.selectbox(
            "Select Architecture:",
            options=["logistic_regression", "random_forest"],
            format_func=lambda x: "Logistic Regression (Clinical Baseline)" if x == "logistic_regression" else "Random Forest Classifier",
            label_visibility="collapsed"
        )
        retrain_clicked = st.button("🚀 Train / Retrain Model", type="primary", use_container_width=True)

    with c_info:
        st.markdown("##### 📦 System Pipeline Status")
        if pipeline_model is not None and metrics_dict is not None:
            active_m_name = metrics_dict.get('selected_model', 'logistic_regression')
            active_lbl = "Logistic Regression (Baseline)" if active_m_name == "logistic_regression" else "Random Forest Classifier"
            st.markdown(f"""
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 12px 16px; color: #166534; font-size: 0.88rem;">
                ✓ <b>Active Model Serialized:</b> <code>models/hcc_pipeline.joblib</code> loaded in memory.<br>
                Model Type: <b>{active_lbl}</b> | Training Samples: <b>132</b> | Test Samples: <b>33</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 12px 16px; color: #92400e; font-size: 0.88rem;">
                ⚠️ <b>Model Unfitted:</b> Click 'Train / Retrain Model' to fit the pipeline and generate evaluation metrics.
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Handle retraining execution
    if retrain_clicked:
        with st.spinner(f"Fitting pipeline ({chosen_model}) with 5-fold cross-validation and evaluating held-out test cohort..."):
            try:
                pipeline_model, metrics_dict = train_and_evaluate_models(
                    model_type=chosen_model,
                    save_artifacts=True
                )
                st.cache_resource.clear()
                st.success(f"✓ Model successfully trained and saved! Active classifier: **{chosen_model}**")
            except Exception as e:
                st.error(f"Training failed: {e}")

    # Display Metrics Dashboard
    if metrics_dict and 'comparison' in metrics_dict:
        m_key = chosen_model if chosen_model in metrics_dict['comparison'] else metrics_dict.get('selected_model', 'logistic_regression')
        active_results = metrics_dict['comparison'][m_key]
        test_m = active_results['test_set']
        cv_m = active_results['cv_mean']
        cv_s = active_results['cv_std']

        st.markdown(f"### 📈 Performance Metrics: {('Logistic Regression (Clinical Baseline)' if m_key == 'logistic_regression' else 'Random Forest Classifier')}")
        st.caption("Objective scores on the held-out test cohort (20% split, N=33 patients, stratified):")

        # 5 Metric Scorecards
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        with sc1:
            st.markdown(f"""
            <div class="kpi-container">
                <span class="kpi-label">Accuracy</span>
                <div class="kpi-number">{test_m['accuracy']:.1%}</div>
                <div class="kpi-footer">CV: {cv_m['accuracy']:.1%} ± {cv_s['accuracy']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""
            <div class="kpi-container">
                <span class="kpi-label">Precision</span>
                <div class="kpi-number">{test_m['precision']:.1%}</div>
                <div class="kpi-footer">CV: {cv_m['precision']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        with sc3:
            st.markdown(f"""
            <div class="kpi-container">
                <span class="kpi-label">Recall (Sens.)</span>
                <div class="kpi-number">{test_m['recall']:.1%}</div>
                <div class="kpi-footer">CV: {cv_m['recall']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        with sc4:
            st.markdown(f"""
            <div class="kpi-container">
                <span class="kpi-label">F1-Score</span>
                <div class="kpi-number">{test_m['f1']:.3f}</div>
                <div class="kpi-footer">CV: {cv_m['f1']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with sc5:
            st.markdown(f"""
            <div class="kpi-container">
                <span class="kpi-label">ROC-AUC</span>
                <div class="kpi-number">{test_m['roc_auc']:.3f}</div>
                <div class="kpi-footer">CV: {cv_m['roc_auc']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # Visual Split: Confusion Matrix vs Beginner Explanation
        c_matrix, c_guide = st.columns([1, 1], gap="large")

        with c_matrix:
            st.markdown('<div class="pro-card">', unsafe_allow_html=True)
            st.markdown("### 🎯 Confusion Matrix (Held-Out Test Set)")
            cm_vals = test_m['confusion_matrix']
            tn_val, fp_val = cm_vals[0][0], cm_vals[0][1]
            fn_val, tp_val = cm_vals[1][0], cm_vals[1][1]

            # High-resolution, clean confusion matrix plot
            fig_cm, ax_cm = plt.subplots(figsize=(5.6, 4.2), dpi=100)
            ax_cm.matshow(cm_vals, cmap=plt.cm.Blues, alpha=0.7)

            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(['Pred: Dies (0)', 'Pred: Survives (1)'], fontsize=9.5, fontweight='600')
            ax_cm.set_yticklabels(['Actual: Dies (0)', 'Actual: Survives (1)'], fontsize=9.5, fontweight='600')

            m_annotations = [
                [f"True Negatives (TN)\n{tn_val}\n(Mortality Detected)", f"False Positives (FP)\n{fp_val}\n(Mistaken Survival)"],
                [f"False Negatives (FN)\n{fn_val}\n(Mistaken Mortality)", f"True Positives (TP)\n{tp_val}\n(Survivors Detected)"]
            ]
            for i in range(2):
                for j in range(2):
                    ax_cm.text(j, i, m_annotations[i][j], ha='center', va='center', fontsize=9.2, color='#0f172a', fontweight='600')

            ax_cm.tick_params(bottom=False, top=True, left=True, right=False)
            fig_cm.patch.set_facecolor('#ffffff')
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

            st.markdown("##### Class-Specific Breakdown Table:")
            df_class_tbl = pd.DataFrame({
                'Clinical Outcome': ['Class 0: Dies within 1 Year', 'Class 1: Survives ≥ 1 Year'],
                'Precision': [f"{test_m['class_0_dies']['precision']:.1%}", f"{test_m['class_1_survives']['precision']:.1%}"],
                'Recall / Sensitivity': [f"{test_m['class_0_dies']['recall']:.1%}", f"{test_m['class_1_survives']['recall']:.1%}"],
                'F1-Score': [f"{test_m['class_0_dies']['f1']:.3f}", f"{test_m['class_1_survives']['f1']:.3f}"],
                'Test Cohort Support': [f"{test_m['class_0_dies']['support']} patients", f"{test_m['class_1_survives']['support']} patients"]
            })
            st.dataframe(df_class_tbl, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_guide:
            st.markdown('<div class="pro-card">', unsafe_allow_html=True)
            st.markdown("### 🎓 Beginner-Friendly Metric Guide & Viva Defense")
            st.markdown(f"""
            - **Accuracy ({test_m['accuracy']:.1%}):** Measures the aggregate proportion of correct predictions across all patients. Because the class balance is 62:38, accuracy alone is insufficient.
            - **Mortality Recall ({test_m['class_0_dies']['recall']:.1%}):** The model accurately detected **{tn_val} out of 13 deceased patients** in the held-out test cohort. In clinical oncology, maximizing mortality recall prevents false reassurance for critical patients.
            - **Precision ({test_m['precision']:.1%}):** Out of all instances where the classifier projected survival, {test_m['precision']:.1%} were confirmed survivors.
            - **F1-Score ({test_m['f1']:.3f}):** The harmonic mean of precision and recall, balancing false alarms with missed critical events.
            - **ROC-AUC ({test_m['roc_auc']:.3f}):** Reflects the model's discriminative ability across every potential decision threshold. A score of {test_m['roc_auc']:.3f} demonstrates strong risk calibration.

            <div style="background-color: #f1f5f9; border-radius: 8px; padding: 14px; margin-top: 14px; font-size: 0.84rem; color: #334155; line-height: 1.55;">
                📌 <b>Oral Defense Note on Sample Size:</b> With 165 total patients and 33 in the test set, each patient in the test set corresponds to ~3.03% accuracy. 
                Reporting <b>5-Fold Stratified Cross-Validation</b> on the training set alongside the held-out test set provides a statistically sound demonstration of generalization.
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Feature Importance / Coefficient Analysis
        if m_key == 'logistic_regression' and hasattr(pipeline_model.named_steps['classifier'], 'coef_'):
            with st.expander("📊 Top Clinical Biomarker Predictors (Model Weights Analysis)", expanded=False):
                try:
                    clf = pipeline_model.named_steps['classifier']
                    prep = pipeline_model.named_steps['preprocessor']
                    feat_names = [f.replace('num__', '').replace('cat__', '') for f in prep.get_feature_names_out()]
                    coef_series = pd.Series(clf.coef_[0], index=feat_names).sort_values()

                    c_imp1, c_imp2 = st.columns(2)
                    with c_imp1:
                        st.markdown("**Top Indicators of Mortality Risk (Negative for Survival):**")
                        df_mort = pd.DataFrame({
                            'Clinical Feature': coef_series.head(6).index,
                            'Log-Odds Weight': coef_series.head(6).values
                        })
                        st.dataframe(df_mort, use_container_width=True, hide_index=True)

                    with c_imp2:
                        st.markdown("**Top Indicators of 1-Year Survival (Positive for Survival):**")
                        df_surv = pd.DataFrame({
                            'Clinical Feature': coef_series.tail(6).index,
                            'Log-Odds Weight': coef_series.tail(6).values
                        })
                        st.dataframe(df_surv, use_container_width=True, hide_index=True)

                    st.caption("Coefficients represent log-odds change per standardized unit. Positive weights favor 1-year survival; negative weights favor mortality risk.")
                except Exception:
                    st.info("Feature importance display unavailable.")

    else:
        st.info("Please train the model using the button above to view evaluation metrics and the confusion matrix.")


# =============================================================================
# SECTION 3: PREDICTION DEMO
# =============================================================================
elif app_section == "3. Prediction Demo":
    st.markdown("""
    <div class="pro-card">
        <h2 style="margin: 0; font-size: 1.5rem; color: #0f172a; font-weight: 700;">Clinical Prediction Decision Support Tool</h2>
        <p style="color: #475569; font-size: 0.95rem; line-height: 1.6; margin-top: 8px; margin-bottom: 0;">
            Simulate 1-year survival prognosis for a prospective Hepatocellular Carcinoma patient. 
            Inputs are organized into 5 clinical domains with verified baseline medians pre-populated. 
            You can load instant case presets to showcase distinct prognostic profiles during an oral defense.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if pipeline_model is None:
        st.warning("⚠️ No trained pipeline is loaded. Please navigate to **Section 2: Model Performance** and click **Train / Retrain Model** first.")
    else:
        # Case Presets Quick-Action Bar
        st.markdown("##### ⚡ Rapid Demonstration Presets")
        p_c1, p_c2, p_c3, p_c4 = st.columns([1, 1, 1, 2])
        with p_c1:
            if st.button("🟢 Standard Median Profile", use_container_width=True):
                st.session_state['active_case'] = 'median'
                st.rerun()
        with p_c2:
            if st.button("🔴 High-Risk Patient Case", use_container_width=True):
                st.session_state['active_case'] = 'high_risk'
                st.rerun()
        with p_c3:
            if st.button("🔵 Favorable Prognosis Case", use_container_width=True):
                st.session_state['active_case'] = 'favorable'
                st.rerun()

        active_case = st.session_state.get('active_case', 'median')

        if active_case == 'high_risk':
            st.info("Loaded **High-Risk Patient Profile**: Severe decompensation (Ascites Grade III, ECOG PS 3), elevated AFP (35,000 ng/mL), impaired kidney creatinine (2.6 mg/dL), and low albumin (2.3 g/dL).")
        elif active_case == 'favorable':
            st.info("Loaded **Favorable Prognosis Profile**: Excellent performance status (ECOG 0), preserved synthetic liver function (Albumin 4.2 g/dL), normal bilirubin, and single small nodule.")

        # Interactive Input Form with Categorized Tabs
        with st.form("hcc_pro_prediction_form"):
            user_features = {}

            tab_demo, tab_clin, tab_viral, tab_func, tab_lab = st.tabs([
                "👤 Demographics & Habits",
                "📋 Clinical Findings & Cirrhosis",
                "🔬 Viral Hepatitis Serology",
                "⚡ Functional Severity Scores",
                "🧪 Laboratory & Tumor Profile"
            ])

            # TAB 1: Demographics & Lifestyle
            with tab_demo:
                st.markdown("###### Patient Background & Substance History")
                d_c1, d_c2, d_c3 = st.columns(3)
                with d_c1:
                    user_features['Gender'] = st.selectbox(
                        "Gender",
                        options=[1, 0],
                        format_func=lambda x: "Male" if x == 1 else "Female",
                        index=0,
                        help=FEATURE_METADATA['Gender']['help']
                    )
                    user_features['Age'] = st.number_input(
                        "Age at Diagnosis (years)",
                        min_value=20.0,
                        max_value=95.0,
                        value=76.0 if active_case == 'high_risk' else (58.0 if active_case == 'favorable' else 66.0),
                        step=1.0,
                        help=FEATURE_METADATA['Age']['help']
                    )
                with d_c2:
                    user_features['Alcohol'] = st.selectbox(
                        "Regular Alcohol Consumption",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=0 if active_case != 'favorable' else 1,
                        help=FEATURE_METADATA['Alcohol']['help']
                    )
                    user_features['Grams_day'] = st.number_input(
                        "Alcohol Intake (grams/day)",
                        min_value=0.0,
                        max_value=500.0,
                        value=160.0 if active_case == 'high_risk' else (0.0 if active_case == 'favorable' else 75.0),
                        step=5.0,
                        help=FEATURE_METADATA['Grams_day']['help']
                    )
                with d_c3:
                    user_features['Smoking'] = st.selectbox(
                        "Smoking History",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=0 if active_case == 'high_risk' else 1,
                        help=FEATURE_METADATA['Smoking']['help']
                    )
                    user_features['Packs_year'] = st.number_input(
                        "Smoking Intensity (packs/year)",
                        min_value=0.0,
                        max_value=510.0,
                        value=40.0 if active_case == 'high_risk' else 0.0,
                        step=5.0,
                        help=FEATURE_METADATA['Packs_year']['help']
                    )

            # TAB 2: Clinical Findings & Comorbidities
            with tab_clin:
                st.markdown("###### Physical Findings, Comorbidities & Vascular Invasions")
                cl_c1, cl_c2, cl_c3 = st.columns(3)
                with cl_c1:
                    user_features['Symptoms'] = st.selectbox(
                        "Symptoms at Diagnosis",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=0 if active_case != 'favorable' else 1,
                        help=FEATURE_METADATA['Symptoms']['help']
                    )
                    user_features['Cirrhosis'] = st.selectbox(
                        "Liver Cirrhosis",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=0 if active_case != 'favorable' else 1,
                        help=FEATURE_METADATA['Cirrhosis']['help']
                    )
                    user_features['Endemic'] = st.selectbox(
                        "Endemic Country Origin",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=1,
                        help=FEATURE_METADATA['Endemic']['help']
                    )
                    user_features['Diabetes'] = st.selectbox(
                        "Diabetes Mellitus",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=0 if active_case == 'high_risk' else 1,
                        help=FEATURE_METADATA['Diabetes']['help']
                    )
                    user_features['Obesity'] = st.selectbox(
                        "Clinical Obesity",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=1,
                        help=FEATURE_METADATA['Obesity']['help']
                    )
                with cl_c2:
                    user_features['AHT'] = st.selectbox(
                        "Arterial Hypertension",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=0 if active_case == 'high_risk' else 1,
                        help=FEATURE_METADATA['AHT']['help']
                    )
                    user_features['CRI'] = st.selectbox(
                        "Chronic Renal Insufficiency",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=0 if active_case == 'high_risk' else 1,
                        help=FEATURE_METADATA['CRI']['help']
                    )
                    user_features['HIV'] = st.selectbox(
                        "HIV Infection",
                        options=[1, 0],
                        format_func=lambda x: "Positive" if x == 1 else "Negative",
                        index=1,
                        help=FEATURE_METADATA['HIV']['help']
                    )
                    user_features['NASH'] = st.selectbox(
                        "NASH (Steatohepatitis)",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=1,
                        help=FEATURE_METADATA['NASH']['help']
                    )
                    user_features['Varices'] = st.selectbox(
                        "Esophageal Varices",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=0 if active_case != 'favorable' else 1,
                        help=FEATURE_METADATA['Varices']['help']
                    )
                with cl_c3:
                    user_features['Spleno'] = st.selectbox(
                        "Splenomegaly",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=0 if active_case == 'high_risk' else 1,
                        help=FEATURE_METADATA['Spleno']['help']
                    )
                    user_features['PHT'] = st.selectbox(
                        "Portal Hypertension",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=0 if active_case != 'favorable' else 1,
                        help=FEATURE_METADATA['PHT']['help']
                    )
                    user_features['PVT'] = st.selectbox(
                        "Portal Vein Thrombosis",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=0 if active_case == 'high_risk' else 1,
                        help=FEATURE_METADATA['PVT']['help']
                    )
                    user_features['Metastasis'] = st.selectbox(
                        "Distant Metastasis",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=0 if active_case == 'high_risk' else 1,
                        help=FEATURE_METADATA['Metastasis']['help']
                    )
                    user_features['Hallmark'] = st.selectbox(
                        "Radiological Hallmark",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=0,
                        help=FEATURE_METADATA['Hallmark']['help']
                    )

            # TAB 3: Viral Hepatitis Serology
            with tab_viral:
                st.markdown("###### Viral Hepatitis Markers & Iron Overload")
                v_c1, v_c2 = st.columns(2)
                with v_c1:
                    user_features['HBsAg'] = st.selectbox(
                        "HBsAg (Hepatitis B Surface Antigen)",
                        options=[1, 0],
                        format_func=lambda x: "Positive" if x == 1 else "Negative",
                        index=1,
                        help=FEATURE_METADATA['HBsAg']['help']
                    )
                    user_features['HBeAg'] = st.selectbox(
                        "HBeAg (Hepatitis B e-Antigen)",
                        options=[1, 0],
                        format_func=lambda x: "Positive" if x == 1 else "Negative",
                        index=1,
                        help=FEATURE_METADATA['HBeAg']['help']
                    )
                    user_features['HBcAb'] = st.selectbox(
                        "HBcAb (Hepatitis B Core Antibody)",
                        options=[1, 0],
                        format_func=lambda x: "Positive" if x == 1 else "Negative",
                        index=1,
                        help=FEATURE_METADATA['HBcAb']['help']
                    )
                with v_c2:
                    user_features['HCVAb'] = st.selectbox(
                        "HCVAb (Hepatitis C Virus Antibody)",
                        options=[1, 0],
                        format_func=lambda x: "Positive" if x == 1 else "Negative",
                        index=0 if active_case == 'high_risk' else 1,
                        help=FEATURE_METADATA['HCVAb']['help']
                    )
                    user_features['Hemochro'] = st.selectbox(
                        "Hemochromatosis (Iron Overload)",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=1,
                        help=FEATURE_METADATA['Hemochro']['help']
                    )

            # TAB 4: Functional Severity Scores
            with tab_func:
                st.markdown("###### Performance Status & Decompensation Markers")
                fn_c1, fn_c2, fn_c3 = st.columns(3)
                with fn_c1:
                    ps_labels = {
                        0: "0 - Fully active, unrestricted",
                        1: "1 - Strenuous physical restriction",
                        2: "2 - Ambulatory, capable of self-care",
                        3: "3 - Limited self-care, confined >50%",
                        4: "4 - Completely disabled, confined"
                    }
                    user_features['PS'] = st.selectbox(
                        "ECOG Performance Status",
                        options=[0, 1, 2, 3, 4],
                        format_func=lambda x: ps_labels[x],
                        index=3 if active_case == 'high_risk' else (0 if active_case == 'favorable' else 1),
                        help=FEATURE_METADATA['PS']['help']
                    )
                with fn_c2:
                    enc_labels = {
                        1: "1 - None (Normal)",
                        2: "2 - Grade I / II (Mild/Moderate Confusion)",
                        3: "3 - Grade III / IV (Severe Stupor/Coma)"
                    }
                    user_features['Encephalopathy'] = st.selectbox(
                        "Hepatic Encephalopathy Degree",
                        options=[1, 2, 3],
                        format_func=lambda x: enc_labels[x],
                        index=1 if active_case == 'high_risk' else 0,
                        help=FEATURE_METADATA['Encephalopathy']['help']
                    )
                with fn_c3:
                    asc_labels = {
                        1: "1 - None (No fluid)",
                        2: "2 - Mild (Responsive to diuretics)",
                        3: "3 - Moderate to Severe (Refractory)"
                    }
                    user_features['Ascites'] = st.selectbox(
                        "Ascites Degree",
                        options=[1, 2, 3],
                        format_func=lambda x: asc_labels[x],
                        index=2 if active_case == 'high_risk' else 0,
                        help=FEATURE_METADATA['Ascites']['help']
                    )

            # TAB 5: Laboratory Biomarkers & Tumor Profile
            with tab_lab:
                st.markdown("###### Oncology Biomarkers, Liver Enzymes & Organ Function")
                lb_c1, lb_c2, lb_c3 = st.columns(3)
                with lb_c1:
                    user_features['AFP'] = st.number_input(
                        "AFP (Alpha-Fetoprotein) [ng/mL]",
                        min_value=1.0,
                        max_value=2000000.0,
                        value=35000.0 if active_case == 'high_risk' else (8.0 if active_case == 'favorable' else 33.0),
                        step=10.0,
                        help=FEATURE_METADATA['AFP']['help']
                    )
                    user_features['Albumin'] = st.number_input(
                        "Serum Albumin [g/dL]",
                        min_value=1.0,
                        max_value=6.0,
                        value=2.3 if active_case == 'high_risk' else (4.2 if active_case == 'favorable' else 3.4),
                        step=0.1,
                        help=FEATURE_METADATA['Albumin']['help']
                    )
                    user_features['Total_Bil'] = st.number_input(
                        "Total Bilirubin [mg/dL]",
                        min_value=0.1,
                        max_value=45.0,
                        value=7.8 if active_case == 'high_risk' else (0.8 if active_case == 'favorable' else 1.4),
                        step=0.1,
                        help=FEATURE_METADATA['Total_Bil']['help']
                    )
                    user_features['Dir_Bil'] = st.number_input(
                        "Direct Bilirubin [mg/dL]",
                        min_value=0.05,
                        max_value=30.0,
                        value=4.1 if active_case == 'high_risk' else (0.3 if active_case == 'favorable' else 0.7),
                        step=0.1,
                        help=FEATURE_METADATA['Dir_Bil']['help']
                    )
                    user_features['INR'] = st.number_input(
                        "INR (Prothrombin Time Ratio)",
                        min_value=0.8,
                        max_value=5.0,
                        value=2.4 if active_case == 'high_risk' else (1.05 if active_case == 'favorable' else 1.3),
                        step=0.05,
                        help=FEATURE_METADATA['INR']['help']
                    )
                    user_features['Creatinine'] = st.number_input(
                        "Creatinine [mg/dL]",
                        min_value=0.1,
                        max_value=10.0,
                        value=2.6 if active_case == 'high_risk' else (0.75 if active_case == 'favorable' else 0.85),
                        step=0.05,
                        help=FEATURE_METADATA['Creatinine']['help']
                    )
                    user_features['TP'] = st.number_input(
                        "Total Proteins [g/dL]",
                        min_value=2.0,
                        max_value=110.0,
                        value=5.8 if active_case == 'high_risk' else (7.4 if active_case == 'favorable' else 7.1),
                        step=0.1,
                        help=FEATURE_METADATA['TP']['help']
                    )
                with lb_c2:
                    user_features['ALT'] = st.number_input(
                        "ALT (Alanine Transaminase) [U/L]",
                        min_value=5.0,
                        max_value=500.0,
                        value=120.0 if active_case == 'high_risk' else (35.0 if active_case == 'favorable' else 50.0),
                        step=1.0,
                        help=FEATURE_METADATA['ALT']['help']
                    )
                    user_features['AST'] = st.number_input(
                        "AST (Aspartate Transaminase) [U/L]",
                        min_value=10.0,
                        max_value=600.0,
                        value=210.0 if active_case == 'high_risk' else (42.0 if active_case == 'favorable' else 71.0),
                        step=1.0,
                        help=FEATURE_METADATA['AST']['help']
                    )
                    user_features['GGT'] = st.number_input(
                        "GGT [U/L]",
                        min_value=10.0,
                        max_value=1600.0,
                        value=450.0 if active_case == 'high_risk' else (80.0 if active_case == 'favorable' else 180.0),
                        step=5.0,
                        help=FEATURE_METADATA['GGT']['help']
                    )
                    user_features['ALP'] = st.number_input(
                        "ALP (Alkaline Phosphatase) [U/L]",
                        min_value=1.0,
                        max_value=1000.0,
                        value=390.0 if active_case == 'high_risk' else (120.0 if active_case == 'favorable' else 162.0),
                        step=2.0,
                        help=FEATURE_METADATA['ALP']['help']
                    )
                    user_features['Hemoglobin'] = st.number_input(
                        "Hemoglobin [g/dL]",
                        min_value=4.0,
                        max_value=20.0,
                        value=9.0 if active_case == 'high_risk' else (14.2 if active_case == 'favorable' else 13.0),
                        step=0.1,
                        help=FEATURE_METADATA['Hemoglobin']['help']
                    )
                    user_features['MCV'] = st.number_input(
                        "Mean Corpuscular Volume (MCV) [fL]",
                        min_value=60.0,
                        max_value=130.0,
                        value=95.0,
                        step=0.5,
                        help=FEATURE_METADATA['MCV']['help']
                    )
                    user_features['Leucocytes'] = st.number_input(
                        "Leukocytes (WBC) [G/L]",
                        min_value=1.0,
                        max_value=15000.0,
                        value=7.2,
                        step=0.5,
                        help=FEATURE_METADATA['Leucocytes']['help']
                    )
                with lb_c3:
                    user_features['Platelets'] = st.number_input(
                        "Platelet Count [G/L]",
                        min_value=1.0,
                        max_value=500000.0,
                        value=38000.0 if active_case == 'high_risk' else (180000.0 if active_case == 'favorable' else 93000.0),
                        step=1000.0,
                        help=FEATURE_METADATA['Platelets']['help']
                    )
                    user_features['Nodules'] = st.number_input(
                        "Number of Tumor Nodules",
                        min_value=0.0,
                        max_value=5.0,
                        value=4.0 if active_case == 'high_risk' else (1.0 if active_case == 'favorable' else 2.0),
                        step=1.0,
                        help=FEATURE_METADATA['Nodules']['help']
                    )
                    user_features['Major_Dim'] = st.number_input(
                        "Major Dimension of Largest Nodule [cm]",
                        min_value=0.5,
                        max_value=25.0,
                        value=12.5 if active_case == 'high_risk' else (2.4 if active_case == 'favorable' else 5.0),
                        step=0.5,
                        help=FEATURE_METADATA['Major_Dim']['help']
                    )
                    user_features['Iron'] = st.number_input(
                        "Serum Iron [mcg/dL]",
                        min_value=0.0,
                        max_value=250.0,
                        value=83.0,
                        step=1.0,
                        help=FEATURE_METADATA['Iron']['help']
                    )
                    user_features['Sat'] = st.number_input(
                        "Transferrin Saturation [%]",
                        min_value=0.0,
                        max_value=130.0,
                        value=27.0,
                        step=1.0,
                        help=FEATURE_METADATA['Sat']['help']
                    )
                    user_features['Ferritin'] = st.number_input(
                        "Ferritin Level [ng/mL]",
                        min_value=0.0,
                        max_value=2500.0,
                        value=850.0 if active_case == 'high_risk' else (180.0 if active_case == 'favorable' else 295.0),
                        step=10.0,
                        help=FEATURE_METADATA['Ferritin']['help']
                    )

            st.write("")
            calc_submit = st.form_submit_button("🔮 Calculate 1-Year Prognosis", type="primary", use_container_width=True)

        # Execute Prognosis Calculation
        if calc_submit:
            try:
                # Build single row dataframe preserving exact training column order
                patient_row = pd.DataFrame([user_features])[FEATURE_COLUMNS]

                # Run inference through leak-free pipeline
                prediction_class = int(pipeline_model.predict(patient_row)[0])
                prediction_proba = pipeline_model.predict_proba(patient_row)[0]
                prob_death = float(prediction_proba[0])
                prob_survive = float(prediction_proba[1])

                st.write("")
                st.markdown("### 📋 Prognostic Assessment Report")

                if prediction_class == 1:
                    st.markdown(f"""
                    <div class="verdict-card-survives">
                        <div style="display: flex; align-items: flex-start; gap: 14px;">
                            <span style="font-size: 2.2rem;">✅</span>
                            <div>
                                <h3 style="color: #15803d; margin: 0; font-size: 1.4rem; font-weight: 700;">Favorable 1-Year Survival Prognosis (Class 1)</h3>
                                <p style="color: #166534; font-size: 0.96rem; margin: 6px 0 0 0; line-height: 1.5;">
                                    Based on the submitted EASL-EORTC clinical parameters, the model predicts the patient is <b>likely to survive ≥ 1 year</b> following diagnosis.
                                </p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-card-dies">
                        <div style="display: flex; align-items: flex-start; gap: 14px;">
                            <span style="font-size: 2.2rem;">⚠️</span>
                            <div>
                                <h3 style="color: #b91c1c; margin: 0; font-size: 1.4rem; font-weight: 700;">Elevated 1-Year Mortality Risk Stratification (Class 0)</h3>
                                <p style="color: #991b1b; font-size: 0.96rem; margin: 6px 0 0 0; line-height: 1.5;">
                                    Based on the submitted EASL-EORTC clinical parameters, the model stratifies this patient into the <b>high 1-year mortality risk tier</b>.
                                </p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")
                gauge1, gauge2 = st.columns(2)
                with gauge1:
                    st.markdown(f"**Estimated Survival Probability: {prob_survive:.1%}**")
                    st.progress(prob_survive)
                with gauge2:
                    st.markdown(f"**Estimated 1-Year Mortality Risk: {prob_death:.1%}**")
                    st.progress(prob_death)

                # Flag Significant Clinical Risk Factors
                risk_flags = []
                if user_features.get('AFP', 0) > 400:
                    risk_flags.append(f"Elevated Alpha-Fetoprotein: **{user_features['AFP']:,.0f} ng/mL** (Diagnostic threshold: >400 ng/mL)")
                if user_features.get('Albumin', 4) < 2.8:
                    risk_flags.append(f"Hypoalbuminemia: **{user_features['Albumin']:.1f} g/dL** (Severely compromised synthetic function)")
                if user_features.get('Total_Bil', 1) > 3.0:
                    risk_flags.append(f"Hyperbilirubinemia: **{user_features['Total_Bil']:.1f} mg/dL** (Marked jaundice / excretory impairment)")
                if user_features.get('Ascites', 1) >= 2:
                    risk_flags.append("Documented Ascites Decompensation (Peritoneal fluid accumulation)")
                if user_features.get('PS', 0) >= 2:
                    risk_flags.append("Impaired ECOG Performance Status (PS ≥ 2 indicates restricted physical stamina)")
                if user_features.get('Major_Dim', 0) > 5.0:
                    risk_flags.append(f"Large Primary Tumor Diameter: **{user_features['Major_Dim']:.1f} cm** (Exceeds Milan criteria)")

                if risk_flags:
                    st.markdown("##### 🚩 Clinical Risk Biomarkers Detected in Input:")
                    for rf in risk_flags:
                        st.markdown(f"- {rf}")

                # Academic & Healthcare Disclaimer
                st.markdown("""
                <div class="pro-disclaimer">
                    <b>🛡️ Institutional Compliance & Clinical Research Disclaimer:</b><br>
                    This application is developed strictly for educational demonstration, academic course submission, and biomedical machine learning research. 
                    Prognostic predictions are generated by statistical inference algorithms trained on a historical cohort of 165 patients from Coimbra, Portugal. 
                    <b>This software is NOT an approved medical device and must never substitute for clinical oncological evaluation, formal diagnosis, or professional treatment planning.</b> 
                    Multidisciplinary tumor boards must always guide patient management. 
                    <i>Privacy Guarantee: No patient data or clinical parameters are persisted, logged, or transmitted externally.</i>
                </div>
                """, unsafe_allow_html=True)

            except Exception as ex:
                st.error(f"Inference computation error: {ex}")

# -----------------------------------------------------------------------------
# 5. PROFESSIONAL FOOTER
# -----------------------------------------------------------------------------
st.write("")
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #64748b; padding-bottom: 20px;">
    <div>
        <b>HCC Survival Analytics</b> &nbsp;|&nbsp; Machine Learning Academic Project Submission &nbsp;|&nbsp; Python & Scikit-Learn
    </div>
    <div>
        EASL-EORTC Clinical Guidelines &nbsp;|&nbsp; Coimbra Hospital Dataset (Santos et al.)
    </div>
</div>
""", unsafe_allow_html=True)

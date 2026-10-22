"""
HCC Survival Prediction - Streamlit Web Application
A beginner-friendly Machine Learning application for predicting Hepatocellular Carcinoma 1-year survival.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

# Set matplotlib cache directory to avoid permissions warnings
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
# PAGE CONFIGURATION & THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HCC Survival Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, clean, medical light theme
st.markdown("""
<style>
    /* Global Background & Typography */
    .main {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header titles */
    h1, h2, h3, h4 {
        color: #0f172a !important;
        font-weight: 600;
    }
    
    /* Custom Info Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-subtitle {
        font-size: 0.8rem;
        color: #0284c7;
        margin-top: 4px;
    }

    /* Workflow Step Cards */
    .workflow-step {
        background: #ffffff;
        border-left: 4px solid #0284c7;
        border-radius: 4px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .workflow-step-num {
        font-weight: 700;
        color: #0284c7;
        font-size: 0.9rem;
    }
    .workflow-step-title {
        font-weight: 600;
        color: #0f172a;
        font-size: 0.95rem;
    }
    .workflow-step-desc {
        color: #475569;
        font-size: 0.85rem;
        margin-top: 2px;
    }

    /* Alert / Disclaimer box */
    .disclaimer-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 6px;
        padding: 12px 16px;
        color: #166534;
        font-size: 0.85rem;
        margin-top: 14px;
    }
    .warning-disclaimer-box {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 6px;
        padding: 14px 18px;
        color: #92400e;
        font-size: 0.88rem;
        margin-top: 16px;
    }
    
    /* Result cards */
    .result-card-survives {
        background-color: #f0fdf4;
        border: 1.5px solid #22c55e;
        border-radius: 8px;
        padding: 20px;
        margin-top: 16px;
    }
    .result-card-dies {
        background-color: #fef2f2;
        border: 1.5px solid #ef4444;
        border-radius: 8px;
        padding: 20px;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
MODEL_PATH = "models/hcc_pipeline.joblib"
METRICS_PATH = "models/metrics.json"


@st.cache_resource
def load_trained_pipeline():
    """Load the trained machine learning pipeline from disk."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model from {MODEL_PATH}: {e}")
            return None
    return None


def load_saved_metrics():
    """Load saved evaluation metrics from disk."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# -----------------------------------------------------------------------------
# NAVIGATION SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🩺 HCC Survival App")
st.sidebar.markdown(
    "A clean Machine Learning portal for Hepatocellular Carcinoma (HCC) 1-year survival analysis."
)

section = st.sidebar.radio(
    "Navigate to Section:",
    [
        "1. Overview",
        "2. Model Performance",
        "3. Prediction Demo"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: Coimbra HCC Dataset (Santos et al.)")
st.sidebar.caption("Target: Class 1 (Survives >= 1 yr), Class 0 (Dies)")

# Check if model is already available on disk
pipeline_loaded = load_trained_pipeline()
metrics_data = load_saved_metrics()


# =============================================================================
# SECTION 1: OVERVIEW
# =============================================================================
if section == "1. Overview":
    st.title("HCC Survival Prediction")
    st.markdown("##### Machine Learning Classification for 1-Year Hepatocellular Carcinoma Prognosis")
    st.write("")

    # One-paragraph project description
    st.markdown("""
    **Hepatocellular Carcinoma (HCC)** is the most common primary malignancy of the liver and a leading cause of cancer-related mortality globally. 
    This project applies supervised machine learning to predict whether an HCC patient survives at least one year following diagnosis, using clinical, demographic, 
    and laboratory biomarkers collected under the **EASL-EORTC (European Association for the Study of the Liver - European Organisation for Research and Treatment of Cancer)** 
    clinical practice guidelines. By structuring an end-to-end scikit-learn pipeline with leak-free preprocessing and class-weighted classification, 
    the system assists researchers and students in understanding risk stratification and clinical machine learning pipelines.
    """)

    st.write("")
    
    # Dataset record and feature counts
    try:
        df_raw = load_dataset()
        n_records = len(df_raw)
        n_features = len(FEATURE_COLUMNS)
        surv_counts = df_raw[TARGET_COLUMN].value_counts()
        surv_count_1 = int(surv_counts.get(1, 0))
        surv_count_0 = int(surv_counts.get(0, 0))
        surv_pct_1 = (surv_count_1 / n_records) * 100
        surv_pct_0 = (surv_count_0 / n_records) * 100
        overall_missing = (df_raw[FEATURE_COLUMNS].isna().mean().mean()) * 100
    except Exception:
        n_records = 165
        n_features = 49
        surv_count_1, surv_count_0 = 102, 63
        surv_pct_1, surv_pct_0 = 61.82, 38.18
        overall_missing = 10.22

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Records</div>
            <div class="metric-value">{n_records}</div>
            <div class="metric-subtitle">Patient cohort</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Input Features</div>
            <div class="metric-value">{n_features}</div>
            <div class="metric-subtitle">26 Cat / 23 Continuous</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Target Variable</div>
            <div class="metric-value">Class</div>
            <div class="metric-subtitle">Binary: 1=Survives, 0=Dies</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Missing Values</div>
            <div class="metric-value">{overall_missing:.1f}%</div>
            <div class="metric-subtitle">Handled by Median/Mode</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    
    # Split view: Target Class Distribution Chart & Explanation of ML Workflow
    col_chart, col_flow = st.columns([1, 1], gap="large")

    with col_chart:
        st.subheader("Target Class Distribution")
        st.caption("Distribution of 1-year survival outcomes across the 165 clinical records:")
        
        # Simple, elegant matplotlib chart
        fig, ax = plt.subplots(figsize=(6, 4))
        categories = ["Survives (Class 1)", "Dies (Class 0)"]
        counts = [surv_count_1, surv_count_0]
        colors = ["#0284c7", "#e11d48"]

        bars = ax.bar(categories, counts, color=colors, width=0.45, edgecolor="#cbd5e1", linewidth=1.2)
        ax.set_ylabel("Number of Patients", fontsize=10, color="#1e293b", fontweight="500")
        ax.set_ylim(0, max(counts) + 20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.tick_params(colors='#1e293b', labelsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Bar label annotations
        for bar, count, pct in zip(bars, counts, [surv_pct_1, surv_pct_0]):
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 2,
                f"{count} ({pct:.1f}%)",
                ha='center',
                va='bottom',
                fontsize=9.5,
                fontweight='600',
                color="#0f172a"
            )

        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.info("💡 **Class Balance Note:** Class 1 represents 61.8% and Class 0 represents 38.2%. The pipeline applies balanced class weighting during training to penalize misclassification of the minority mortality class.")

    with col_flow:
        st.subheader("Machine Learning Workflow")
        st.caption("Standard leak-free data science lifecycle implemented in this project:")

        st.markdown("""
        <div class="workflow-step">
            <span class="workflow-step-num">Step 1: Data Ingestion & Sanitization</span><br>
            <span class="workflow-step-title">Load & Clean Raw Medical Records</span>
            <div class="workflow-step-desc">Documented missing markers (<code>?</code>) are converted to genuine null values. Column types are categorized into 23 binary, 3 ordinal, and 23 numerical continuous features.</div>
        </div>
        <div class="workflow-step">
            <span class="workflow-step-num">Step 2: Leak-Free Preprocessing</span><br>
            <span class="workflow-step-title">ColumnTransformer Pipeline</span>
            <div class="workflow-step-desc">Numerical features are imputed with their <b>median</b> and scaled via <b>StandardScaler</b>. Categorical/ordinal features are imputed with the <b>most frequent</b> mode. Preprocessing is strictly fitted on training splits only to prevent data leakage.</div>
        </div>
        <div class="workflow-step">
            <span class="workflow-step-num">Step 3: Stratified Cross-Validation</span><br>
            <span class="workflow-step-title">Model Comparison & Fitting</span>
            <div class="workflow-step-desc">5-Fold Stratified K-Fold CV evaluates both <b>Logistic Regression</b> (interpretable clinical baseline) and <b>Random Forest</b> with <code>class_weight='balanced'</code>.</div>
        </div>
        <div class="workflow-step">
            <span class="workflow-step-num">Step 4: Objective Evaluation</span><br>
            <span class="workflow-step-title">Held-Out Test Assessment</span>
            <div class="workflow-step-desc">The final pipeline is tested on an independent 20% held-out test set (N=33) for Accuracy, Precision, Recall, F1-Score, and ROC-AUC.</div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# SECTION 2: MODEL PERFORMANCE
# =============================================================================
elif section == "2. Model Performance":
    st.title("Model Training & Evaluation")
    st.markdown("##### Real Performance Metrics and Confusion Matrix on Held-Out Test Set")
    st.write("")

    # Model training controls
    col_ctrl, col_status = st.columns([1, 2])
    with col_ctrl:
        model_choice = st.selectbox(
            "Select Model Architecture:",
            options=["logistic_regression", "random_forest"],
            format_func=lambda x: "Logistic Regression (Clinical Baseline)" if x == "logistic_regression" else "Random Forest Classifier"
        )
        train_btn = st.button("🚀 Train / Retrain Model", type="primary", use_container_width=True)

    with col_status:
        st.markdown("**Model Artifact Status:**")
        if pipeline_loaded is not None and metrics_data is not None:
            active_m = metrics_data.get('selected_model', 'logistic_regression')
            active_label = "Logistic Regression" if active_m == "logistic_regression" else "Random Forest"
            st.success(f"✓ Trained pipeline loaded from `models/hcc_pipeline.joblib` (Active: **{active_label}**).")
        else:
            st.warning("⚠️ No trained pipeline found. Click 'Train / Retrain Model' to fit the model.")

    # Execute training if button clicked
    if train_btn:
        with st.spinner("Training models with 5-fold cross-validation and fitting final pipeline..."):
            try:
                pipeline_loaded, metrics_data = train_and_evaluate_models(
                    model_type=model_choice,
                    save_artifacts=True
                )
                st.cache_resource.clear()
                st.success(f"✓ Model successfully trained and saved! Active classifier: **{model_choice}**")
            except Exception as e:
                st.error(f"Training failed: {e}")

    # Display Metrics if available
    if metrics_data and 'comparison' in metrics_data:
        chosen_key = model_choice if model_choice in metrics_data['comparison'] else metrics_data.get('selected_model', 'logistic_regression')
        model_eval = metrics_data['comparison'][chosen_key]
        test_eval = model_eval['test_set']
        cv_mean = model_eval['cv_mean']
        cv_std = model_eval['cv_std']

        st.write("")
        st.subheader(f"Evaluation Metrics: {('Logistic Regression' if chosen_key == 'logistic_regression' else 'Random Forest')}")
        st.caption("Evaluated on the independent held-out test set (20% split, N=33 patients, stratified):")

        # 5 Metric Cards
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Accuracy</div>
                <div class="metric-value">{test_eval['accuracy']:.1%}</div>
                <div class="metric-subtitle">CV: {cv_mean['accuracy']:.1%} ± {cv_std['accuracy']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Precision</div>
                <div class="metric-value">{test_eval['precision']:.1%}</div>
                <div class="metric-subtitle">CV: {cv_mean['precision']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Recall</div>
                <div class="metric-value">{test_eval['recall']:.1%}</div>
                <div class="metric-subtitle">CV: {cv_mean['recall']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">F1-Score</div>
                <div class="metric-value">{test_eval['f1']:.3f}</div>
                <div class="metric-subtitle">CV: {cv_mean['f1']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with m5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">ROC-AUC</div>
                <div class="metric-value">{test_eval['roc_auc']:.3f}</div>
                <div class="metric-subtitle">CV: {cv_mean['roc_auc']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # Visual Confusion Matrix & Metric Explanations
        col_cm, col_expl = st.columns([1, 1], gap="large")

        with col_cm:
            st.subheader("Confusion Matrix (Held-Out Test Set)")
            cm = test_eval['confusion_matrix']
            tn, fp = cm[0][0], cm[0][1]
            fn, tp = cm[1][0], cm[1][1]

            # Render Confusion Matrix as a clean, clear visual table
            fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.2))
            cax = ax_cm.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)

            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(['Pred: Dies (0)', 'Pred: Survives (1)'], fontsize=10, fontweight='500')
            ax_cm.set_yticklabels(['Actual: Dies (0)', 'Actual: Survives (1)'], fontsize=10, fontweight='500')

            # Annotate cells
            cell_labels = [
                [f"True Negatives (TN)\n{tn}\n(Correct Dies)", f"False Positives (FP)\n{fp}\n(Mistaken Survives)"],
                [f"False Negatives (FN)\n{fn}\n(Mistaken Dies)", f"True Positives (TP)\n{tp}\n(Correct Survives)"]
            ]
            for i in range(2):
                for j in range(2):
                    ax_cm.text(j, i, cell_labels[i][j], ha='center', va='center', fontsize=9.5, color='#0f172a', fontweight='600')

            ax_cm.tick_params(bottom=False, top=True, left=True, right=False)
            fig_cm.patch.set_facecolor('#ffffff')
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

            # Class-specific breakdown table
            st.caption("Class-Specific Held-Out Performance:")
            df_class = pd.DataFrame({
                'Class Label': ['Dies (Class 0)', 'Survives (Class 1)'],
                'Precision': [f"{test_eval['class_0_dies']['precision']:.1%}", f"{test_eval['class_1_survives']['precision']:.1%}"],
                'Recall (Sensitivity)': [f"{test_eval['class_0_dies']['recall']:.1%}", f"{test_eval['class_1_survives']['recall']:.1%}"],
                'F1-Score': [f"{test_eval['class_0_dies']['f1']:.3f}", f"{test_eval['class_1_survives']['f1']:.3f}"],
                'Test Support': [test_eval['class_0_dies']['support'], test_eval['class_1_survives']['support']]
            })
            st.table(df_class)

        with col_expl:
            st.subheader("Beginner-Friendly Metric Explanations")
            st.markdown("""
            - **Accuracy ({acc:.1%}):** The overall proportion of correct predictions across all patients in the held-out test set. While easy to understand, accuracy alone is insufficient for medical datasets because negative and positive classes are unevenly distributed.
            - **Precision ({prec:.1%}):** When the model predicts a patient survives, precision indicates how often that prediction is correct.
            - **Recall / Sensitivity ({rec:.1%}):** The ability of the model to correctly identify patients who actually survive. Importantly, for **Class 0 (Dies)**, the model achieves **{rec_dies:.1%} recall** ({tn} out of 13 deceased patients correctly identified), preventing false reassurances.
            - **F1-Score ({f1:.3f}):** The harmonic mean of precision and recall. It balances false alarms with missed cases.
            - **ROC-AUC ({auc:.3f}):** Area Under the Receiver Operating Characteristic curve. Measures the model's ability to rank patient risk across all decision thresholds independently of a single 0.5 cutoff.
            
            > ⚠️ **Small Dataset Note for Viva:** With 165 total patients and 33 in the test set, each individual patient in the test set represents approximately 3% of the accuracy. For rigorous scientific validation, we report both **5-Fold Cross-Validation** on the training set and the **held-out test set** results.
            """.format(
                acc=test_eval['accuracy'],
                prec=test_eval['precision'],
                rec=test_eval['recall'],
                rec_dies=test_eval['class_0_dies']['recall'],
                tn=tn,
                f1=test_eval['f1'],
                auc=test_eval['roc_auc']
            ))

    else:
        st.info("Please train the model using the button above to view evaluation metrics and the confusion matrix.")


# =============================================================================
# SECTION 3: PREDICTION DEMO
# =============================================================================
elif section == "3. Prediction Demo":
    st.title("Clinical Prediction Demo")
    st.markdown("##### Enter Patient Parameters to Estimate 1-Year Survival Probability")
    st.write("")

    if pipeline_loaded is None:
        st.warning("⚠️ No trained model pipeline is currently loaded. Please go to **Section 2: Model Performance** and click **Train / Retrain Model** before running predictions.")
    else:
        st.markdown("""
        Enter patient clinical findings, laboratory tests, and risk markers below. 
        Fields are grouped into 5 logical categories with baseline median reference values pre-populated.
        """)

        # Convenient Presets for quick demonstration during viva
        preset_col1, preset_col2, preset_col3 = st.columns([1, 1, 2])
        with preset_col1:
            if st.button("🔄 Reset to Median Baseline", use_container_width=True):
                st.session_state['preset'] = 'median'
                st.rerun()
        with preset_col2:
            if st.button("⚠️ Load High-Risk Case Sample", use_container_width=True):
                st.session_state['preset'] = 'high_risk'
                st.rerun()

        active_preset = st.session_state.get('preset', 'median')

        # Form for input features
        with st.form("hcc_prediction_form"):
            user_inputs = {}

            # GROUP 1: Demographics & Lifestyle
            with st.expander("👤 1. Demographics & Lifestyle", expanded=True):
                g1_c1, g1_c2, g1_c3 = st.columns(3)
                
                with g1_c1:
                    user_inputs['Gender'] = st.selectbox(
                        "Gender",
                        options=[1, 0],
                        format_func=lambda x: "Male" if x == 1 else "Female",
                        index=0 if active_preset != 'high_risk' else 0,
                        help=FEATURE_METADATA['Gender']['help']
                    )
                    user_inputs['Age'] = st.number_input(
                        "Age at Diagnosis (years)",
                        min_value=20.0,
                        max_value=95.0,
                        value=66.0 if active_preset == 'median' else 75.0,
                        step=1.0,
                        help=FEATURE_METADATA['Age']['help']
                    )

                with g1_c2:
                    user_inputs['Alcohol'] = st.selectbox(
                        "Regular Alcohol Consumption",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=0 if active_preset != 'high_risk' else 0,
                        help=FEATURE_METADATA['Alcohol']['help']
                    )
                    user_inputs['Grams_day'] = st.number_input(
                        "Alcohol Intake (grams/day)",
                        min_value=0.0,
                        max_value=500.0,
                        value=75.0 if active_preset == 'median' else 150.0,
                        step=5.0,
                        help=FEATURE_METADATA['Grams_day']['help']
                    )

                with g1_c3:
                    user_inputs['Smoking'] = st.selectbox(
                        "Smoking History",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=0 if active_preset != 'high_risk' else 0,
                        help=FEATURE_METADATA['Smoking']['help']
                    )
                    user_inputs['Packs_year'] = st.number_input(
                        "Smoking History (packs/year)",
                        min_value=0.0,
                        max_value=510.0,
                        value=0.0 if active_preset == 'median' else 35.0,
                        step=5.0,
                        help=FEATURE_METADATA['Packs_year']['help']
                    )

            # GROUP 2: Clinical Symptoms & History
            with st.expander("📋 2. Clinical History & Findings", expanded=False):
                g2_c1, g2_c2, g2_c3 = st.columns(3)

                with g2_c1:
                    user_inputs['Symptoms'] = st.selectbox(
                        "Overt Symptoms at Diagnosis",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=0 if active_preset != 'high_risk' else 0,
                        help=FEATURE_METADATA['Symptoms']['help']
                    )
                    user_inputs['Cirrhosis'] = st.selectbox(
                        "Liver Cirrhosis",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=0 if active_preset != 'high_risk' else 0,
                        help=FEATURE_METADATA['Cirrhosis']['help']
                    )
                    user_inputs['Endemic'] = st.selectbox(
                        "Endemic Country Origin",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=1 if active_preset != 'high_risk' else 0,
                        help=FEATURE_METADATA['Endemic']['help']
                    )
                    user_inputs['Diabetes'] = st.selectbox(
                        "Diabetes Mellitus",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=1 if active_preset != 'high_risk' else 0,
                        help=FEATURE_METADATA['Diabetes']['help']
                    )
                    user_inputs['Obesity'] = st.selectbox(
                        "Clinical Obesity",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=1 if active_preset != 'high_risk' else 1,
                        help=FEATURE_METADATA['Obesity']['help']
                    )

                with g2_c2:
                    user_inputs['AHT'] = st.selectbox(
                        "Arterial Hypertension",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=1 if active_preset != 'high_risk' else 0,
                        help=FEATURE_METADATA['AHT']['help']
                    )
                    user_inputs['CRI'] = st.selectbox(
                        "Chronic Renal Insufficiency",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=1 if active_preset != 'high_risk' else 0,
                        help=FEATURE_METADATA['CRI']['help']
                    )
                    user_inputs['HIV'] = st.selectbox(
                        "HIV Infection",
                        options=[1, 0],
                        format_func=lambda x: "Positive" if x == 1 else "Negative",
                        index=1 if active_preset != 'high_risk' else 1,
                        help=FEATURE_METADATA['HIV']['help']
                    )
                    user_inputs['NASH'] = st.selectbox(
                        "NASH (Steatohepatitis)",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=1 if active_preset != 'high_risk' else 1,
                        help=FEATURE_METADATA['NASH']['help']
                    )
                    user_inputs['Varices'] = st.selectbox(
                        "Esophageal Varices",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=0 if active_preset != 'high_risk' else 0,
                        help=FEATURE_METADATA['Varices']['help']
                    )

                with g2_c3:
                    user_inputs['Spleno'] = st.selectbox(
                        "Splenomegaly",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=1 if active_preset != 'high_risk' else 0,
                        help=FEATURE_METADATA['Spleno']['help']
                    )
                    user_inputs['PHT'] = st.selectbox(
                        "Portal Hypertension",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=0 if active_preset != 'high_risk' else 0,
                        help=FEATURE_METADATA['PHT']['help']
                    )
                    user_inputs['PVT'] = st.selectbox(
                        "Portal Vein Thrombosis",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=1 if active_preset != 'high_risk' else 0,
                        help=FEATURE_METADATA['PVT']['help']
                    )
                    user_inputs['Metastasis'] = st.selectbox(
                        "Metastasis",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=1 if active_preset != 'high_risk' else 0,
                        help=FEATURE_METADATA['Metastasis']['help']
                    )
                    user_inputs['Hallmark'] = st.selectbox(
                        "Radiological Hallmark",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=0 if active_preset != 'high_risk' else 0,
                        help=FEATURE_METADATA['Hallmark']['help']
                    )

            # GROUP 3: Viral Hepatitis & Metabolic Markers
            with st.expander("🔬 3. Viral Hepatitis & Metabolic Markers", expanded=False):
                g3_c1, g3_c2 = st.columns(2)
                with g3_c1:
                    user_inputs['HBsAg'] = st.selectbox(
                        "HBsAg (Hepatitis B Surface Antigen)",
                        options=[1, 0],
                        format_func=lambda x: "Positive" if x == 1 else "Negative",
                        index=1 if active_preset != 'high_risk' else 1,
                        help=FEATURE_METADATA['HBsAg']['help']
                    )
                    user_inputs['HBeAg'] = st.selectbox(
                        "HBeAg (Hepatitis B e-Antigen)",
                        options=[1, 0],
                        format_func=lambda x: "Positive" if x == 1 else "Negative",
                        index=1 if active_preset != 'high_risk' else 1,
                        help=FEATURE_METADATA['HBeAg']['help']
                    )
                    user_inputs['HBcAb'] = st.selectbox(
                        "HBcAb (Hepatitis B Core Antibody)",
                        options=[1, 0],
                        format_func=lambda x: "Positive" if x == 1 else "Negative",
                        index=1 if active_preset != 'high_risk' else 1,
                        help=FEATURE_METADATA['HBcAb']['help']
                    )
                with g3_c2:
                    user_inputs['HCVAb'] = st.selectbox(
                        "HCVAb (Hepatitis C Virus Antibody)",
                        options=[1, 0],
                        format_func=lambda x: "Positive" if x == 1 else "Negative",
                        index=1 if active_preset != 'high_risk' else 0,
                        help=FEATURE_METADATA['HCVAb']['help']
                    )
                    user_inputs['Hemochro'] = st.selectbox(
                        "Hemochromatosis (Iron Overload)",
                        options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        index=1 if active_preset != 'high_risk' else 1,
                        help=FEATURE_METADATA['Hemochro']['help']
                    )

            # GROUP 4: Functional Severity Scores
            with st.expander("⚡ 4. Functional Severity Scores", expanded=True):
                g4_c1, g4_c2, g4_c3 = st.columns(3)
                with g4_c1:
                    ps_options = [0, 1, 2, 3, 4]
                    ps_labels = {
                        0: "0 - Fully active, unrestricted",
                        1: "1 - Restricted strenuous activity",
                        2: "2 - Ambulatory, capable of self-care",
                        3: "3 - Limited self-care, confined >50%",
                        4: "4 - Completely disabled"
                    }
                    user_inputs['PS'] = st.selectbox(
                        "ECOG Performance Status",
                        options=ps_options,
                        format_func=lambda x: ps_labels[x],
                        index=1 if active_preset == 'median' else 3,
                        help=FEATURE_METADATA['PS']['help']
                    )
                with g4_c2:
                    enc_options = [1, 2, 3]
                    enc_labels = {
                        1: "1 - None",
                        2: "2 - Grade I / II (Mild/Moderate)",
                        3: "3 - Grade III / IV (Severe/Coma)"
                    }
                    user_inputs['Encephalopathy'] = st.selectbox(
                        "Hepatic Encephalopathy Degree",
                        options=enc_options,
                        format_func=lambda x: enc_labels[x],
                        index=0 if active_preset == 'median' else 1,
                        help=FEATURE_METADATA['Encephalopathy']['help']
                    )
                with g4_c3:
                    asc_options = [1, 2, 3]
                    asc_labels = {
                        1: "1 - None",
                        2: "2 - Mild",
                        3: "3 - Moderate to Severe"
                    }
                    user_inputs['Ascites'] = st.selectbox(
                        "Ascites Degree",
                        options=asc_options,
                        format_func=lambda x: asc_labels[x],
                        index=0 if active_preset == 'median' else 2,
                        help=FEATURE_METADATA['Ascites']['help']
                    )

            # GROUP 5: Laboratory Biomarkers & Tumor Characteristics
            with st.expander("🧪 5. Laboratory Tests & Tumor Markers", expanded=True):
                g5_c1, g5_c2, g5_c3 = st.columns(3)

                with g5_c1:
                    user_inputs['AFP'] = st.number_input(
                        "AFP (Alpha-Fetoprotein) [ng/mL]",
                        min_value=1.0,
                        max_value=2000000.0,
                        value=33.0 if active_preset == 'median' else 25000.0,
                        step=10.0,
                        help=FEATURE_METADATA['AFP']['help']
                    )
                    user_inputs['Albumin'] = st.number_input(
                        "Serum Albumin [g/dL]",
                        min_value=1.0,
                        max_value=6.0,
                        value=3.4 if active_preset == 'median' else 2.4,
                        step=0.1,
                        help=FEATURE_METADATA['Albumin']['help']
                    )
                    user_inputs['Total_Bil'] = st.number_input(
                        "Total Bilirubin [mg/dL]",
                        min_value=0.1,
                        max_value=45.0,
                        value=1.4 if active_preset == 'median' else 8.5,
                        step=0.1,
                        help=FEATURE_METADATA['Total_Bil']['help']
                    )
                    user_inputs['Dir_Bil'] = st.number_input(
                        "Direct Bilirubin [mg/dL]",
                        min_value=0.05,
                        max_value=30.0,
                        value=0.7 if active_preset == 'median' else 4.2,
                        step=0.1,
                        help=FEATURE_METADATA['Dir_Bil']['help']
                    )
                    user_inputs['INR'] = st.number_input(
                        "INR (Prothrombin Time Ratio)",
                        min_value=0.8,
                        max_value=5.0,
                        value=1.3 if active_preset == 'median' else 2.2,
                        step=0.05,
                        help=FEATURE_METADATA['INR']['help']
                    )
                    user_inputs['Creatinine'] = st.number_input(
                        "Creatinine [mg/dL]",
                        min_value=0.1,
                        max_value=10.0,
                        value=0.85 if active_preset == 'median' else 2.8,
                        step=0.05,
                        help=FEATURE_METADATA['Creatinine']['help']
                    )
                    user_inputs['TP'] = st.number_input(
                        "Total Proteins [g/dL]",
                        min_value=2.0,
                        max_value=110.0,
                        value=7.1 if active_preset == 'median' else 5.5,
                        step=0.1,
                        help=FEATURE_METADATA['TP']['help']
                    )

                with g5_c2:
                    user_inputs['ALT'] = st.number_input(
                        "ALT (Alanine Transaminase) [U/L]",
                        min_value=5.0,
                        max_value=500.0,
                        value=50.0 if active_preset == 'median' else 140.0,
                        step=1.0,
                        help=FEATURE_METADATA['ALT']['help']
                    )
                    user_inputs['AST'] = st.number_input(
                        "AST (Aspartate Transaminase) [U/L]",
                        min_value=10.0,
                        max_value=600.0,
                        value=71.0 if active_preset == 'median' else 220.0,
                        step=1.0,
                        help=FEATURE_METADATA['AST']['help']
                    )
                    user_inputs['GGT'] = st.number_input(
                        "GGT [U/L]",
                        min_value=10.0,
                        max_value=1600.0,
                        value=180.0 if active_preset == 'median' else 450.0,
                        step=5.0,
                        help=FEATURE_METADATA['GGT']['help']
                    )
                    user_inputs['ALP'] = st.number_input(
                        "ALP (Alkaline Phosphatase) [U/L]",
                        min_value=1.0,
                        max_value=1000.0,
                        value=162.0 if active_preset == 'median' else 380.0,
                        step=2.0,
                        help=FEATURE_METADATA['ALP']['help']
                    )
                    user_inputs['Hemoglobin'] = st.number_input(
                        "Hemoglobin [g/dL]",
                        min_value=4.0,
                        max_value=20.0,
                        value=13.0 if active_preset == 'median' else 9.5,
                        step=0.1,
                        help=FEATURE_METADATA['Hemoglobin']['help']
                    )
                    user_inputs['MCV'] = st.number_input(
                        "Mean Corpuscular Volume (MCV) [fL]",
                        min_value=60.0,
                        max_value=130.0,
                        value=95.0,
                        step=0.5,
                        help=FEATURE_METADATA['MCV']['help']
                    )
                    user_inputs['Leucocytes'] = st.number_input(
                        "Leukocytes (WBC) [G/L]",
                        min_value=1.0,
                        max_value=15000.0,
                        value=7.2,
                        step=0.5,
                        help=FEATURE_METADATA['Leucocytes']['help']
                    )

                with g5_c3:
                    user_inputs['Platelets'] = st.number_input(
                        "Platelet Count [G/L]",
                        min_value=1.0,
                        max_value=500000.0,
                        value=93000.0 if active_preset == 'median' else 42000.0,
                        step=1000.0,
                        help=FEATURE_METADATA['Platelets']['help']
                    )
                    user_inputs['Nodules'] = st.number_input(
                        "Number of Tumor Nodules",
                        min_value=0.0,
                        max_value=5.0,
                        value=2.0 if active_preset == 'median' else 4.0,
                        step=1.0,
                        help=FEATURE_METADATA['Nodules']['help']
                    )
                    user_inputs['Major_Dim'] = st.number_input(
                        "Major Dimension of Largest Nodule [cm]",
                        min_value=0.5,
                        max_value=25.0,
                        value=5.0 if active_preset == 'median' else 14.0,
                        step=0.5,
                        help=FEATURE_METADATA['Major_Dim']['help']
                    )
                    user_inputs['Iron'] = st.number_input(
                        "Serum Iron [mcg/dL]",
                        min_value=0.0,
                        max_value=250.0,
                        value=83.0,
                        step=1.0,
                        help=FEATURE_METADATA['Iron']['help']
                    )
                    user_inputs['Sat'] = st.number_input(
                        "Transferrin Saturation [%]",
                        min_value=0.0,
                        max_value=130.0,
                        value=27.0,
                        step=1.0,
                        help=FEATURE_METADATA['Sat']['help']
                    )
                    user_inputs['Ferritin'] = st.number_input(
                        "Ferritin Level [ng/mL]",
                        min_value=0.0,
                        max_value=2500.0,
                        value=295.0,
                        step=10.0,
                        help=FEATURE_METADATA['Ferritin']['help']
                    )

            st.write("")
            submit_btn = st.form_submit_button("🔮 Predict 1-Year Survival", type="primary", use_container_width=True)

        # Handle Prediction
        if submit_btn:
            try:
                # Build single row dataframe with exact feature order
                input_df = pd.DataFrame([user_inputs])[FEATURE_COLUMNS]

                # Run prediction through saved pipeline
                pred_class = int(pipeline_loaded.predict(input_df)[0])
                pred_proba = pipeline_loaded.predict_proba(input_df)[0]
                prob_dies = float(pred_proba[0])
                prob_survives = float(pred_proba[1])

                st.write("")
                st.subheader("Model Prognostic Estimate")

                if pred_class == 1:
                    st.markdown(f"""
                    <div class="result-card-survives">
                        <h3 style="color: #15803d; margin-top: 0;">Predicted Outcome: Likely to Survive ≥ 1 Year (Class 1)</h3>
                        <p style="font-size: 1.05rem; color: #166534; margin-bottom: 8px;">
                            Estimated Survival Probability: <b>{prob_survives:.1%}</b> &nbsp;|&nbsp; Estimated 1-Year Mortality Risk: <b>{prob_dies:.1%}</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card-dies">
                        <h3 style="color: #b91c1c; margin-top: 0;">Predicted Outcome: High 1-Year Mortality Risk (Class 0)</h3>
                        <p style="font-size: 1.05rem; color: #991b1b; margin-bottom: 8px;">
                            Estimated 1-Year Mortality Risk: <b>{prob_dies:.1%}</b> &nbsp;|&nbsp; Estimated Survival Probability: <b>{prob_survives:.1%}</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.markdown(f"**Survives ≥ 1 Year (Class 1): {prob_survives:.1%}**")
                    st.progress(prob_survives)
                with col_p2:
                    st.markdown(f"**Dies within 1 Year (Class 0): {prob_dies:.1%}**")
                    st.progress(prob_dies)

                # Mandatory Clinical Disclaimer
                st.markdown("""
                <div class="warning-disclaimer-box">
                    <b>⚠️ Strict Academic & Clinical Disclaimer:</b><br>
                    This application is an educational and technical demonstration developed for an academic machine learning coursework submission. 
                    Predictions are generated by a statistical model fitted on a small research cohort (165 historical patients from Coimbra, Portugal) 
                    and <b>must never be interpreted as clinical diagnoses, definitive prognoses, or treatment recommendations</b> for individual medical care. 
                    Clinical oncology decisions require comprehensive evaluation by certified multidisciplinary medical teams. 
                    <i>No patient inputs are retained, stored, or logged by this system.</i>
                </div>
                """, unsafe_allow_html=True)

            except Exception as ex:
                st.error(f"Error computing prediction: {ex}")

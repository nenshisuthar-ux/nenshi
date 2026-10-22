# 🩺 Hepatocellular Carcinoma (HCC) Survival Prediction

A beginner-friendly, end-to-end Machine Learning web application designed to predict 1-year survival in Hepatocellular Carcinoma (liver cancer) patients using clinical, demographic, and laboratory biomarkers.

Built with **Python**, **Scikit-learn**, **Pandas**, and **Streamlit**.

---

## 📌 1. Project Overview & Problem Statement

**Hepatocellular Carcinoma (HCC)** is the most frequent primary malignancy of the liver and one of the leading causes of cancer mortality worldwide. Staging and prognosis for HCC patients depend on complex interactions between patient performance status, underlying liver cirrhosis severity, and tumor burden.

### Objective
To build a functional, leak-free, reproducible machine learning classification pipeline and interactive web interface that:
1. Ingests real patient clinical data based on **EASL-EORTC (European Association for the Study of the Liver - European Organisation for Research and Treatment of Cancer)** clinical guidelines.
2. Cleans and handles missing attributes through statistical imputation without discarding records.
3. Fits interpretable baseline classifiers with class balancing.
4. Evaluates performance using rigorous cross-validation and held-out testing.
5. Provides a clean, medical-grade web interface for real-time risk stratification demonstrations.

---

## 📊 2. Dataset Description & Target Variable

The dataset was curated at the Hospital and University Centre of Coimbra, Portugal.

- **Total Patient Records (Instances):** 165
- **Input Features:** 49 clinical attributes
  - **23 Nominal / Binary Features** (e.g., Symptoms, Cirrhosis, Alcohol, Hepatitis markers)
  - **3 Ordinal Features** (ECOG Performance Status [0–4], Encephalopathy [1–3], Ascites [1–3])
  - **23 Continuous / Numerical Features** (e.g., Age, Bilirubin, Albumin, AFP, INR, Platelets, Creatinine)
- **Target Variable (`Class`):** Binary survival at 1 year post-diagnosis:
  - `1` = **Survives** (102 patients, ~61.8%)
  - `0` = **Dies** (63 patients, ~38.2%)
- **Missing Values:** Documented by `?` in the raw data, accounting for **10.22%** of the entire dataset. Only 8 patients have complete observations across all 49 fields, making statistical imputation mandatory.

### Dataset Citation
> Santos, M. S., Abreu, P. H., García-Laencina, P. J., Simão, A., & Carvalho, A. (2015).  
> *A new cluster-based oversampling method for improving survival prediction of hepatocellular carcinoma patients.*  
> Journal of Biomedical Informatics, 58, 49-59.

---

## ⚙️ 3. Technologies Used

- **Language:** Python (3.10+)
- **Data Wrangling:** `pandas`, `numpy`
- **Machine Learning & Preprocessing:** `scikit-learn` (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `LogisticRegression`, `RandomForestClassifier`)
- **Visualizations:** `matplotlib`
- **Model Persistence:** `joblib`
- **Web Interface:** `streamlit`

---

## 🔬 4. Machine Learning Pipeline & Leak-Free Preprocessing

```
[Raw Dataset (165 x 50)]
         │
         ▼
[Convert '?' to NaN & Separate Target 'Class']
         │
         ├────────────────────────────────────────┐
         │ (80% Train, N=132)                     │ (20% Held-Out Test, N=33)
         ▼                                        ▼
[ColumnTransformer (Fitted on Train Only)]   [Transformed using Train Statistics]
  ├── Continuous (23): Median Imputer + Scaler
  └── Categorical (26): Most-Frequent Imputer
         │
         ▼
[Stratified 5-Fold Cross Validation]
  ├── Logistic Regression (Clinical Baseline, Balanced Weights)
  └── Random Forest Classifier (Depth-constrained)
         │
         ▼
[Final Model Fit on Full Training Split]
         │
         ▼
[Evaluation on Held-Out Test Set (Unseen N=33)]
         │
         ▼
[Saved Artifacts: models/hcc_pipeline.joblib & metrics.json]
```

### Key Preprocessing Principles:
1. **No Data Leakage:** Preprocessing transformations (`fit_transform`) are strictly computed on training data (`X_train`) only. Test samples and inference inputs are strictly transformed using training statistics.
2. **Target Isolation:** The target column (`Class`) is separated before any pipeline transformations occur.
3. **Missing Value Strategy:**
   - Numerical continuous features are imputed with the **median** (robust to skewed outliers in lab values like AFP).
   - Categorical/binary/ordinal features are imputed with the **mode (most frequent)** category.
4. **Class Imbalance Handling:** `class_weight='balanced'` penalizes misclassification of the minority mortality class (Class 0: 38.2%) inversely proportional to class frequencies.

---

## 📈 5. Model Evaluation & Real Results

Evaluated on an independent **held-out test split (20%, N=33 patients, stratified)**:

| Metric | Logistic Regression (Baseline) | Random Forest | Description |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **78.8%** | 75.8% | Overall proportion of correct predictions |
| **Precision (Class 1)** | **84.2%** | 83.3% | Accuracy when predicting survival |
| **Recall / Sensitivity (Class 1)** | **80.0%** | 75.0% | Proportion of actual survivors detected |
| **F1-Score (Class 1)** | **0.821** | 0.790 | Harmonic balance of precision and recall |
| **ROC-AUC** | **0.877** | 0.869 | Discriminative ability across all risk thresholds |
| **Mortality Recall (Class 0)** | **76.9%** (10/13) | **76.9%** (10/13) | Detection of high-risk patients who passed away |
| **5-Fold CV Accuracy (Train)** | 67.3% ± 8.6% | 72.6% ± 7.7% | Mean cross-validation performance on training folds |
| **5-Fold CV ROC-AUC (Train)** | 75.5% ± 10.2% | 79.5% ± 6.9% | Mean CV ROC-AUC on training folds |

### Held-Out Test Set Confusion Matrix (Logistic Regression, N=33):
```
                       Predicted: Dies (0)   Predicted: Survives (1)
 Actual: Dies (0)            10 (TN)                 3 (FP)
 Actual: Survives (1)         4 (FN)                16 (TP)
```
- **True Negatives (10):** Correctly identified 10 out of 13 deceased patients.
- **True Positives (16):** Correctly identified 16 out of 20 surviving patients.

---

## 📁 6. Project Structure

```
hcc-survival/
├── app.py                     # Streamlit multi-section interactive web application
├── train.py                   # Data ingestion, preprocessing, cross-validation & training
├── test_app.py                # Automated quality assurance test suite (5 verification checks)
├── requirements.txt           # Minimal, verified production dependencies
├── README.md                  # Comprehensive documentation and viva preparation guide
├── .gitignore                 # Excludes cache, virtual environments, and temporary files
├── .streamlit/
│   └── config.toml            # Clean theme configuration (light mode, navy text, blue accents)
├── data/
│   ├── README.md              # Feature dictionary and dataset documentation
│   ├── hcc-data.csv           # Cleaned dataset with full column headers
│   ├── hcc-data.txt           # Original raw dataset
│   └── hcc-description.txt    # Original clinical documentation
└── models/
    ├── hcc_pipeline.joblib    # Serialized fitted Scikit-learn Pipeline
    └── metrics.json           # Machine-readable performance metrics & metadata
```

---

## 💻 7. Setup & Run Instructions

### Prerequisites
Make sure Python 3.10, 3.11, or compatible is installed.

### Step 1: Clone or Navigate to Project
```bash
cd "hcc-survival"
```

### Step 2: Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts scripts, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

**On macOS / Linux (Terminal):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Machine Learning Training
```bash
python train.py
```
This trains the pipeline, performs 5-fold cross-validation, evaluates on the test set, and updates `models/hcc_pipeline.joblib` and `models/metrics.json`.

### Step 5: Run Automated Tests
```bash
python test_app.py
```

### Step 6: Launch Streamlit Web Application
```bash
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501`.

---

## 🖥️ 8. Web Application Features

The UI is built with a minimalist, professional healthcare aesthetic (light grey background, dark navy typography, subtle blue/teal accents).

### Section 1: Overview
- Complete project background on Hepatocellular Carcinoma.
- Real dataset metrics cards: Total Records (165), Input Features (49), Target Variable (`Class`), and Missing Rate (10.22%).
- Dynamic, annotated class distribution chart.
- Visual breakdown of the 4-step leak-free machine learning workflow.

### Section 2: Model Performance
- Interactive model selector: **Logistic Regression** (baseline) vs **Random Forest**.
- Live **Train / Retrain Model** button with training status indicators.
- Real held-out test set metrics cards (Accuracy, Precision, Recall, F1, ROC-AUC).
- Visual Confusion Matrix heatmap and class-specific breakdown table.
- Beginner-friendly explanations of each metric tailored for viva examinations.

### Section 3: Prediction Demo
- User input form grouped into 5 logical clinical categories:
  1. *Demographics & Lifestyle* (Age, Gender, Alcohol intake, Smoking packs/year)
  2. *Clinical Findings* (Symptoms, Cirrhosis, Varices, Portal Hypertension, Metastasis, etc.)
  3. *Viral Hepatitis & Metabolic Markers* (HBsAg, HBeAg, HBcAb, HCVAb, Hemochromatosis)
  4. *Functional Severity Scores* (ECOG Performance Status, Encephalopathy degree, Ascites)
  5. *Laboratory Tests & Tumor Markers* (AFP, Albumin, Bilirubin, INR, Platelets, Creatinine, Nodules, Major Dimension)
- Preset buttons: **Reset to Median Baseline** or **Load High-Risk Case Sample**.
- Clear prognostic outcome cards with probability bars for both survival and 1-year mortality.
- Complete input validation and clinical research disclaimer.
- Strict privacy: No user inputs or predictions are stored or logged.

---

## 🎓 9. Viva Questions & Project Explanations (College Defense Guide)

Here are direct questions examiners often ask and how to answer them:

### Q1: Why did you choose Logistic Regression as the baseline model?
> **Answer:** In clinical predictive modeling, interpretability and calibration are paramount. Logistic Regression provides direct, well-calibrated class probabilities and linear log-odds coefficients that clinical researchers can inspect. Given the dataset size (165 patients), complex black-box models are prone to overfitting, making regularized Logistic Regression a sound benchmark.

### Q2: How did you prevent Data Leakage?
> **Answer:** We constructed an end-to-end `Pipeline` and `ColumnTransformer`. The median for numerical imputation, mode for categorical imputation, and the mean/standard deviation for `StandardScaler` are fitted strictly on `X_train` inside each cross-validation fold. The held-out test split was transformed using training statistics and was never seen during fitting. Furthermore, the target variable (`Class`) was detached before preprocessing.

### Q3: Why is Accuracy alone not enough for this dataset?
> **Answer:** The dataset is imbalanced (61.8% survivors vs 38.2% deaths). A naive model predicting everyone survives would achieve ~61.8% accuracy but fail to identify any patient who dies. In medical diagnosis, high **Recall on the mortality class (Class 0)** is critical so patients at high risk are not missed. That is why we also evaluate Precision, Recall, F1-Score, and ROC-AUC with `class_weight='balanced'`.

### Q4: Why did you use Stratified K-Fold Cross-Validation?
> **Answer:** Stratified sampling ensures every fold maintains the exact same 62:38 class ratio as the full dataset. With only 165 records, standard random splits could yield folds with too few death events, leading to unstable metric estimations.

### Q5: Why did you impute missing values instead of dropping rows?
> **Answer:** Overall missing data is 10.22%, but only 8 patients have complete records across all 49 attributes. Dropping incomplete rows would delete over 95% of the data. Imputing numerical features with the median and categorical features with the mode preserves all 165 clinical cases while preventing outlier distortion.

---

## ⚠️ 10. Known Limitations & Medical Disclaimer

1. **Small Sample Size:** With 165 historical patients from a single university hospital center in Portugal, results have statistical confidence intervals and may not generalize across different global demographic populations.
2. **Class Imbalance & High Dimensionality:** 49 features for 165 samples means the feature-to-sample ratio is relatively high (~1:3.3). Feature selection or dimensionality reduction could be explored in future work.
3. **Clinical Disclaimer:** This software is strictly an academic demonstration and engineering project. It does not provide medical advice or diagnosis. No medical decisions should ever be made based on these predictions.

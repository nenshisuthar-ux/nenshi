"""
HCC Survival Prediction - Training and Evaluation Pipeline
Author: HCC ML Project Team
Dataset: Coimbra HCC Survival Dataset (165 patients, 49 features, 1 target)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# All 50 column names in the exact order of the dataset
FEATURE_COLUMNS = [
    'Gender', 'Symptoms', 'Alcohol', 'HBsAg', 'HBeAg', 'HBcAb', 'HCVAb', 'Cirrhosis',
    'Endemic', 'Smoking', 'Diabetes', 'Obesity', 'Hemochro', 'AHT', 'CRI', 'HIV',
    'NASH', 'Varices', 'Spleno', 'PHT', 'PVT', 'Metastasis', 'Hallmark', 'Age',
    'Grams_day', 'Packs_year', 'PS', 'Encephalopathy', 'Ascites', 'INR', 'AFP',
    'Hemoglobin', 'MCV', 'Leucocytes', 'Platelets', 'Albumin', 'Total_Bil', 'ALT',
    'AST', 'GGT', 'ALP', 'TP', 'Creatinine', 'Nodules', 'Major_Dim', 'Dir_Bil',
    'Iron', 'Sat', 'Ferritin'
]

TARGET_COLUMN = 'Class'
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# 23 Binary + 3 Ordinal = 26 Categorical features
CATEGORICAL_COLS = [
    'Gender', 'Symptoms', 'Alcohol', 'HBsAg', 'HBeAg', 'HBcAb', 'HCVAb', 'Cirrhosis',
    'Endemic', 'Smoking', 'Diabetes', 'Obesity', 'Hemochro', 'AHT', 'CRI', 'HIV',
    'NASH', 'Varices', 'Spleno', 'PHT', 'PVT', 'Metastasis', 'Hallmark',
    'PS', 'Encephalopathy', 'Ascites'
]

# 23 Continuous/Quantitative features
NUMERICAL_COLS = [
    'Age', 'Grams_day', 'Packs_year', 'INR', 'AFP', 'Hemoglobin', 'MCV',
    'Leucocytes', 'Platelets', 'Albumin', 'Total_Bil', 'ALT', 'AST', 'GGT',
    'ALP', 'TP', 'Creatinine', 'Nodules', 'Major_Dim', 'Dir_Bil', 'Iron',
    'Sat', 'Ferritin'
]

# Feature metadata for user interface, validation, and documentation
FEATURE_METADATA = {
    # Group 1: Demographics & Lifestyle
    'Gender': {
        'label': 'Gender',
        'type': 'binary',
        'group': 'Demographics & Lifestyle',
        'options': {1: 'Male', 0: 'Female'},
        'default': 1,
        'help': 'Biological sex of the patient (1=Male, 0=Female).'
    },
    'Age': {
        'label': 'Age at Diagnosis',
        'type': 'numerical',
        'group': 'Demographics & Lifestyle',
        'unit': 'years',
        'min': 20.0,
        'max': 95.0,
        'default': 66.0,
        'step': 1.0,
        'help': 'Patient age in years (dataset range: 20-93).'
    },
    'Alcohol': {
        'label': 'Alcohol Consumption History',
        'type': 'binary',
        'group': 'Demographics & Lifestyle',
        'options': {1: 'Yes', 0: 'No'},
        'default': 1,
        'help': 'History of regular alcohol intake.'
    },
    'Grams_day': {
        'label': 'Alcohol Intake',
        'type': 'numerical',
        'group': 'Demographics & Lifestyle',
        'unit': 'grams/day',
        'min': 0.0,
        'max': 500.0,
        'default': 75.0,
        'step': 5.0,
        'help': 'Average grams of alcohol consumed per day.'
    },
    'Smoking': {
        'label': 'Smoking History',
        'type': 'binary',
        'group': 'Demographics & Lifestyle',
        'options': {1: 'Yes', 0: 'No'},
        'default': 1,
        'help': 'History of cigarette smoking.'
    },
    'Packs_year': {
        'label': 'Smoking Intensity',
        'type': 'numerical',
        'group': 'Demographics & Lifestyle',
        'unit': 'packs/year',
        'min': 0.0,
        'max': 510.0,
        'default': 0.0,
        'step': 5.0,
        'help': 'Cigarette packs smoked per year.'
    },

    # Group 2: Clinical Symptoms & History
    'Symptoms': {
        'label': 'Symptoms at Diagnosis',
        'type': 'binary',
        'group': 'Clinical History & Findings',
        'options': {1: 'Yes', 0: 'No'},
        'default': 1,
        'help': 'Presence of overt clinical symptoms at time of diagnosis.'
    },
    'Cirrhosis': {
        'label': 'Liver Cirrhosis',
        'type': 'binary',
        'group': 'Clinical History & Findings',
        'options': {1: 'Yes', 0: 'No'},
        'default': 1,
        'help': 'Documented presence of underlying liver cirrhosis.'
    },
    'Endemic': {
        'label': 'Endemic Country Residence',
        'type': 'binary',
        'group': 'Clinical History & Findings',
        'options': {1: 'Yes', 0: 'No'},
        'default': 0,
        'help': 'Origin or residence in countries endemic for hepatitis.'
    },
    'Diabetes': {
        'label': 'Diabetes Mellitus',
        'type': 'binary',
        'group': 'Clinical History & Findings',
        'options': {1: 'Yes', 0: 'No'},
        'default': 0,
        'help': 'History of diabetes mellitus.'
    },
    'Obesity': {
        'label': 'Obesity',
        'type': 'binary',
        'group': 'Clinical History & Findings',
        'options': {1: 'Yes', 0: 'No'},
        'default': 0,
        'help': 'Clinical obesity (BMI >= 30 kg/m2).'
    },
    'AHT': {
        'label': 'Arterial Hypertension',
        'type': 'binary',
        'group': 'Clinical History & Findings',
        'options': {1: 'Yes', 0: 'No'},
        'default': 0,
        'help': 'Diagnosed high blood pressure.'
    },
    'CRI': {
        'label': 'Chronic Renal Insufficiency',
        'type': 'binary',
        'group': 'Clinical History & Findings',
        'options': {1: 'Yes', 0: 'No'},
        'default': 0,
        'help': 'Impaired baseline kidney function.'
    },
    'HIV': {
        'label': 'HIV Infection',
        'type': 'binary',
        'group': 'Clinical History & Findings',
        'options': {1: 'Positive', 0: 'Negative'},
        'default': 0,
        'help': 'Human immunodeficiency virus seropositivity.'
    },
    'NASH': {
        'label': 'NASH (Steatohepatitis)',
        'type': 'binary',
        'group': 'Clinical History & Findings',
        'options': {1: 'Yes', 0: 'No'},
        'default': 0,
        'help': 'Nonalcoholic steatohepatitis.'
    },
    'Varices': {
        'label': 'Esophageal Varices',
        'type': 'binary',
        'group': 'Clinical History & Findings',
        'options': {1: 'Yes', 0: 'No'},
        'default': 1,
        'help': 'Enlarged veins in esophagus from portal hypertension.'
    },
    'Spleno': {
        'label': 'Splenomegaly',
        'type': 'binary',
        'group': 'Clinical History & Findings',
        'options': {1: 'Yes', 0: 'No'},
        'default': 0,
        'help': 'Abnormal enlargement of the spleen.'
    },
    'PHT': {
        'label': 'Portal Hypertension',
        'type': 'binary',
        'group': 'Clinical History & Findings',
        'options': {1: 'Yes', 0: 'No'},
        'default': 1,
        'help': 'Elevated blood pressure within portal venous system.'
    },
    'PVT': {
        'label': 'Portal Vein Thrombosis',
        'type': 'binary',
        'group': 'Clinical History & Findings',
        'options': {1: 'Yes', 0: 'No'},
        'default': 0,
        'help': 'Blood clot in the portal vein.'
    },
    'Metastasis': {
        'label': 'Metastasis',
        'type': 'binary',
        'group': 'Clinical History & Findings',
        'options': {1: 'Yes', 0: 'No'},
        'default': 0,
        'help': 'Distant metastasis outside primary liver lesion.'
    },
    'Hallmark': {
        'label': 'Radiological Hallmark',
        'type': 'binary',
        'group': 'Clinical History & Findings',
        'options': {1: 'Yes', 0: 'No'},
        'default': 1,
        'help': 'Arterial hyperenhancement with venous washout on CT/MRI.'
    },

    # Group 3: Viral Hepatitis & Metabolic
    'HBsAg': {
        'label': 'Hepatitis B Surface Antigen (HBsAg)',
        'type': 'binary',
        'group': 'Viral Hepatitis & Risk Markers',
        'options': {1: 'Positive', 0: 'Negative'},
        'default': 0,
        'help': 'Active Hepatitis B surface antigen.'
    },
    'HBeAg': {
        'label': 'Hepatitis B e-Antigen (HBeAg)',
        'type': 'binary',
        'group': 'Viral Hepatitis & Risk Markers',
        'options': {1: 'Positive', 0: 'Negative'},
        'default': 0,
        'help': 'Hepatitis B viral replication marker.'
    },
    'HBcAb': {
        'label': 'Hepatitis B Core Antibody (HBcAb)',
        'type': 'binary',
        'group': 'Viral Hepatitis & Risk Markers',
        'options': {1: 'Positive', 0: 'Negative'},
        'default': 0,
        'help': 'Hepatitis B core total antibody.'
    },
    'HCVAb': {
        'label': 'Hepatitis C Antibody (HCVAb)',
        'type': 'binary',
        'group': 'Viral Hepatitis & Risk Markers',
        'options': {1: 'Positive', 0: 'Negative'},
        'default': 0,
        'help': 'Hepatitis C virus antibody.'
    },
    'Hemochro': {
        'label': 'Hemochromatosis',
        'type': 'binary',
        'group': 'Viral Hepatitis & Risk Markers',
        'options': {1: 'Yes', 0: 'No'},
        'default': 0,
        'help': 'Iron overload genetic condition.'
    },

    # Group 4: Functional Severity Scores
    'PS': {
        'label': 'ECOG Performance Status',
        'type': 'ordinal',
        'group': 'Functional Severity Scores',
        'options': {
            0: '0 - Fully active, unrestricted',
            1: '1 - Strenuous physical restriction',
            2: '2 - Ambulatory, capable of self-care',
            3: '3 - Limited self-care, >50% bed/chair',
            4: '4 - Completely disabled, confined'
        },
        'default': 1,
        'help': 'ECOG performance status grade (0-4 in dataset).'
    },
    'Encephalopathy': {
        'label': 'Hepatic Encephalopathy Degree',
        'type': 'ordinal',
        'group': 'Functional Severity Scores',
        'options': {
            1: '1 - None',
            2: '2 - Grade I / II (Mild/Moderate)',
            3: '3 - Grade III / IV (Severe/Coma)'
        },
        'default': 1,
        'help': 'Neuropsychiatric abnormalities caused by liver insufficiency.'
    },
    'Ascites': {
        'label': 'Ascites Severity Degree',
        'type': 'ordinal',
        'group': 'Functional Severity Scores',
        'options': {
            1: '1 - None',
            2: '2 - Mild (responsive to diuretics)',
            3: '3 - Moderate to Severe (refractory)'
        },
        'default': 1,
        'help': 'Pathological fluid accumulation in peritoneal cavity.'
    },

    # Group 5: Laboratory Tests & Tumor Markers
    'INR': {
        'label': 'INR (International Normalised Ratio)',
        'type': 'numerical',
        'group': 'Laboratory Biomarkers & Tumor',
        'unit': 'ratio',
        'min': 0.8,
        'max': 5.0,
        'default': 1.3,
        'step': 0.05,
        'help': 'Prothrombin time measure of liver synthetic function.'
    },
    'AFP': {
        'label': 'Alpha-Fetoprotein (AFP)',
        'type': 'numerical',
        'group': 'Laboratory Biomarkers & Tumor',
        'unit': 'ng/mL',
        'min': 1.0,
        'max': 2000000.0,
        'default': 33.0,
        'step': 10.0,
        'help': 'Key hepatocellular carcinoma oncological biomarker.'
    },
    'Hemoglobin': {
        'label': 'Hemoglobin',
        'type': 'numerical',
        'group': 'Laboratory Biomarkers & Tumor',
        'unit': 'g/dL',
        'min': 4.0,
        'max': 20.0,
        'default': 13.0,
        'step': 0.1,
        'help': 'Blood hemoglobin concentration.'
    },
    'MCV': {
        'label': 'Mean Corpuscular Volume (MCV)',
        'type': 'numerical',
        'group': 'Laboratory Biomarkers & Tumor',
        'unit': 'fL',
        'min': 60.0,
        'max': 130.0,
        'default': 95.0,
        'step': 0.5,
        'help': 'Average volume of red blood cells.'
    },
    'Leucocytes': {
        'label': 'Leukocytes (WBC)',
        'type': 'numerical',
        'group': 'Laboratory Biomarkers & Tumor',
        'unit': 'G/L',
        'min': 1.0,
        'max': 15000.0,
        'default': 7.2,
        'step': 0.5,
        'help': 'Total white blood cell count.'
    },
    'Platelets': {
        'label': 'Platelet Count',
        'type': 'numerical',
        'group': 'Laboratory Biomarkers & Tumor',
        'unit': 'G/L',
        'min': 1.0,
        'max': 500000.0,
        'default': 93000.0,
        'step': 1000.0,
        'help': 'Platelet count (reflects portal hypertension/hypersplenism).'
    },
    'Albumin': {
        'label': 'Serum Albumin',
        'type': 'numerical',
        'group': 'Laboratory Biomarkers & Tumor',
        'unit': 'g/dL',
        'min': 1.0,
        'max': 6.0,
        'default': 3.4,
        'step': 0.1,
        'help': 'Major liver-synthesized plasma protein.'
    },
    'Total_Bil': {
        'label': 'Total Bilirubin',
        'type': 'numerical',
        'group': 'Laboratory Biomarkers & Tumor',
        'unit': 'mg/dL',
        'min': 0.1,
        'max': 45.0,
        'default': 1.4,
        'step': 0.1,
        'help': 'Liver excretory biomarker.'
    },
    'Dir_Bil': {
        'label': 'Direct Bilirubin',
        'type': 'numerical',
        'group': 'Laboratory Biomarkers & Tumor',
        'unit': 'mg/dL',
        'min': 0.05,
        'max': 30.0,
        'default': 0.7,
        'step': 0.1,
        'help': 'Conjugated bilirubin fraction.'
    },
    'ALT': {
        'label': 'Alanine Transaminase (ALT)',
        'type': 'numerical',
        'group': 'Laboratory Biomarkers & Tumor',
        'unit': 'U/L',
        'min': 5.0,
        'max': 500.0,
        'default': 50.0,
        'step': 1.0,
        'help': 'Enzyme indicating liver cell injury.'
    },
    'AST': {
        'label': 'Aspartate Transaminase (AST)',
        'type': 'numerical',
        'group': 'Laboratory Biomarkers & Tumor',
        'unit': 'U/L',
        'min': 10.0,
        'max': 600.0,
        'default': 71.0,
        'step': 1.0,
        'help': 'Liver and mitochondrial injury enzyme.'
    },
    'GGT': {
        'label': 'Gamma-Glutamyl Transferase (GGT)',
        'type': 'numerical',
        'group': 'Laboratory Biomarkers & Tumor',
        'unit': 'U/L',
        'min': 10.0,
        'max': 1600.0,
        'default': 180.0,
        'step': 5.0,
        'help': 'Biliary enzyme sensitive to alcohol and cholestasis.'
    },
    'ALP': {
        'label': 'Alkaline Phosphatase (ALP)',
        'type': 'numerical',
        'group': 'Laboratory Biomarkers & Tumor',
        'unit': 'U/L',
        'min': 1.0,
        'max': 1000.0,
        'default': 162.0,
        'step': 2.0,
        'help': 'Enzyme associated with biliary duct obstruction.'
    },
    'TP': {
        'label': 'Total Proteins',
        'type': 'numerical',
        'group': 'Laboratory Biomarkers & Tumor',
        'unit': 'g/dL',
        'min': 2.0,
        'max': 110.0,
        'default': 7.1,
        'step': 0.1,
        'help': 'Total serum protein concentration.'
    },
    'Creatinine': {
        'label': 'Serum Creatinine',
        'type': 'numerical',
        'group': 'Laboratory Biomarkers & Tumor',
        'unit': 'mg/dL',
        'min': 0.1,
        'max': 10.0,
        'default': 0.85,
        'step': 0.05,
        'help': 'Kidney filtration marker (crucial for hepatorenal syndrome).'
    },
    'Nodules': {
        'label': 'Number of Nodules',
        'type': 'numerical',
        'group': 'Laboratory Biomarkers & Tumor',
        'unit': 'nodules',
        'min': 0.0,
        'max': 5.0,
        'default': 2.0,
        'step': 1.0,
        'help': 'Count of detected tumor nodules (0-5, 5 indicates 5 or more).'
    },
    'Major_Dim': {
        'label': 'Major Dimension of Largest Nodule',
        'type': 'numerical',
        'group': 'Laboratory Biomarkers & Tumor',
        'unit': 'cm',
        'min': 0.5,
        'max': 25.0,
        'default': 5.0,
        'step': 0.5,
        'help': 'Diameter of largest tumor lesion in centimeters.'
    },
    'Iron': {
        'label': 'Serum Iron',
        'type': 'numerical',
        'group': 'Laboratory Biomarkers & Tumor',
        'unit': 'mcg/dL',
        'min': 0.0,
        'max': 250.0,
        'default': 83.0,
        'step': 2.0,
        'help': 'Circulating iron in bloodstream.'
    },
    'Sat': {
        'label': 'Transferrin Oxygen/Iron Saturation',
        'type': 'numerical',
        'group': 'Laboratory Biomarkers & Tumor',
        'unit': '%',
        'min': 0.0,
        'max': 130.0,
        'default': 27.0,
        'step': 1.0,
        'help': 'Percentage of transferrin saturated with iron.'
    },
    'Ferritin': {
        'label': 'Ferritin Level',
        'type': 'numerical',
        'group': 'Laboratory Biomarkers & Tumor',
        'unit': 'ng/mL',
        'min': 0.0,
        'max': 2500.0,
        'default': 295.0,
        'step': 10.0,
        'help': 'Total intracellular iron storage protein.'
    }
}


def find_dataset_file():
    """Locate dataset file in current directory or data/ directory."""
    candidates = [
        'data/hcc-data.csv',
        'data/hcc-data.txt',
        'hcc-data.csv',
        'hcc-data.txt'
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_dataset(file_path=None):
    """
    Load the HCC dataset, properly converting '?' to NaN.
    Returns: DataFrame with correct column names and data types.
    """
    if file_path is None:
        file_path = find_dataset_file()
    if file_path is None:
        raise FileNotFoundError(
            "Dataset file not found! Please place 'hcc-data.txt' or 'hcc-data.csv' in the workspace or data/ directory."
        )

    # Check if file has header or not
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline().strip()

    if 'Gender' in first_line:
        # File has header
        df = pd.read_csv(file_path, na_values='?')
    else:
        # Raw file without header
        df = pd.read_csv(file_path, header=None, names=ALL_COLUMNS, na_values='?')

    # Ensure all required columns are present
    missing_cols = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    return df


def build_preprocessing_pipeline():
    """
    Constructs a Scikit-learn ColumnTransformer:
    - Numerical features: Median Imputation + Standard Scaling
    - Categorical/Ordinal features: Most Frequent Imputation (retaining original coding)
    Prevents data leakage by fitting only on training data.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_COLS),
            ('cat', cat_pipeline, CATEGORICAL_COLS)
        ],
        remainder='drop'
    )
    return preprocessor


def create_classifier(model_name='logistic_regression'):
    """Instantiate classifier with balanced class weighting for reproducibility."""
    if model_name == 'random_forest':
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced',
            random_state=42
        )
    # Default: Logistic Regression (interpretable clinical baseline)
    return LogisticRegression(
        C=0.5,
        class_weight='balanced',
        solver='liblinear',
        random_state=42,
        max_iter=1000
    )


def train_and_evaluate_models(data_path=None, model_type='logistic_regression', save_artifacts=True):
    """
    Executes full Machine Learning workflow:
    1. Loads dataset and separates features X and target y.
    2. Stratified 80/20 train/test split.
    3. Runs 5-fold Stratified Cross-Validation on the training set.
    4. Fits pipeline strictly on training data (no data leakage).
    5. Evaluates on held-out test set.
    6. Saves pipeline to models/hcc_pipeline.joblib and metrics to models/metrics.json.
    """
    df = load_dataset(data_path)

    # Separate input features and target column (prevent target leakage)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    # Stratified Train/Test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Cross-validation comparison for both models on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    models_to_test = {
        'logistic_regression': create_classifier('logistic_regression'),
        'random_forest': create_classifier('random_forest')
    }

    comparison_results = {}
    fitted_pipelines = {}

    for name, clf in models_to_test.items():
        preprocessor = build_preprocessing_pipeline()
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        # Stratified CV on training set only
        cv_scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)

        # Fit final pipeline on full training set
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        # Evaluate on held-out test set
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Class-specific metrics
        prec_dies = float(precision_score(y_test, y_pred, pos_label=0, zero_division=0))
        rec_dies = float(recall_score(y_test, y_pred, pos_label=0, zero_division=0))
        f1_dies = float(f1_score(y_test, y_pred, pos_label=0, zero_division=0))

        comparison_results[name] = {
            'cv_mean': {
                'accuracy': float(np.mean(cv_scores['test_accuracy'])),
                'precision': float(np.mean(cv_scores['test_precision'])),
                'recall': float(np.mean(cv_scores['test_recall'])),
                'f1': float(np.mean(cv_scores['test_f1'])),
                'roc_auc': float(np.mean(cv_scores['test_roc_auc']))
            },
            'cv_std': {
                'accuracy': float(np.std(cv_scores['test_accuracy'])),
                'precision': float(np.std(cv_scores['test_precision'])),
                'recall': float(np.std(cv_scores['test_recall'])),
                'f1': float(np.std(cv_scores['test_f1'])),
                'roc_auc': float(np.std(cv_scores['test_roc_auc']))
            },
            'test_set': {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'roc_auc': auc,
                'confusion_matrix': cm,
                'class_0_dies': {
                    'precision': prec_dies,
                    'recall': rec_dies,
                    'f1': f1_dies,
                    'support': int((y_test == 0).sum())
                },
                'class_1_survives': {
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'support': int((y_test == 1).sum())
                }
            }
        }

    # Selected model pipeline
    selected_name = model_type if model_type in fitted_pipelines else 'logistic_regression'
    chosen_pipeline = fitted_pipelines[selected_name]

    # Package comprehensive summary
    dataset_summary = {
        'total_records': int(len(df)),
        'total_features': int(len(FEATURE_COLUMNS)),
        'class_distribution': {
            'survives_1': int((y == 1).sum()),
            'dies_0': int((y == 0).sum()),
            'pct_survives': float(round((y == 1).mean() * 100, 2)),
            'pct_dies': float(round((y == 0).mean() * 100, 2))
        },
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'missing_values_overall_pct': float(round(df[FEATURE_COLUMNS].isna().mean().mean() * 100, 2))
    }

    results_payload = {
        'selected_model': selected_name,
        'dataset_summary': dataset_summary,
        'comparison': comparison_results,
        'feature_columns': FEATURE_COLUMNS,
        'numerical_columns': NUMERICAL_COLS,
        'categorical_columns': CATEGORICAL_COLS
    }

    if save_artifacts:
        os.makedirs('models', exist_ok=True)
        # Atomic safe write of model
        temp_model_path = 'models/hcc_pipeline.joblib.tmp'
        joblib.dump(chosen_pipeline, temp_model_path)
        os.replace(temp_model_path, 'models/hcc_pipeline.joblib')

        # Write metrics JSON
        with open('models/metrics.json', 'w', encoding='utf-8') as f:
            json.dump(results_payload, f, indent=2)

    return chosen_pipeline, results_payload


if __name__ == '__main__':
    print("==================================================================")
    print("      HCC SURVIVAL PREDICTION - MACHINE LEARNING PIPELINE         ")
    print("==================================================================")
    print("[1/5] Loading real HCC dataset...")
    data_file = find_dataset_file()
    print(f"      Dataset found at: {data_file}")

    print("[2/5] Training models with Stratified CV & held-out test set...")
    pipeline, results = train_and_evaluate_models()

    summary = results['dataset_summary']
    print(f"\n[Dataset Overview]")
    print(f" - Total Patient Records: {summary['total_records']}")
    print(f" - Clinical Features:     {summary['total_features']}")
    print(f" - Class 1 (Survives):    {summary['class_distribution']['survives_1']} ({summary['class_distribution']['pct_survives']}%)")
    print(f" - Class 0 (Dies):        {summary['class_distribution']['dies_0']} ({summary['class_distribution']['pct_dies']}%)")
    print(f" - Overall Missing Rate:  {summary['missing_values_overall_pct']}%")

    print("\n[3/5] Model Performance Comparison:")
    for model_name, eval_data in results['comparison'].items():
        title = "Logistic Regression (Baseline)" if model_name == 'logistic_regression' else "Random Forest Classifier"
        cv = eval_data['cv_mean']
        cv_std = eval_data['cv_std']
        test = eval_data['test_set']

        print(f"\n--------------------------------------------------------------")
        print(f" MODEL: {title}")
        print(f"--------------------------------------------------------------")
        print(f" 5-Fold CV on Training Set (Mean +/- Std):")
        print(f"   Accuracy:  {cv['accuracy']:.4f} +/- {cv_std['accuracy']:.4f}")
        print(f"   F1-Score:  {cv['f1']:.4f} +/- {cv_std['f1']:.4f}")
        print(f"   ROC-AUC:   {cv['roc_auc']:.4f} +/- {cv_std['roc_auc']:.4f}")
        print(f" Final Held-out Test Set (N={summary['test_samples']}):")
        print(f"   Accuracy:  {test['accuracy']:.4f}")
        print(f"   Precision: {test['precision']:.4f}")
        print(f"   Recall:    {test['recall']:.4f}")
        print(f"   F1-Score:  {test['f1']:.4f}")
        print(f"   ROC-AUC:   {test['roc_auc']:.4f}")
        cm = test['confusion_matrix']
        print(f"   Confusion Matrix:")
        print(f"     [TN={cm[0][0]:2d} (True Dies)   FP={cm[0][1]:2d} (False Survives)]")
        print(f"     [FN={cm[1][0]:2d} (False Dies)  TP={cm[1][1]:2d} (True Survives)]")

    print("\n[4/5] Model Artifacts Saved:")
    print(" - Pipeline: models/hcc_pipeline.joblib")
    print(" - Metrics:  models/metrics.json")
    print("\n[5/5] Done! Training completed successfully.")

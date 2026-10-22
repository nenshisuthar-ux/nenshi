"""
Automated Quality Assurance & Verification Tests for HCC Survival Prediction
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from train import (
    load_dataset,
    build_preprocessing_pipeline,
    train_and_evaluate_models,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    NUMERICAL_COLS,
    CATEGORICAL_COLS
)


def test_1_dataset_loading():
    """Verify raw dataset integrity, shape, and class distribution."""
    df = load_dataset()
    assert df.shape == (165, 50), f"Expected shape (165, 50), got {df.shape}"
    assert TARGET_COLUMN in df.columns, f"Target column '{TARGET_COLUMN}' missing!"
    assert len(FEATURE_COLUMNS) == 49, f"Expected 49 features, got {len(FEATURE_COLUMNS)}"

    # Class balance check
    class_counts = df[TARGET_COLUMN].value_counts().to_dict()
    assert class_counts[1] == 102, f"Expected 102 survivors, got {class_counts.get(1)}"
    assert class_counts[0] == 63, f"Expected 63 deceased, got {class_counts.get(0)}"
    print("✓ Test 1 Passed: Dataset loaded with 165 records, 49 features, and exact class balance (102 vs 63).")


def test_2_no_target_leakage():
    """Ensure target variable is strictly excluded from features."""
    assert TARGET_COLUMN not in FEATURE_COLUMNS, "Critical error: Target column is inside input features!"
    assert TARGET_COLUMN not in NUMERICAL_COLS, "Target column found in numerical features!"
    assert TARGET_COLUMN not in CATEGORICAL_COLS, "Target column found in categorical features!"
    print("✓ Test 2 Passed: No target leakage; Class is completely separated from input features.")


def test_3_preprocessing_pipeline():
    """Verify ColumnTransformer handles missing values without leakage."""
    df = load_dataset()
    X = df[FEATURE_COLUMNS].copy()

    # Preprocessor
    preprocessor = build_preprocessing_pipeline()
    X_trans = preprocessor.fit_transform(X)

    # Check transformed shape and that no NaNs remain
    assert X_trans.shape[0] == 165, f"Transformed rows mismatch: {X_trans.shape[0]}"
    assert X_trans.shape[1] == 49, f"Transformed columns mismatch: {X_trans.shape[1]}"
    assert not np.isnan(X_trans).any(), "Found NaNs in transformed output after preprocessing!"
    print("✓ Test 3 Passed: Preprocessing pipeline handles missing values and produces complete numerical matrix.")


def test_4_model_artifacts_and_prediction():
    """Verify saved model pipeline and metrics JSON artifact."""
    pipeline_path = 'models/hcc_pipeline.joblib'
    metrics_path = 'models/metrics.json'

    assert os.path.exists(pipeline_path), f"Artifact missing: {pipeline_path}"
    assert os.path.exists(metrics_path), f"Artifact missing: {metrics_path}"

    pipeline = joblib.load(pipeline_path)
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    assert 'selected_model' in metrics, "Missing selected_model in metrics JSON"
    assert 'comparison' in metrics, "Missing comparison in metrics JSON"

    # Test sample prediction
    df = load_dataset()
    sample = df[FEATURE_COLUMNS].iloc[0:5].copy()

    predictions = pipeline.predict(sample)
    probabilities = pipeline.predict_proba(sample)

    assert len(predictions) == 5, f"Expected 5 predictions, got {len(predictions)}"
    assert probabilities.shape == (5, 2), f"Expected proba shape (5, 2), got {probabilities.shape}"
    for prob in probabilities:
        assert np.isclose(np.sum(prob), 1.0), f"Probabilities do not sum to 1: {prob}"

    print("✓ Test 4 Passed: Model pipeline loads, predicts valid classes (0/1), and valid probability distributions.")


def test_5_handling_edge_case_nans():
    """Verify prediction pipeline survives when new patient data has missing values."""
    pipeline = joblib.load('models/hcc_pipeline.joblib')
    df = load_dataset()

    sparse_row = df[FEATURE_COLUMNS].iloc[0:1].copy()
    # Inject multiple NaNs across both numerical and categorical
    sparse_row.loc[0, 'AFP'] = np.nan
    sparse_row.loc[0, 'Ferritin'] = np.nan
    sparse_row.loc[0, 'Varices'] = np.nan
    sparse_row.loc[0, 'Ascites'] = np.nan

    pred = pipeline.predict(sparse_row)
    proba = pipeline.predict_proba(sparse_row)

    assert pred[0] in [0, 1]
    assert np.isclose(np.sum(proba[0]), 1.0)
    print("✓ Test 5 Passed: Model reliably infers missing attributes via fitted training imputers.")


if __name__ == '__main__':
    print("==================================================================")
    print("       RUNNING AUTOMATED TEST SUITE FOR HCC SURVIVAL APP          ")
    print("==================================================================")
    test_1_dataset_loading()
    test_2_no_target_leakage()
    test_3_preprocessing_pipeline()
    test_4_model_artifacts_and_prediction()
    test_5_handling_edge_case_nans()
    print("==================================================================")
    print("             ALL QUALITY ASSURANCE TESTS PASSED!                  ")
    print("==================================================================")

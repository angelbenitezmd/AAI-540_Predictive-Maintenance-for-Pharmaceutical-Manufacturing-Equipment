import argparse
import json
import os
import pathlib
import tarfile
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def calculate_f2_score(precision, recall):
    """Calculate F2 score which weighs recall higher than precision"""
    if precision + recall == 0:
        return 0
    return (5 * precision * recall) / (4 * precision + recall)

def evaluate_model(model_path, test_data_path, output_path):
    """
    Evaluate the predictive maintenance model using the same metrics
    as defined in the model monitoring setup.
    """
    # Load the model
    model_file = None
    with tarfile.open(model_path) as tar:
        for file in tar.getmembers():
            if file.name.endswith('.model'):
                model_file = file.name
                tar.extract(file)
    
    if model_file is None:
        raise ValueError("No XGBoost model file found in tarball")
    
    model = xgb.Booster()
    model.load_model(model_file)
    
    # Load test data
    test_data = pd.read_csv(test_data_path)
    X_test = test_data.drop(['label'], axis=1)
    y_test = test_data['label']
    
    # Convert to DMatrix for XGBoost
    dtest = xgb.DMatrix(X_test)
    
    # Make predictions
    probabilities = model.predict(dtest)
    predictions = (probabilities > 0.8).astype(int)  # Use same threshold as monitoring
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    
    # Calculate metrics matching the monitoring setup
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f2 = calculate_f2_score(precision, recall)
    
    # Create evaluation report matching monitoring metrics format
    evaluation_report = {
        "binary_classification_metrics": {
            "confusion_matrix": {
                "0": {"0": int(tn), "1": int(fp)},
                "1": {"0": int(fn), "1": int(tp)}
            },
            "accuracy": {
                "value": float(accuracy_score(y_test, predictions)),
                "standard_deviation": float(np.std((y_test == predictions).astype(float)))
            },
            "precision": {
                "value": float(precision),
                "standard_deviation": float(np.std((predictions == 1) & (y_test == 1)))
            },
            "recall": {
                "value": float(recall),
                "standard_deviation": float(np.std((y_test == 1) & (predictions == 1)))
            },
            "f1": {
                "value": float(f1_score(y_test, predictions)),
                "standard_deviation": 0.0
            },
            "f2": {
                "value": float(f2),
                "standard_deviation": 0.0
            }
        },
        "regression_metrics": {
            "mse": {
                "value": float(mean_squared_error(y_test, probabilities)),
                "standard_deviation": float(np.std((y_test - probabilities) ** 2))
            }
        }
    }
    
    # Save report
    os.makedirs(output_path, exist_ok=True)
    evaluation_path = os.path.join(output_path, "evaluation.json")
    with open(evaluation_path, "w") as f:
        json.dump(evaluation_report, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--test-data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.test_data_path, args.output_path)
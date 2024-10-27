import argparse
import json
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import tarfile
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix

def evaluate_model(model_path, test_path, output_path):
    """
    Comprehensive model evaluation with enhanced error handling and path management
    """
    try:
        print("\n=== Starting Enhanced Evaluation ===")
        print(f"Model path: {model_path}")
        print(f"Test path: {test_path}")
        print(f"Output path: {output_path}")
        
        # Create output directory structure
        os.makedirs(output_path, exist_ok=True)
        
        # Load and verify model
        print("Loading model...")
        model_tar = os.path.join(model_path, "model.tar.gz")
        if not os.path.exists(model_tar):
            raise FileNotFoundError(f"Model file not found at {model_tar}")
            
        with tarfile.open(model_tar, 'r:gz') as tar:
            tar.extractall(path=model_path)
        
        model_files = [f for f in os.listdir(model_path) if f.endswith('.model') or f == 'xgboost-model']
        if not model_files:
            raise ValueError(f"No model files found in {model_path}")
            
        model = xgb.Booster()
        model.load_model(os.path.join(model_path, model_files[0]))
        print("Model loaded successfully")
        
        # Load and verify test data
        print("Loading test data...")
        test_files = os.listdir(test_path)
        print(f"Files in test path: {test_files}")
        
        test_csv = os.path.join(test_path, "test.csv")
        if not os.path.exists(test_csv):
            raise FileNotFoundError(f"Test data not found at {test_csv}")
            
        test_data = pd.read_csv(test_csv, header=None)
        print(f"Test data shape: {test_data.shape}")
        
        # Prepare test data
        if test_data.shape[1] < 2:
            raise ValueError(f"Test data has insufficient columns: {test_data.shape[1]}")
            
        X_test = test_data.iloc[:, 1:]  # All columns except first
        y_test = test_data.iloc[:, 0]   # First column is label
        
        print(f"Features shape: {X_test.shape}")
        print(f"Labels shape: {y_test.shape}")
        print(f"Unique labels in test data: {np.unique(y_test)}")
        
        # Make predictions
        print("Making predictions...")
        dtest = xgb.DMatrix(X_test)
        probabilities = model.predict(dtest)
        predictions = (probabilities > 0.5).astype(int)
        
        # Calculate metrics
        print("Calculating metrics...")
        metrics = {
            "binary_classification_metrics": {
                "accuracy": {
                    "value": float(accuracy_score(y_test, predictions)),
                    "standard_deviation": float(np.std((y_test == predictions).astype(float)))
                },
                "precision": {
                    "value": float(precision_score(y_test, predictions, zero_division=0)),
                    "standard_deviation": 0.0
                },
                "recall": {
                    "value": float(recall_score(y_test, predictions, zero_division=0)),
                    "standard_deviation": 0.0
                },
                "f1": {
                    "value": float(f1_score(y_test, predictions, zero_division=0)),
                    "standard_deviation": 0.0
                },
                "auc_roc": {
                    "value": float(roc_auc_score(y_test, probabilities)),
                    "standard_deviation": 0.0
                }
            }
        }
        
        # Save metrics with explicit path handling
        print("Saving metrics...")
        metrics_file = os.path.join(output_path, "evaluation.json")
        print(f"Writing metrics to: {metrics_file}")
        
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print("Files in output directory after saving:")
        print(os.listdir(output_path))
        
        print("Evaluation completed successfully")
        print("Metrics summary:")
        print(json.dumps(metrics, indent=2))
        
    except Exception as e:
        print(f"\nERROR: Evaluation failed with error: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.test_path, args.output_path)
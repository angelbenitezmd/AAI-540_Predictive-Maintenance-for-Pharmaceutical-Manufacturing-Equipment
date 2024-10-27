import argparse
import os
import pandas as pd
import numpy as np

def preprocess_maintenance_data(input_path, output_path):
    """
    Process train and test data with numeric columns
    """
    print(f"Starting preprocessing with input_path={input_path}, output_path={output_path}")
    
    try:
        # Read training data
        train_data = pd.read_csv(os.path.join(input_path, "train_data.csv"), header=None)
        print("\nTraining Data Shape:", train_data.shape)
        print("Training Data Sample:")
        print(train_data.head())
        
        # First column (0) is the label/target
        train_features = train_data.iloc[:, 1:]  # All columns except first
        train_labels = train_data.iloc[:, 0]     # First column is label
        
        # Read test data
        test_data = pd.read_csv(os.path.join(input_path, "test_data.csv"), header=None)
        print("\nTest Data Shape:", test_data.shape)
        print("Test Data Sample:")
        print(test_data.head())
        
        # For test data, drop date column and use second column as label
        test_features = test_data.iloc[:, 2:]    # All columns except first two
        test_labels = test_data.iloc[:, 1]       # Second column is label
        
        # Combine features and labels
        processed_train = pd.concat([train_labels, train_features], axis=1)
        processed_test = pd.concat([test_labels, test_features], axis=1)
        
        print("\nProcessed Data Info:")
        print(f"Training data shape: {processed_train.shape}")
        print(f"Testing data shape: {processed_test.shape}")
        
        # Create output directories
        os.makedirs(os.path.join(output_path, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "test"), exist_ok=True)
        
        # Save processed data
        train_output = os.path.join(output_path, "train", "train.csv")
        test_output = os.path.join(output_path, "test", "test.csv")
        
        # Save without headers since we're using numeric columns
        processed_train.to_csv(train_output, index=False, header=False)
        processed_test.to_csv(test_output, index=False, header=False)
        
        print("\nFiles saved:")
        print(f"Training data: {train_output}")
        print(f"Testing data: {test_output}")
        
        # Print column information for verification
        print("\nColumn Structure:")
        print("Training data:")
        print(f"- Label column: 0")
        print(f"- Feature columns: 1-{processed_train.shape[1]-1}")
        print("\nTest data:")
        print(f"- Label column: 0")
        print(f"- Feature columns: 1-{processed_test.shape[1]-1}")
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    
    preprocess_maintenance_data(args.input_path, args.output_path)
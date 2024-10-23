import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_maintenance_data(input_path, output_path):
    """
    Preprocess predictive maintenance data for model training and testing.
    Matches the format used in the model monitoring setup.
    """
    print(f"Input Path: {input_path}")
    print(f"Output Path: {output_path}")
    
    # Create output directories
    os.makedirs(os.path.join(output_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "test"), exist_ok=True)
    
    # List input files and load data
    input_files = os.listdir(input_path)
    print(f"Input files: {input_files}")
    
    # Load the data - handle both single file and multiple file cases
    if "data.csv" in input_files:
        df = pd.read_csv(os.path.join(input_path, "data.csv"))
    else:
        # Load all CSV files in the input directory
        dfs = []
        for file in input_files:
            if file.endswith('.csv'):
                file_path = os.path.join(input_path, file)
                print(f"Reading file: {file_path}")
                dfs.append(pd.read_csv(file_path))
        df = pd.concat(dfs, ignore_index=True)
    
    print(f"Initial data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Extract features and label
    if 'label' not in df.columns:
        # If processing captured data, create label from predictions
        if 'probability' in df.columns:
            df['label'] = (df['probability'] > 0.8).astype(int)
        else:
            raise ValueError("Neither 'label' nor 'probability' column found in input data")
    
    # Feature engineering
    scaler = StandardScaler()
    numeric_columns = [col for col in df.columns if col not in ['label']]
    
    # Handle potential NaN values
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Scale numeric features
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    print(f"Processed data shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts(normalize=True)}")
    
    # Split the data
    train_data, test_data = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['label']  # Ensure balanced split for binary classification
    )
    
    # Save processed datasets
    train_output_path = os.path.join(output_path, "train", "train.csv")
    test_output_path = os.path.join(output_path, "test", "test.csv")
    
    print(f"Saving training data to: {train_output_path}")
    print(f"Saving test data to: {test_output_path}")
    
    train_data.to_csv(train_output_path, index=False, header=True)
    test_data.to_csv(test_output_path, index=False, header=True)
    
    print("Preprocessing completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    
    print("Starting preprocessing job")
    preprocess_maintenance_data(args.input_path, args.output_path)
    print("Preprocessing job completed")
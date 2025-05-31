import os
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_digits
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_dataset(X, y, name, base_dir):
    """
    Saves the dataset (X, y) to the specified directory.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Label vector.
        name (str): Name of the dataset (e.g., 'breast_cancer').
        base_dir (str): The base directory to save the 'sets' folder.
    """
    sets_dir = os.path.join(base_dir, 'sets')
    os.makedirs(sets_dir, exist_ok=True)

    filepath = os.path.join(sets_dir, f"{name}.npz")
    try:
        np.savez_compressed(filepath, X=X, y=y)
        logging.info(f"Dataset '{name}' saved successfully to '{filepath}'")
        logging.info(f"X shape: {X.shape}, y shape: {y.shape}, y unique values: {np.unique(y)}")
    except Exception as e:
        logging.error(f"Error saving dataset '{name}': {e}")

def preprocess_features(X):
    """
    Ensures all features are numeric and scales them.
    For this script, sklearn datasets mostly provide numeric features.
    If categorical features were present, more complex preprocessing like
    one-hot encoding would be needed here.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.astype(np.float64))
    return X_scaled

def main():
    """
    Loads, processes, and saves multiple datasets.
    """
    # Determine the base directory (e.g., ../ from the script's directory)
    # __file__ is the path to the current script
    # os.path.abspath(__file__) gets the absolute path of the script
    # os.path.dirname(...) gets the directory of the script
    # os.path.join(..., '..') goes one level up
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback for environments where __file__ is not defined (e.g., some notebooks)
        script_dir = os.getcwd()
        logging.warning(f"'__file__' not defined. Using current working directory: {script_dir}")

    base_storage_dir = os.path.join(script_dir, '..')
    logging.info(f"Base storage directory for 'sets': {os.path.abspath(base_storage_dir)}")

    # 1. Breast Cancer Dataset (already binary)
    logging.info("Processing Breast Cancer dataset...")
    cancer = load_breast_cancer()
    X_cancer, y_cancer = cancer.data, cancer.target
    # Ensure labels are 0 and 1 (already the case for breast_cancer)
    # Ensure features are numeric (already the case)
    X_cancer_processed = preprocess_features(X_cancer)
    save_dataset(X_cancer_processed, y_cancer.astype(int), 'breast_cancer', base_storage_dir)
    print("-" * 30)

    # 2. Iris Dataset (convert to binary: class 0 vs. rest)
    logging.info("Processing Iris dataset (class 0 vs. rest)...")
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    # Convert to binary: class 0 vs. (class 1 and class 2)
    # Label class 0 as 0, and classes 1 and 2 as 1
    y_iris_binary = np.where(y_iris == 0, 0, 1)
    X_iris_processed = preprocess_features(X_iris)
    save_dataset(X_iris_processed, y_iris_binary.astype(int), 'iris_binary_0_vs_rest', base_storage_dir)
    print("-" * 30)

    # 3. Iris Dataset (convert to binary: class 0 vs. class 1)
    logging.info("Processing Iris dataset (class 0 vs. class 1)...")
    iris_0_vs_1_filter = np.where(y_iris <= 1) # Filter for classes 0 and 1
    X_iris_01 = X_iris[iris_0_vs_1_filter]
    y_iris_01 = y_iris[iris_0_vs_1_filter]
    # Labels are already 0 and 1 for this subset
    X_iris_01_processed = preprocess_features(X_iris_01)
    save_dataset(X_iris_01_processed, y_iris_01.astype(int), 'iris_binary_0_vs_1', base_storage_dir)
    print("-" * 30)

    # 4. Digits Dataset (convert to binary: digit 0 vs. digit 1)
    logging.info("Processing Digits dataset (digit 0 vs. digit 1)...")
    digits = load_digits()
    X_digits, y_digits = digits.data, digits.target
    # Filter for digits 0 and 1
    filter_01 = np.logical_or(y_digits == 0, y_digits == 1)
    X_digits_01 = X_digits[filter_01]
    y_digits_01 = y_digits[filter_01]
    # Labels are already 0 and 1 for this subset
    X_digits_01_processed = preprocess_features(X_digits_01)
    save_dataset(X_digits_01_processed, y_digits_01.astype(int), 'digits_binary_0_vs_1', base_storage_dir)
    print("-" * 30)

    # 5. Digits Dataset (convert to binary: digit 5 vs. rest)
    logging.info("Processing Digits dataset (digit 5 vs. rest)...")
    # Convert to binary: digit 5 as class 1, all other digits as class 0
    y_digits_5_vs_rest = np.where(y_digits == 5, 1, 0)
    X_digits_processed = preprocess_features(X_digits)
    save_dataset(X_digits_processed, y_digits_5_vs_rest.astype(int), 'digits_binary_5_vs_rest', base_storage_dir)
    print("-" * 30)

    logging.info("All datasets processed and saved.")

if __name__ == '__main__':
    main()
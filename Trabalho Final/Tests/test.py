import multiprocessing
import os
import sys
import numpy as np
import pandas as pd
from typing import cast, Tuple, Dict, Any, List
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import logging
from tqdm import tqdm
import time # For timestamping output files if desired

# --- Path Setup ---
# Assuming the script is in a 'scripts' directory, and 'Preprocessing', 'CustomKernelSVC', 'sets'
# are in the parent directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SETS_DIR = os.path.join(BASE_DIR, 'sets')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs') # Save outputs in a subdirectory

sys.path.append(BASE_DIR)

try:
    from Preprocessing.preprocessing import Preprocessor
    from CustomKernelSVC.CustomKernelSVC import CustomKernelSVC
except ImportError as e:
    print(f"Critical Error: Could not import custom modules (Preprocessor or CustomKernelSVC): {e}")
    print("Please ensure these modules are in the correct path (e.g., parent directory) and are error-free.")
    print(f"BASE_DIR (for sys.path): {BASE_DIR}")
    sys.exit(1)

# --- Dataset Names ---
# Ensure these .npz files exist in the SETS_DIR
DATASET_NAMES = [
    # "adult",
    "banknote-authentication",
    "blood-transfusion-service-center",
    "breast_cancer",
    "diabetes",
    "digits_binary_0_vs_1",
    "digits_binary_5_vs_rest",
    "german_credit_g",
    "ionosphere",
    "iris_binary_setosa_vs_rest",
    "iris_binary_setosa_vs_versicolor",
    "kc1",
    # "mushroom",
    "qsar-biodeg",
    "sonar",
    # "spambase",
    "titanic",
    "vote",
    "wpbc"
]
# Example of a potentially missing dataset for testing error handling:
# DATASET_NAMES.append("non_existent_dataset")


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)]) # Also log to console

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
FAILED_LOG_FILE = os.path.join(OUTPUT_DIR, "failed_datasets.log")
# Add a file handler for errors/failed datasets
file_error_handler = logging.FileHandler(FAILED_LOG_FILE, mode='w') # Overwrite for each run
file_error_handler.setLevel(logging.ERROR)
file_error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_error_handler)


def load_npz_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads X and y from a .npz file."""
    path = os.path.join(SETS_DIR, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file {path} not found.")
    data = np.load(path)
    X = data["X"]
    y = data["y"].ravel()  # Ensure y is 1D
    if X.ndim == 1: # Handle cases where X might be 1D (e.g. single feature or error in data generation)
        X = X.reshape(-1, 1)
    return X, y

def run_test_on_dataset_worker(args: Tuple[str, Tuple[float, float]]) -> Dict[str, Any]:
    """
    Worker function to be called by multiprocessing pool.
    Unpacks arguments and calls the main processing function.
    """
    dataset_name, hs_bounds = args
    try:
        X, y = load_npz_dataset(dataset_name)

        # Ensure y is binary if possible, or that StratifiedKFold can handle it.
        # SVC and accuracy scoring generally expect distinct classes.
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            raise ValueError(f"Dataset {dataset_name} has only one class ({unique_labels}). Cannot perform classification.")

        skf = StratifiedKFold(n_splits=10, shuffle=True)

        # 1. Reference SVM
        pipeline_ref = Pipeline([
            ('preprocessor', Preprocessor()), # Assuming Preprocessor is stateless or safely re-initialized
            ('svm', SVC())
        ])

        # 2. Custom Kernel SVM
        pipeline_opt = Pipeline([
            ('preprocessor', Preprocessor()), # Assuming Preprocessor is stateless or safely re-initialized
            ('custom_svm', CustomKernelSVC(
                h_bounds=hs_bounds
            ))
        ])

        svm_ref_scores = cross_val_score(pipeline_ref, X, y, cv=skf, scoring='accuracy', error_score='raise')
        svm_opt_scores = cross_val_score(pipeline_opt, X, y, cv=skf, scoring='accuracy', error_score='raise')

        # Perform paired t-test
        # Cast is used because ttest_rel can return union types depending on scipy version,
        # but for standard use it's (float, float)
        t_stat, p_value = cast(Tuple[float, float], ttest_rel(svm_ref_scores, svm_opt_scores))

        return {
            'dataset': dataset_name,
            'opt_score_mean': float(np.mean(svm_opt_scores)),
            'opt_score_std': float(np.std(svm_opt_scores)),
            'ref_score_mean': float(np.mean(svm_ref_scores)),
            'ref_score_std': float(np.std(svm_ref_scores)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'conclusion': "Not equivalent" if p_value < 0.05 else "Equivalent",
            'status': 'success'
        }
    except FileNotFoundError as fnf_err:
        logging.error(f"Dataset Error for '{dataset_name}': {fnf_err}")
        return {'dataset': dataset_name, 'status': 'error', 'message': str(fnf_err)}
    except ValueError as val_err: # Catch specific errors like single class
        logging.error(f"Data Error for '{dataset_name}': {val_err}")
        return {'dataset': dataset_name, 'status': 'error', 'message': str(val_err)}
    except Exception as e:
        # Log the full traceback for unexpected errors
        logging.error(f"Runtime Error processing dataset '{dataset_name}': {e}", exc_info=True)
        return {'dataset': dataset_name, 'status': 'error', 'message': f"Runtime error: {e}"}


if __name__ == "__main__":
    hs_param_search_range_global = (5e-1, 5e0) # Use a more descriptive name

    try:
        # Use a slightly less aggressive number of processes to leave resources for OS, etc.
        num_processes = max(1, multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1)
    except NotImplementedError:
        num_processes = 2 # Fallback for systems where cpu_count is not implemented
    logging.info(f"Starting dataset tests in parallel using up to {num_processes} processes...")
    logging.info(f"Datasets will be loaded from: {SETS_DIR}")
    logging.info(f"Results and logs will be saved in: {OUTPUT_DIR}")


    # Prepare arguments for starmap-like behavior with imap_unordered
    tasks_args = [(name, hs_param_search_range_global) for name in DATASET_NAMES]

    successful_results: List[Dict[str, Any]] = []
    failed_datasets_info: List[Dict[str, Any]] = []

    # Using imap_unordered to get results as they complete and allow tqdm progress updates
    with multiprocessing.Pool(processes=num_processes) as pool:
        with tqdm(total=len(tasks_args), desc="Processing datasets") as pbar:
            for result in pool.imap_unordered(run_test_on_dataset_worker, tasks_args):
                if result.get('status') == 'success':
                    successful_results.append(result)
                else:
                    failed_datasets_info.append(result)
                pbar.update(1)

    logging.info("All dataset test runs completed or attempted.")

    # --- Save Results ---
    if successful_results:
        df_results = pd.DataFrame(successful_results)
        # Remove status column before saving/printing
        df_to_save = df_results.drop(columns=['status'])

        output_csv_path = os.path.join(OUTPUT_DIR, "all_results.csv")
        try:
            df_to_save.to_csv(output_csv_path, index=False)
            logging.info(f"Detailed results saved to: {output_csv_path}")
        except Exception as e:
            logging.error(f"Failed to save detailed results CSV: {e}")

        print("\nResults for successfully processed datasets:")
        print(df_to_save.to_string(index=False))

        # Calculate and save summary statistics
        # Adjusted column names to match the new DataFrame
        summary_cols = ['opt_score_mean', 'opt_score_std', 'ref_score_mean', 'ref_score_std', 't_statistic', 'p_value']
        # Check if all summary_cols are present in df_to_save to avoid KeyError
        valid_summary_cols = [col for col in summary_cols if col in df_to_save.columns]
        if valid_summary_cols:
            summary = df_to_save[valid_summary_cols].agg(['mean', 'std']).transpose()
            output_summary_path = os.path.join(OUTPUT_DIR, "summary_statistics.txt")
            try:
                with open(output_summary_path, 'w') as f:
                    f.write("Summary statistics (mean and std of metrics over successful datasets):\n")
                    f.write(summary.to_string())
                logging.info(f"Summary statistics saved to: {output_summary_path}")
                print("\nSummary statistics (mean and std of metrics over successful datasets):")
                print(summary)
            except Exception as e:
                logging.error(f"Failed to save summary statistics: {e}")
        else:
            logging.warning("No valid columns found for summary statistics. Skipping summary.")

    else:
        logging.warning("No datasets were processed successfully. No results to save or summarize.")

    if failed_datasets_info:
        logging.warning(f"\n--- Failed Datasets ({len(failed_datasets_info)}) ---")
        for failure in failed_datasets_info:
            logging.warning(f"Dataset: {failure['dataset']}, Reason: {failure['message']}")
        logging.warning(f"Details for failed datasets logged to: {FAILED_LOG_FILE}")
        print(f"\nInformation on {len(failed_datasets_info)} failed dataset(s) logged to: {FAILED_LOG_FILE}")

    logging.info("Script finished.")
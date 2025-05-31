import multiprocessing
import os
import sys
import numpy as np
import pandas as pd
from typing import cast, Any
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import logging
from tqdm import tqdm

# --- Path Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SETS_DIR = os.path.join(BASE_DIR, 'sets')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')

sys.path.append(BASE_DIR)

try:
    from Preprocessing.preprocessing import Preprocessor
    from CustomKernelSVC.CustomKernelSVC import CustomKernelSVC
except ImportError as e:
    print(f"Critical Error: Could not import custom modules (Preprocessor or CustomKernelSVC): {e}")
    sys.exit(1)

DATASET_NAMES = [
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
    "qsar-biodeg",
    "sonar",
    "titanic",
    "vote",
    "wpbc"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
FAILED_LOG_FILE = os.path.join(OUTPUT_DIR, "failed_datasets.log")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
file_error_handler = logging.FileHandler(FAILED_LOG_FILE, mode='w')
file_error_handler.setLevel(logging.ERROR)
file_error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_error_handler)

def load_npz_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    path = os.path.join(SETS_DIR, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file {path} not found.")
    data = np.load(path)
    X = data["X"]
    y = data["y"].ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X, y

def run_compare_worker(args: tuple[str, tuple[np.float32, np.float32], str]) -> dict[str, Any]:
    dataset_name, hs_bounds, objective_metric = args
    try:
        X, y = load_npz_dataset(dataset_name)
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            raise ValueError(f"Dataset {dataset_name} has only one class ({unique_labels}). Cannot perform classification.")

        skf = StratifiedKFold(n_splits=10, shuffle=True)

        # Reference SVM
        pipeline_ref = Pipeline([
            ('preprocessor', Preprocessor()),
            ('svm', SVC())
        ])

        # Custom Kernel SVM with specified objective
        pipeline_custom = Pipeline([
            ('preprocessor', Preprocessor()),
            ('custom_svm', CustomKernelSVC(
                h_bounds=hs_bounds,
                objective_metric=objective_metric
            ))
        ])

        ref_scores = cross_val_score(pipeline_ref, X, y, cv=skf, scoring='accuracy', error_score='raise')
        custom_scores = cross_val_score(pipeline_custom, X, y, cv=skf, scoring='accuracy', error_score='raise')

        t_stat, p_value = cast(tuple[np.float32, np.float32], ttest_rel(ref_scores, custom_scores))

        return {
            'dataset': dataset_name,
            'objective_metric': objective_metric,
            'custom_score_mean': np.float32(np.mean(custom_scores)),
            'custom_score_std': np.float32(np.std(custom_scores)),
            'ref_score_mean': np.float32(np.mean(ref_scores)),
            'ref_score_std': np.float32(np.std(ref_scores)),
            't_statistic': np.float32(t_stat),
            'p_value': np.float32(p_value),
            'conclusion': "Not equivalent" if p_value < 0.05 else "Equivalent",
            'status': 'success'
        }
    except Exception as e:
        logging.error(f"Error processing {dataset_name} ({objective_metric}): {e}", exc_info=True)
        return {'dataset': dataset_name, 'objective_metric': objective_metric, 'status': 'error', 'message': str(e)}

if __name__ == "__main__":
    hs_param_search_range_global = (5e-1, 5e0)
    try:
        num_processes = max(1, multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1)
    except NotImplementedError:
        num_processes = 2
    logging.info(f"Comparing SVC and CustomKernelSVC (both objectives) using up to {num_processes} processes...")

    tasks_args = []
    for name in DATASET_NAMES:
        for obj_metric in ["spatial", "axis"]:
            tasks_args.append((name, hs_param_search_range_global, obj_metric))

    successful_results: list[dict[str, Any]] = []
    failed_datasets_info: list[dict[str, Any]] = []

    with multiprocessing.Pool(processes=num_processes) as pool:
        with tqdm(total=len(tasks_args), desc="Comparing datasets") as pbar:
            for result in pool.imap_unordered(run_compare_worker, tasks_args):
                if result.get('status') == 'success':
                    successful_results.append(result)
                else:
                    failed_datasets_info.append(result)
                pbar.update(1)

    logging.info("All comparisons completed.")

    if successful_results:
        df_results = pd.DataFrame(successful_results)
        df_to_save = df_results.drop(columns=['status'])

        output_csv_path = os.path.join(OUTPUT_DIR, "compare_results.csv")
        try:
            df_to_save.to_csv(output_csv_path, index=False)
            logging.info(f"Comparison results saved to: {output_csv_path}")
        except Exception as e:
            logging.error(f"Failed to save comparison results CSV: {e}")

        print("\nComparison results:")
        print(df_to_save.to_string(index=False))

        summary_cols = ['custom_score_mean', 'custom_score_std', 'ref_score_mean', 'ref_score_std', 't_statistic', 'p_value']
        valid_summary_cols = [col for col in summary_cols if col in df_to_save.columns]
        if valid_summary_cols:
            summary = df_to_save.groupby('objective_metric')[valid_summary_cols].agg(['mean', 'std'])
            output_summary_path = os.path.join(OUTPUT_DIR, "compare_summary_statistics.txt")
            try:
                with open(output_summary_path, 'w') as f:
                    f.write("Summary statistics by objective_metric:\n")
                    f.write(summary.to_string())
                logging.info(f"Summary statistics saved to: {output_summary_path}")
                print("\nSummary statistics by objective_metric:")
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
            logging.warning(f"Dataset: {failure['dataset']} ({failure.get('objective_metric', '')}), Reason: {failure['message']}")
        logging.warning(f"Details for failed datasets logged to: {FAILED_LOG_FILE}")
        print(f"\nInformation on {len(failed_datasets_info)} failed dataset(s) logged to: {FAILED_LOG_FILE}")

    logging.info("Script finished.")
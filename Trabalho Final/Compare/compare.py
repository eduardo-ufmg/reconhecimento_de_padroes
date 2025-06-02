import multiprocessing
import os
import sys
import numpy as np
import json # For hierarchical output
from typing import Any, cast
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
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs') # Save outputs in a subdirectory

# Ensure the project's base directory is in the Python path
# to allow importing custom modules.
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from Preprocessing.preprocessing import Preprocessor
from CustomKernelSVC.CustomKernelSVC import CustomKernelSVC

# --- Dataset Names ---
# Ensure these .npz files exist in the SETS_DIR
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
    "wpbc",
    "adult", # Potentially large/long-running
    "mushroom", # Often has perfect scores or issues with variance
    "spambase", # Potentially large/long-running
]

# --- Logging Configuration ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
FAILED_LOG_FILE = os.path.join(OUTPUT_DIR, "comparison_failed_datasets.log")
# Add a file handler for errors/failed datasets
file_error_handler = logging.FileHandler(FAILED_LOG_FILE, mode='w') # Overwrite for each run
file_error_handler.setLevel(logging.ERROR)
file_error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_error_handler)


def load_npz_dataset(name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Loads X and y from a .npz file."""
    path = os.path.join(SETS_DIR, f"{name}.npz")
    if not os.path.exists(path):
        logger.error(f"Dataset file {path} not found for dataset '{name}'.")
        raise FileNotFoundError(f"Dataset file {path} not found.")
    try:
        data = np.load(path)
        X = data["X"]
        y = data["y"].ravel()  # Ensure y is 1D
        if X.ndim == 1: # Handle cases where X might be 1D
            X = X.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            logger.error(f"Shape mismatch in '{name}': X shape {X.shape}, y shape {y.shape}")
            raise ValueError(f"Shape mismatch between X and y for dataset '{name}'.")
        return X, y
    except Exception as e:
        logger.error(f"Error loading dataset '{name}' from {path}: {e}")
        raise

def run_comparison_on_dataset_worker(
    args: tuple[str, tuple[np.float32, np.float32]]
) -> dict[str, Any]: # Improved type hint
    """
    Worker function to run comparisons for a single dataset.
    Runs three classifiers (ref, opt_2d, opt_axis) once, then performs t-tests.
    """
    dataset_name, hs_bounds = args
    dataset_results: dict[str, Any] = {'dataset': dataset_name} # Improved type hint

    try:
        load_attempt = load_npz_dataset(dataset_name)
        if load_attempt is None:
             dataset_results['status'] = 'error'
             dataset_results['message'] = f"Failed to load dataset '{dataset_name}' (returned None)."
             return dataset_results
        X, y = load_attempt

        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            raise ValueError(f"Dataset {dataset_name} has only one class ({unique_labels}). Cannot perform classification.")

        # Added random_state for reproducible splits
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

        # Define pipelines
        pipeline_ref = Pipeline([
            ('preprocessor', Preprocessor()),
            ('svm', SVC(random_state=0)) # Added random_state
        ])

        pipeline_opt_2d = Pipeline([
            ('preprocessor', Preprocessor()),
            ('custom_svm', CustomKernelSVC(
                h_bounds=hs_bounds,
                objective_metric='spatial',
                kernel_fit_type='scale',
                svm_kwargs={'random_state': 0} # Added for reproducibility of internal SVC
            ))
        ])

        pipeline_opt_axis = Pipeline([
            ('preprocessor', Preprocessor()),
            ('custom_svm', CustomKernelSVC(
                h_bounds=hs_bounds,
                objective_metric='axis',
                kernel_fit_type='scale',
                svm_kwargs={'random_state': 0} # Added for reproducibility of internal SVC
            ))
        ])

        pipelines = {
            "ref": pipeline_ref,
            "opt_2d": pipeline_opt_2d,
            "opt_axis": pipeline_opt_axis
        }

        classifier_scores_raw: dict[str, np.ndarray] = {} # Improved type hint
        classifier_metrics: dict[str, dict[str, np.float32]] = {} # Improved type hint

        for clf_name, pipeline in pipelines.items():
            try:
                scores = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy', error_score='raise')
                classifier_scores_raw[clf_name] = scores
                classifier_metrics[clf_name] = {
                    "mean_accuracy": np.float32(np.mean(scores)),
                    "std_accuracy": np.float32(np.std(scores))
                }
            except Exception as e:
                logger.error(f"Error during cross-validation for {clf_name} on {dataset_name}: {e}")
                raise RuntimeError(f"Cross-validation failed for {clf_name} on {dataset_name}: {e}")

        dataset_results['classifier_metrics'] = classifier_metrics
        
        comparisons_data: dict[str, dict[str, Any]] = {} # Improved type hint
        
        comparison_pairs = [
            ('ref', 'opt_2d', 'ref_vs_opt_2d'),
            ('ref', 'opt_axis', 'ref_vs_opt_axis'),
            ('opt_2d', 'opt_axis', 'opt_2d_vs_opt_axis')
        ]

        for clf1_name, clf2_name, pair_key in comparison_pairs:
            scores1 = classifier_scores_raw[clf1_name]
            scores2 = classifier_scores_raw[clf2_name]
            
            # Check if scores are identical, which leads to NaN in ttest_rel
            if np.array_equal(scores1, scores2):
                t_stat, p_val = np.nan, np.nan
                logger.debug(f"Scores for {clf1_name} and {clf2_name} are identical on dataset {dataset_name}. T-test will result in NaN.")
            else:
                # Cast is used because ttest_rel can return union types
                stats, pval = ttest_rel(scores1, scores2, nan_policy='propagate') # Added nan_policy
                t_stat = cast(np.float32, stats)  # Ensure t_stat is float32
                p_val = cast(np.float32, pval)  # Ensure p_val is float32


            comparisons_data[pair_key] = {
                't_statistic': t_stat,
                'p_value': p_val,
                'conclusion': "Not equivalent" if p_val < 0.05 else "Equivalent" # p_val can be NaN
            }
            # If p_val is NaN, 'Not equivalent' if NaN < 0.05 would be False, so it correctly becomes 'Equivalent'.
        
        dataset_results['statistical_comparisons'] = comparisons_data
        dataset_results['status'] = 'success'

    except FileNotFoundError as fnf_err:
        logger.error(f"Dataset Error for '{dataset_name}': {fnf_err}")
        dataset_results['status'] = 'error'
        dataset_results['message'] = str(fnf_err)
    except ValueError as val_err:
        logger.error(f"Data Error for '{dataset_name}': {val_err}")
        dataset_results['status'] = 'error'
        dataset_results['message'] = str(val_err)
    except RuntimeError as rt_err:
        logger.error(f"Runtime Error during processing '{dataset_name}': {rt_err}")
        dataset_results['status'] = 'error'
        dataset_results['message'] = str(rt_err)
    except Exception as e:
        logger.error(f"Unexpected error processing dataset '{dataset_name}': {e}", exc_info=True)
        dataset_results['status'] = 'error'
        dataset_results['message'] = f"Unexpected error: {e}"
        
    return dataset_results

if __name__ == "__main__":
    hs_param_search_range_global = (np.float32(1e-1), np.float32(1e1))

    try:
        # Consider system load when setting num_processes
        cpu_count = multiprocessing.cpu_count()
        num_processes = max(1, cpu_count - 2 if cpu_count > 2 else (cpu_count - 1 if cpu_count > 1 else 1))
    except NotImplementedError:
        num_processes = 2 
    logger.info(f"Starting dataset comparisons in parallel using up to {num_processes} processes...")
    logger.info(f"Datasets will be loaded from: {SETS_DIR}")
    logger.info(f"Results and logs will be saved in: {OUTPUT_DIR}")

    tasks_args = [(name, hs_param_search_range_global) for name in DATASET_NAMES]

    all_results_dict: dict[str, Any] = {"datasets": {}} # Improved type hint
    failed_datasets_summary: dict[str, str] = {} # Improved type hint
    successful_dataset_count = 0

    with multiprocessing.Pool(processes=num_processes) as pool:
        with tqdm(total=len(tasks_args), desc="Processing datasets") as pbar:
            for result in pool.imap_unordered(run_comparison_on_dataset_worker, tasks_args):
                dataset_name = result['dataset']
                if result.get('status') == 'success':
                    successful_dataset_count += 1
                    all_results_dict["datasets"][dataset_name] = {
                        k: v for k, v in result.items() if k not in ['dataset', 'status']
                    }
                else:
                    failed_datasets_summary[dataset_name] = result.get('message', 'Unknown error')
                    all_results_dict["datasets"][dataset_name] = {
                        'status': 'error',
                        'message': result.get('message', 'Unknown error')
                    }
                pbar.set_postfix_str(f"Success: {successful_dataset_count}, Failed: {len(failed_datasets_summary)}")
                pbar.update(1)
    
    logger.info("All dataset comparison runs completed or attempted.")

    # --- Save Results ---
    output_json_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
    try:
        with open(output_json_path, 'w') as f:
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        # Handle NaN, Inf, -Inf specifically for JSON
                        if np.isnan(obj): return "NaN"
                        if np.isinf(obj): return "Infinity" if obj > 0 else "-Infinity"
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)
            json.dump(all_results_dict, f, indent=4, cls=NumpyEncoder)
        logger.info(f"Hierarchical comparison results saved to: {output_json_path}")
    except Exception as e:
        logger.error(f"Failed to save comparison results JSON: {e}")

    # --- Final Summary Logging ---
    logger.info(f"\n--- Run Summary ---")
    logger.info(f"Total datasets attempted: {len(DATASET_NAMES)}")
    logger.info(f"Successfully processed: {successful_dataset_count}")
    logger.info(f"Failed to process: {len(failed_datasets_summary)}")

    if successful_dataset_count > 0:
        # Calculate average 'ref' accuracy only on successfully processed datasets
        ref_means = []
        for dataset_name, data_content in all_results_dict["datasets"].items():
            if data_content.get('status') != 'error' and 'classifier_metrics' in data_content:
                 if 'ref' in data_content['classifier_metrics'] and 'mean_accuracy' in data_content['classifier_metrics']['ref']:
                    ref_means.append(data_content['classifier_metrics']['ref']['mean_accuracy'])
        
        if ref_means:
             logger.info(f"Average 'ref' classifier mean accuracy across successful datasets: {np.mean(ref_means):.4f}")
    else:
        if len(DATASET_NAMES) > 0 : # Only show this warning if datasets were actually attempted
            logger.warning("No datasets were processed successfully to calculate summary statistics.")
    
    if failed_datasets_summary:
        all_results_dict["failed_datasets_summary"] = failed_datasets_summary # Ensure it's in the dict if not already
        logger.info(f"\n--- Details for Failed Datasets ({len(failed_datasets_summary)}) ---")
        for name, reason in failed_datasets_summary.items():
            logger.warning(f"  Dataset: {name}, Reason: {reason}")
        logger.info(f"Further error details for failed datasets may be found in: {FAILED_LOG_FILE}")
    
    logger.info("Comparison script finished.")
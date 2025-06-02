import os
import sys
import numpy as np
import json
import logging
from glob import glob
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from scipy.stats import ttest_rel
import time

# --- Setup sys.path to find custom modules ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Now import custom modules
from Preprocessing.preprocessing import Preprocessor
from CustomKernelSVC.CustomKernelSVC import CustomKernelSVC

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Configuration ---
RANDOM_STATE = 0
N_SPLITS_CV = 10
P_VALUE_THRESHOLD = 0.05 # For equivalence testing
RESULTS_FILE = "comparison_results.json"
DATASET_DIR = os.path.join(SCRIPT_DIR, "sets")

def load_results(filepath: str) -> dict:
    """Loads existing results from a JSON file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode JSON from {filepath}. Starting with empty results.")
            return {}
    return {}

def save_results(filepath: str, results: dict):
    """Saves results to a JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results successfully saved to {filepath}")
    except IOError as e:
        logger.error(f"Error saving results to {filepath}: {e}")

def get_models():
    """Defines the models to be compared."""
    models = {
        "opt_2d": Pipeline([
            ('preprocessor', Preprocessor()),
            ('svc', CustomKernelSVC(objective_metric='spatial',
                                    svm_kwargs={'random_state': RANDOM_STATE, 'probability': False},
                                    kernel_fit_type='scale'))
        ]),
        "opt_axis": Pipeline([
            ('preprocessor', Preprocessor()),
            ('svc', CustomKernelSVC(objective_metric='axis',
                                    svm_kwargs={'random_state': RANDOM_STATE, 'probability': False},
                                    kernel_fit_type='scale'))
        ]),
        "ref": Pipeline([
            ('preprocessor', Preprocessor()),
            ('svc', SVC(random_state=RANDOM_STATE, probability=False))
        ])
    }
    return models

def main():
    logger.info("Starting model comparison script.")
    
    if not os.path.isdir(DATASET_DIR):
        logger.error(f"Dataset directory not found: {DATASET_DIR}")
        logger.error("Please ensure datasets are generated and available in 'sets/'.")
        logger.error("You might need to run '/StoreSets/store_sets.py' first.")
        return

    dataset_files = glob(os.path.join(DATASET_DIR, "*.npz"))
    if not dataset_files:
        logger.error(f"No .npz dataset files found in {DATASET_DIR}.")
        return

    logger.info(f"Found {len(dataset_files)} datasets to process.")

    all_results = load_results(RESULTS_FILE)
    models = get_models()
    
    # shuffle=True is important for StratifiedKFold robustness if data isn't already randomly ordered
    kf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

    for dataset_path in dataset_files:
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        logger.info(f"--- Processing dataset: {dataset_name} ---")

        if dataset_name in all_results:
            logger.info(f"Results for {dataset_name} already exist in {RESULTS_FILE}. Skipping.")
            continue
        
        try:
            data = np.load(dataset_path)
            X, y = data['X'], data['y']
            
            if X.dtype != np.float32:
                X = X.astype(np.float32)
            if y.dtype != np.int32: # CustomKernelSVC and other parts might expect int32
                y = y.astype(np.int32)

            if X.shape[0] == 0:
                logger.warning(f"Dataset {dataset_name} has 0 samples. Skipping.")
                continue
            if X.shape[0] < N_SPLITS_CV:
                logger.warning(f"Dataset {dataset_name} has {X.shape[0]} samples, less than N_SPLITS_CV={N_SPLITS_CV}. Skipping.")
                continue
            
            unique_classes_y = np.unique(y)
            if len(unique_classes_y) < 2:
                logger.warning(f"Dataset {dataset_name} has only {len(unique_classes_y)} unique class(es). Skipping.")
                continue
            
            class_counts = np.bincount(y[y >= 0]) # Ensure y is non-negative for bincount if necessary
            min_class_count = np.min(class_counts[class_counts > 0]) if len(class_counts[class_counts > 0]) > 0 else 0
            if min_class_count < N_SPLITS_CV:
                logger.warning(f"Dataset {dataset_name} has a class with {min_class_count} samples, less than N_SPLITS_CV={N_SPLITS_CV}. StratifiedKFold may fail. Skipping.")
                continue

        except Exception as e:
            logger.error(f"Error loading or preparing dataset {dataset_name}: {e}")
            continue

        model_scores_dict = {}
        dataset_start_time = time.time()

        for model_name, model_pipeline in models.items():
            logger.info(f"Running {N_SPLITS_CV}-fold CV for model '{model_name}' on dataset '{dataset_name}'...")
            model_start_time = time.time()
            try:
                scores = cross_val_score(model_pipeline, X, y, cv=kf, 
                                         scoring='accuracy', n_jobs=-1, error_score='raise')
                model_scores_dict[model_name] = scores
                logger.info(f"Finished CV for '{model_name}'. Time: {time.time() - model_start_time:.2f}s. Avg Acc: {np.mean(scores):.4f}")
            except Exception as e:
                logger.error(f"Error CV model '{model_name}' on '{dataset_name}': {e}")
                model_scores_dict[model_name] = None

        dataset_results = {
            "accuracy": {},
            "ttest_pvalue": {},
            "equivalent": {}
        }

        for model_name, scores_val in model_scores_dict.items():
            if scores_val is not None and len(scores_val) > 0 :
                mean_acc = np.mean(scores_val)
                std_acc = np.std(scores_val)
                dataset_results["accuracy"][model_name] = f"{mean_acc:.4f} ± {std_acc:.4f}"
            else:
                dataset_results["accuracy"][model_name] = "Error or no scores"
        
        comparisons = [
            ("opt_2d", "ref"),
            ("opt_axis", "ref"),
            ("opt_2d", "opt_axis")
        ]

        for m1_name, m2_name in comparisons:
            comp_key = f"{m1_name} - {m2_name}"
            scores_m1 = model_scores_dict.get(m1_name)
            scores_m2 = model_scores_dict.get(m2_name)

            if scores_m1 is not None and scores_m2 is not None and len(scores_m1) == N_SPLITS_CV and len(scores_m2) == N_SPLITS_CV:
                # Perform t-test only if both models have complete scores for all folds
                try:
                    ttest_result = ttest_rel(scores_m1, scores_m2)
                    # Check for NaN p-value (e.g. if all scores are identical for both models in all folds)
                    if np.isnan(ttest_result.pvalue):
                        pval = 1.0 # Or handle as a special case, e.g. perfectly equivalent
                        logger.warning(f"NaN p-value for {comp_key} on {dataset_name}. Scores might be identical. Treating as p-value=1.0 for equivalence.")
                    else:
                        pval = float(ttest_result.pvalue)
                    
                    dataset_results["ttest_pvalue"][comp_key] = pval
                    dataset_results["equivalent"][comp_key] = bool(pval > P_VALUE_THRESHOLD)
                except Exception as e:
                    logger.error(f"Error during t-test for {comp_key} on dataset {dataset_name}: {e}")
                    dataset_results["ttest_pvalue"][comp_key] = "Error in t-test"
                    dataset_results["equivalent"][comp_key] = "Error in t-test"
            else:
                dataset_results["ttest_pvalue"][comp_key] = "Scores unavailable"
                dataset_results["equivalent"][comp_key] = "Scores unavailable"
            
        all_results[dataset_name] = dataset_results
        save_results(RESULTS_FILE, all_results)
        logger.info(f"Finished processing dataset {dataset_name}. Time: {time.time() - dataset_start_time:.2f}s")
        logger.info("-" * 50)

    logger.info(f"All datasets processed. Final results are in {RESULTS_FILE}")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    main()
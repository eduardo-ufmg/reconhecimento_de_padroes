import multiprocessing
import os
import sys
import numpy as np
import pandas as pd
from typing import cast
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from Preprocessing.preprocessing import Preprocessor
    from CustomKernelSVC.CustomKernelSVC import CustomKernelSVC
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure Preprocessor and SVM classes are correctly defined and accessible.")
    sys.exit(1)

SETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sets'))
DATASET_NAMES = [
    "breast_cancer",
    "iris_binary_0_vs_rest",
    "iris_binary_0_vs_1",
    "digits_binary_0_vs_1",
    "digits_binary_5_vs_rest"
]

def load_npz_dataset(name: str):
    path = os.path.join(SETS_DIR, f"{name}.npz")
    data = np.load(path)
    X = data["X"]
    y = data["y"].ravel()  # Flatten to 1D for sklearn compatibility
    return X, y

class ClassifierScore:
    mean: float
    std: float
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

class StatisticalTestResult:
    t_statistic: float
    p_value: float
    conclusion: str
    def __init__(self, t_statistic: float, p_value: float):
        self.t_statistic = t_statistic
        self.p_value = p_value
        if p_value < 0.05:
            self.conclusion = "Not equivalent"
        else:
            self.conclusion = "Equivalent"

def run_test_on_dataset(dataset_name: str, hs_bounds: tuple[float, float]) -> dict:
    """
    Executes the classifier comparison on a given dataset.
    """
    X, y = load_npz_dataset(dataset_name)
    skf = StratifiedKFold(n_splits=10, shuffle=True)

    # 1. Reference SVM
    pipeline_ref = Pipeline([
        ('preprocessor', Preprocessor()),
        ('svm', SVC())
    ])

    # 2. Custom Kernel SVM
    pipeline_opt = Pipeline([
        ('preprocessor', Preprocessor()),
        ('custom_svm', CustomKernelSVC(
            h_bounds=hs_bounds,
        ))
    ])


    svm_ref_scores = cross_val_score(pipeline_ref, X, y, cv=skf, scoring='accuracy', error_score='raise')
    svm_opt_scores = cross_val_score(pipeline_opt, X, y, cv=skf, scoring='accuracy', error_score='raise')

    t_stat, p_value = cast(tuple[float, float], ttest_rel(svm_ref_scores, svm_opt_scores))

    return {
        'dataset': dataset_name,
        'opt score': float(np.mean(svm_opt_scores)),
        'opt std': float(np.std(svm_opt_scores)),
        'ref score': float(np.mean(svm_ref_scores)),
        'ref std': float(np.std(svm_ref_scores)),
        'T statistic': float(t_stat),
        'p value': float(p_value),
        'conclusion': "Not equivalent" if p_value < 0.05 else "Equivalent"
    }

if __name__ == "__main__":
    hs_param_search_range = (5e-1, 5e0)

    try:
        num_processes = multiprocessing.cpu_count()
    except NotImplementedError:
        num_processes = 4 # Fallback
    print(f"Starting dataset tests in parallel using up to {num_processes} processes...")

    args_for_starmap = [(name, hs_param_search_range) for name in DATASET_NAMES]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results_data = pool.starmap(run_test_on_dataset, args_for_starmap)

    print("All dataset test runs completed.")

    df = pd.DataFrame(results_data)
    summary = df.describe().loc[['mean', 'std'], ['opt score', 'opt std', 'ref score', 'ref std', 'T statistic', 'p value']]

    print("\nResults for each dataset:")
    print(df.to_string(index=False))
    print("\nSummary statistics (mean and std of metrics over datasets):")
    print(summary)

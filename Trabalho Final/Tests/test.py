import multiprocessing
import os
import sys
import numpy as np
import pandas as pd
from typing import cast
from scipy.stats import ttest_rel
from sklearn.datasets import make_classification
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

def run_test(hs_bounds: tuple[float, float], run_index: int) -> tuple[ClassifierScore, ClassifierScore, StatisticalTestResult]:
    """
    Executes a single run of the classifier comparison.
    
    Args:
        hs_search_range (ArrayLike): The hyperparameter search range for the custom SVM.
        run_index (int): The index for the current run, used for seeding.
    
    Returns:
        tuple[ClassifierScore, ClassifierScore, StatisticalTestResult]: Scores and test results.
    """

    np.random.seed(run_index)

    X_original, y_original = make_classification(
        n_samples=1000,
        n_classes=2
    )

    skf = StratifiedKFold(n_splits=10, shuffle=True)

    # 1. Reference SVM
    pipeline_ref = Pipeline([
        ('preprocessor', Preprocessor()),
        ('svm', SVC())
    ])

    # As noted by the joblib's UserWarning, n_jobs=-1 would likely be overridden to n_jobs=1 here
    svm_ref_scores = cross_val_score(pipeline_ref, X_original, y_original, cv=skf, scoring='accuracy')

    # 2. Custom Kernel SVM
    pipeline_opt = Pipeline([
        ('preprocessor', Preprocessor()),
        ('custom_svm', CustomKernelSVC(
            h_bounds=hs_bounds,
        ))
    ])
    svm_opt_scores = cross_val_score(pipeline_opt, X_original, y_original, cv=skf, scoring='accuracy')
    
    t_stat, p_value = cast(tuple[float, float], ttest_rel(svm_ref_scores, svm_opt_scores))

    return (
        ClassifierScore(mean=np.mean(svm_opt_scores).astype(float), std=np.std(svm_opt_scores).astype(float)),
        ClassifierScore(mean=np.mean(svm_ref_scores).astype(float), std=np.std(svm_ref_scores).astype(float)),
        StatisticalTestResult(t_statistic=t_stat, p_value=p_value)
    )

if __name__ == "__main__":
    n_runs = 10
    hs_param_search_range = (5e-1, 5e0)

    try:
        num_processes = multiprocessing.cpu_count()
    except NotImplementedError:
        num_processes = 4 # Fallback
    print(f"Starting {n_runs} test runs in parallel using up to {num_processes} processes...")

    # Prepare a list of argument tuples for starmap
    # Each tuple will be (hs_param_search_range, run_idx)
    # These arguments correspond to the signature: run_test(hs_search_range, run_index)
    args_for_starmap = [(hs_param_search_range, i) for i in range(n_runs)]

    with multiprocessing.Pool(processes=num_processes) as pool:
        # `pool.starmap` calls run_test(*args_tuple) for each tuple in args_for_starmap.
        # So, it will call run_test(hs_param_search_range, 0), run_test(hs_param_search_range, 1), ...
        parallel_results = pool.starmap(run_test, args_for_starmap)
    
    print("All test runs completed.")

    results_data = []
    for opt_score_info, ref_score_info, test_result in parallel_results:
        results_data.append({
            'opt score': opt_score_info.mean,
            'opt std': opt_score_info.std,
            'ref score': ref_score_info.mean,
            'ref std': ref_score_info.std,
            'T statistic': test_result.t_statistic,
            'p value': test_result.p_value,
            'conclusion': test_result.conclusion
        })

    df = pd.DataFrame(results_data)
    summary = df.describe().loc[['mean', 'std'], ['opt score', 'opt std', 'ref score', 'ref std', 'T statistic', 'p value']]

    print("\nResults for each run:")
    print(df.to_string(index=False))
    print("\nSummary statistics (mean and std of metrics over runs):")
    print(summary)

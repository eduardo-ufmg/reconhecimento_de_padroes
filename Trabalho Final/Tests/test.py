import os
import sys

import numpy as np
import pandas as pd

from typing import cast
from numpy.typing import ArrayLike

from scipy.stats import ttest_rel

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Preprocessing.preprocessing import Preprocessor
from SVM.svm import SVM

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

def run_test(hs_search_range: ArrayLike) -> tuple[ClassifierScore, ClassifierScore, StatisticalTestResult]:
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
    svm_ref_scores = cross_val_score(pipeline_ref, X_original, y_original, cv=skf, scoring='accuracy')

    # 2. Custom Kernel SVM
    pipeline_opt = Pipeline([
        ('preprocessor', Preprocessor()),
        ('custom_svm', SVM(
            hs_range=hs_search_range,
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
    hs_param_search_range = np.linspace(5e-1, 5e0, 100)

    results_data = []
    for i in range(n_runs):
        print(f"Running test run {i+1}/{n_runs}...")
        opt_score_info, ref_score_info, test_result = run_test(hs_param_search_range)
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


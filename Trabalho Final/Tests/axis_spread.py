import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import cast

from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

from Preprocessing.preprocessing import preprocess
from Kernel.kernel import kernel, kernel_fit
from AxisSpread.vector_spread import vector_spread, objective_function
from scipy.stats import ttest_rel

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

def run_test() -> tuple[ClassifierScore, ClassifierScore, StatisticalTestResult, float]:

    X, y = make_classification(
        n_samples=1000,
        n_classes=2
    )

    X = preprocess(X, y)

    cov_inv, norm_factor = kernel_fit(X, type='scale')

    hs = np.linspace(5e-1, 5e0, 100)

    spreads = []
    scores = []

    for h in hs:
        K = kernel(X, X, cov_inv, norm_factor, h)
        Q0 = np.sum(K[:, y == 0], axis=1)
        Q1 = np.sum(K[:, y == 1], axis=1)

        Q0C0 = Q0[y == 0]
        Q1C1 = Q1[y == 1]

        spread_Q0C0 = vector_spread(Q0C0)
        spread_Q1C1 = vector_spread(Q1C1)

        if spread_Q0C0 is None or spread_Q1C1 is None:
            score = np.nan
        else:
            score = objective_function(spread_Q0C0, spread_Q1C1)

        spreads.append((spread_Q0C0, spread_Q1C1))
        scores.append(score)

    spreads = np.array(spreads)
    scores = np.array(scores)

    h_opt = hs[np.nanargmax(scores)]

    K_opt = kernel(X, X, cov_inv, norm_factor, h_opt)

    svm_ref = SVC()
    svm_opt = SVC(kernel='precomputed')

    skf = StratifiedKFold(n_splits=10, shuffle=True)

    svm_ref_scores = cross_val_score(svm_ref, X, y, cv=skf, scoring='accuracy')
    svm_opt_scores = cross_val_score(svm_opt, K_opt, y, cv=skf, scoring='accuracy')

    t_stat, p_value = cast(tuple[float, float], ttest_rel(svm_ref_scores, svm_opt_scores))

    return (
        ClassifierScore(mean=np.mean(svm_opt_scores).astype(float), std=np.std(svm_opt_scores).astype(float)),
        ClassifierScore(mean=np.mean(svm_ref_scores).astype(float), std=np.std(svm_ref_scores).astype(float)),
        StatisticalTestResult(t_statistic=t_stat, p_value=p_value),
        h_opt
    )

if __name__ == "__main__":
    n_runs = 10
    results = [run_test() for _ in range(n_runs)]

    df = pd.DataFrame([{
        'opt score': r[0].mean,
        'opt std': r[0].std,
        'ref score': r[1].mean,
        'ref std': r[1].std,
        'T statistic': r[2].t_statistic,
        'p value': r[2].p_value,
        'conclusion': r[2].conclusion,
        'opt h': r[3]
    } for r in results])

    summary = df.describe().T[['mean', 'std']]

    print("Results for each run:")
    print(df.to_string(index=False))
    print("\nSummary statistics:")
    print(summary)

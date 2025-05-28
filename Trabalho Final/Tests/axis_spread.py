import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

if __name__ == "__main__":

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

    plot = False
    if plot:
        plt.figure()
        plt.plot(hs, spreads[:, 0], label='0', linestyle='--')
        plt.plot(hs, spreads[:, 1], label='1', linestyle='--')
        plt.plot(hs, scores, label='Score')
        plt.xlabel('h')
        plt.ylabel('Value')
        plt.title('Spread and Score vs h')
        plt.legend()
        plt.show()


    h_opt = hs[np.nanargmax(scores)]
    print(f'Optimal h: {h_opt:.2g}')

    K_opt = kernel(X, X, cov_inv, norm_factor, h_opt)

    svm_ref = SVC()
    svm_opt = SVC(kernel='precomputed')

    skf = StratifiedKFold(n_splits=10, shuffle=True)

    svm_ref_scores = cross_val_score(svm_ref, X, y, cv=skf)
    svm_opt_scores = cross_val_score(svm_opt, K_opt, y, cv=skf)

    print(f"Reference SVM score: {np.mean(svm_ref_scores):.2g} ± {np.std(svm_ref_scores):.2g}")
    print(f"Optimized SVM score: {np.mean(svm_opt_scores):.2g} ± {np.std(svm_opt_scores):.2g}")

    # Paired t-test
    t_stat, p_value = cast(tuple[float, float], ttest_rel(svm_ref_scores, svm_opt_scores))
    print(f"Paired t-test: t-statistic = {t_stat:.2g}, p-value = {p_value:.2g}")

    if p_value > 0.05:
        print("No statistically significant difference between the models (p > 0.05).")
    else:
        print("Statistically significant difference between the models (p <= 0.05).")
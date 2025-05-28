import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

from Preprocessing.preprocessing import preprocess
from Kernel.kernel import kernel, kernel_fit
from AxisSpread.vector_spread import vector_spread
from ObjectiveFunction.objective_funcion import objective_function

if __name__ == "__main__":

    X, y = make_classification(
        n_samples=1000,
        n_classes=2
    )

    X = preprocess(X, y)

    cov_inv, norm_factor = kernel_fit(X, type='scale')

    hs = np.linspace(1e-3, 5.0, 100)

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

    plt.figure()
    plt.plot(hs, spreads[:, 0], label='0', linestyle='--')
    plt.plot(hs, spreads[:, 1], label='1', linestyle='--')
    plt.plot(hs, scores, label='Score')
    plt.xlabel('h')
    plt.ylabel('Value')
    plt.title('Spread and Score vs h')
    plt.legend()
    plt.show()
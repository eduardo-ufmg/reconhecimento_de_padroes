import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from numpy.typing import NDArray

from Kernel.kernel import kernel, kernel_fit
from AxisSpread.vector_spread import vector_spread
from Preprocessing.preprocessing import preprocess

if __name__ == "__main__":

    X, y = make_classification(
        n_samples=1000,
        n_classes=2
    )

    X = preprocess(X, y)

    cov_inv, norm_factor = kernel_fit(X)

    hs = np.linspace(1e-3, 2.0, 100)

    spreads = []

    for h in hs:
        K = kernel(X, X, cov_inv, norm_factor, h)
        Q0 = np.sum(K[:, y == 0], axis=1)
        Q1 = np.sum(K[:, y == 1], axis=1)

        spread_Q0 = vector_spread(Q0)
        spread_Q1 = vector_spread(Q1)

        spreads.append((spread_Q0, spread_Q1))

    spreads = np.array(spreads)
    plt.plot(hs, spreads[:, 0], label='Spread Q0')
    plt.plot(hs, spreads[:, 1], label='Spread Q1')
    plt.xlabel('Bandwidth (h)')
    plt.ylabel('Spread')
    plt.legend()
    plt.title('Spread vs Bandwidth')
    plt.show()

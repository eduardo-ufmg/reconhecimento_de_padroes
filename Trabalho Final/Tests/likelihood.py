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

    K = kernel(X, X, cov_inv, norm_factor, 1.0)

    plt.imshow(K)
    plt.show()

    Q0 = np.sum(K[:, y == 0], axis=1)
    Q1 = np.sum(K[:, y == 1], axis=1)

    spread_Q0 = vector_spread(Q0)
    spread_Q1 = vector_spread(Q1)

    plt.scatter(Q0, Q1, c=y)
    plt.xlabel(f'Spread: {spread_Q0:.2g}')
    plt.ylabel(f'Spread: {spread_Q1:.2g}')
    plt.show()

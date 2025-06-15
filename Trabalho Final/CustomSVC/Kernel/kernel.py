import numpy as np


def kernel_matrix(
    X1: np.ndarray, X2: np.ndarray | None = None, h: float = 1.0
) -> np.ndarray:
    """
    Compute the Gaussian RBF kernel matrix.

    This function calculates the kernel matrix using the Gaussian (RBF) kernel.
    The formula is:
    K(x, y) = exp(-gamma * ||x - y||^2)
    where gamma = 1 / (2 * h^2), and 'h' is the bandwidth parameter.

    Parameters:
    - X1: np.ndarray, shape (n_samples1, n_features)
        The first set of input data points.
    - X2: np.ndarray, shape (n_samples2, n_features), optional
        The second set of input data points. If None, X2 is set to X1.
    - h: float
        The bandwidth parameter for the kernel. Must be a positive value.

    Returns:
    - K: np.ndarray, shape (n_samples1, n_samples2)
        The computed RBF kernel matrix.

    Raises:
    - ValueError: If h is not positive or if X1 and X2 have mismatched feature dimensions.
    """
    if h <= 0:
        raise ValueError("Bandwidth 'h' must be positive.")
    if X2 is None:
        X2 = X1

    n_features1 = X1.shape[1]
    n_features2 = X2.shape[1]

    if n_features1 != n_features2:
        raise ValueError(
            f"X1 and X2 must have the same number of features, but got {n_features1} and {n_features2}."
        )

    # Calculate squared Euclidean distances efficiently: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    sq_distances = (
        np.sum(X1**2, axis=1)[:, np.newaxis]
        + np.sum(X2**2, axis=1)
        - 2 * np.dot(X1, X2.T)
    )

    # Ensure distances are non-negative before exponentiation
    sq_distances = np.maximum(sq_distances, 0)

    # Standard RBF kernel formula where gamma = 1 / (2 * h^2)
    gamma = 1 / (2 * h**2)
    K = np.exp(-gamma * sq_distances)

    return K

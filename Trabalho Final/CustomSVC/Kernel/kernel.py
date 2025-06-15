import numpy as np
from scipy.stats import multivariate_normal


def kernel_matrix(
    X1: np.ndarray, X2: np.ndarray | None = None, h: float = 1.0, kernel: str = "kde"
) -> np.ndarray:
    """
    Compute the kernel matrix for the given data points X1 and X2 and bandwidth h.

    Parameters:
    - X1: np.ndarray, shape (n_samples1, n_features)
        The first set of input data points.
    - X2: np.ndarray, shape (n_samples2, n_features), optional
        The second set of input data points. If None, X2 is set to X1.
    - h: float
        The bandwidth parameter for the kernel.
    - kernel: str, optional
        The type of kernel to use. Default is 'kde'.
        Supported kernels: 'kde' (Gaussian kernel for KDE), 'pdf' (Multivariate Normal PDF).

    Returns:
    - K: np.ndarray, shape (n_samples1, n_samples2)
        The computed kernel matrix.

    Raises:
    - ValueError: If an unsupported kernel type is provided.
    """
    if X2 is None:
        X2 = X1

    n_samples1, n_features1 = X1.shape
    n_samples2, n_features2 = X2.shape

    if n_features1 != n_features2:
        raise ValueError("X1 and X2 must have the same number of features.")

    if kernel == "kde":
        # Gaussian kernel calculation for KDE
        # Calculate squared Euclidean distances
        sq_distances = (
            np.sum(X1**2, axis=1)[:, np.newaxis]
            + np.sum(X2**2, axis=1)
            - 2 * np.dot(X1, X2.T)
        )

        log_norm_const = n_features1 * (np.log(h) + 0.5 * np.log(2 * np.pi))
        K = np.exp(-sq_distances / (2 * h**2) - log_norm_const)

    elif kernel == "pdf":
        # Multivariate Normal Distribution Probability Density Function
        covariance_matrix = (h**2) * np.cov(
            X2, rowvar=False
        )  # Scale the covariance by h^2
        K = np.zeros((n_samples1, n_samples2))

        for i in range(n_samples2):
            mean_vector = X2[i]
            # Create a multivariate_normal object for the current mean
            mvn = multivariate_normal(mean=mean_vector, cov=covariance_matrix)  # type: ignore
            # Calculate the PDF for all points X1
            K[:, i] = mvn.pdf(X1)

    else:
        raise ValueError(
            f"Unsupported kernel type: '{kernel}'. Supported types are 'kde' and 'pdf'."
        )

    return K

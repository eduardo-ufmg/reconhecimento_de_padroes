import numpy as np
from scipy.stats import multivariate_normal


def kernel_matrix(X: np.ndarray, h: float, kernel: str = "kde") -> np.ndarray:
    """
    Compute the kernel matrix for the given data points X and bandwidth h.

    Parameters:
    - X: np.ndarray, shape (n_samples, n_features)
        The input data points.
    - h: float
        The bandwidth parameter for the kernel.
    - kernel: str, optional
        The type of kernel to use. Default is 'kde'.
        Supported kernels: 'kde' (Gaussian kernel for KDE), 'pdf' (Multivariate Normal PDF).

    Returns:
    - K: np.ndarray, shape (n_samples, n_samples)
        The computed kernel matrix.

    Raises:
    - ValueError: If an unsupported kernel type is provided.
    """
    n_samples, n_features = X.shape
    K = np.zeros((n_samples, n_samples))

    if kernel == "kde":
        # Gaussian kernel calculation for KDE
        # Calculate squared Euclidean distances
        sq_distances = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
        # Apply the Gaussian kernel formula
        K = np.exp(-sq_distances / (2 * h**2))
        # Normalization constant for the Gaussian kernel in KDE context
        K /= (h * np.sqrt(2 * np.pi)) ** n_features

    elif kernel == "pdf":
        # Multivariate Normal Distribution Probability Density Function
        # This assumes h acts as a scaling factor for the covariance matrix
        # For PDF, each K[i,j] would typically be the PDF value of X[j] at X[i]
        # or the PDF value of the difference X[i]-X[j]
        # Here, we'll calculate the PDF of X[i] with respect to X[j] as the mean
        # and a spherical covariance based on h.

        # We need a covariance matrix. For the full covariance matrix,
        # we can use a diagonal covariance matrix scaled by h^2.
        covariance_matrix = (h**2) * np.cov(
            X, rowvar=False
        )  # Scale the covariance by h^2

        # For each pair (i, j), calculate the PDF of X[j] with X[i] as mean
        # and the defined covariance.
        for i in range(n_samples):
            mean_vector = X[i]
            # Create a multivariate_normal object for the current mean
            mvn = multivariate_normal(mean=mean_vector, cov=covariance_matrix)
            # Calculate the PDF for all points X[j]
            K[i, :] = mvn.pdf(X)

    else:
        raise ValueError(
            f"Unsupported kernel type: '{kernel}'. Supported types are 'kde' and 'pdf'."
        )

    return K

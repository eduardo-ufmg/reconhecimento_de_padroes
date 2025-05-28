import numpy as np

from typing import cast
from numpy.typing import NDArray

def kernel(X: NDArray[np.float64], Y: NDArray[np.float64],
           norm_factor: float, cov_inv: NDArray[np.float64],
           h: float) -> NDArray[np.float64]:
    """
    Computes the kernel matrix between two sets of data points.

    Parameters:
    X : NDArray[np.float64]
        First set of data points (shape: [n_samples_X, n_features]).
    Y : NDArray[np.float64]
        Second set of data points (shape: [n_samples_Y, n_features]).
    norm_factor : float
        Normalization factor for the kernel.
    cov_inv : NDArray[np.float64]
        Inverse covariance matrix (shape: [n_features, n_features]).
    h : float
        Bandwidth parameter

    Returns:
    NDArray[np.float64]
        Kernel matrix (shape: [n_samples_X, n_samples_Y]).
    """

    scaled_cov_inv: NDArray[np.float64] = cov_inv / (h ** 2)

    n_features: int = X.shape[1]

    scaled_norm_factor: float = norm_factor / (h ** n_features)

    pdf = mvpdf(X, Y, scaled_norm_factor, scaled_cov_inv)

    pdf_normalized = pdf / np.max(pdf)

    return pdf_normalized

def mvpdf(X: NDArray[np.float64], Y: NDArray[np.float64],
          norm_factor: float, cov_inv: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Computes the multivariate probability density function (PDF) for two sets of data points.
    The PDF is defined as:
    K(X, Y) = norm_factor * exp(-0.5 * (X - Y)^T * cov_inv * (X - Y))

    Parameters:
    X : NDArray[np.float64]
        First set of data points (shape: [n_samples_X, n_features]).
    Y : NDArray[np.float64]
        Second set of data points (shape: [n_samples_Y, n_features]).
    norm_factor : float
        Normalization factor for the pdf.
    cov_inv : NDArray[np.float64]
        Inverse covariance matrix (shape: [n_features, n_features]).
    Returns:
    NDArray[np.float64]
        pdf matrix (shape: [n_samples_X, n_samples_Y]).
    """

    # The exponent term involves (x_i - y_j)^T @ cov_inv @ (x_i - y_j) for each pair of rows.
    # Let C = cov_inv. This quadratic form can be expanded:
    # (x - y)^T @ C @ (x - y) = x^T @ C @ x - 2 * x^T @ C @ y + y^T @ C @ y

    # 1. Calculate x_i^T @ C @ x_i for all x_i in X
    # This results in a 1D array of shape (n_samples_X,).
    # (X @ cov_inv) has shape (n_samples_X, n_features).
    # Element-wise multiplication with X, then sum over features (axis=1).
    # This computes diag(X @ cov_inv @ X.T).
    term_X_C_X: NDArray[np.float64] = np.sum((X @ cov_inv) * X, axis=1)

    # 2. Calculate y_j^T @ C @ y_j for all y_j in Y
    # This results in a 1D array of shape (n_samples_Y,).
    # Similarly, this computes diag(Y @ cov_inv @ Y.T).
    term_Y_C_Y: NDArray[np.float64] = np.sum((Y @ cov_inv) * Y, axis=1)

    # 3. Calculate x_i^T @ C @ y_j for all pairs (x_i, y_j)
    # This results in a 2D array of shape (n_samples_X, n_samples_Y).
    # This is equivalent to X @ cov_inv @ Y.T.
    term_X_C_Y: NDArray[np.float64] = X @ cov_inv @ Y.T

    # 4. Combine terms to get the full quadratic form for all pairs.
    # We need to make term_X_C_X a column vector and term_Y_C_Y a row vector
    # to use broadcasting correctly.
    # term_X_C_X[:, np.newaxis] has shape (n_samples_X, 1)
    # term_Y_C_Y[np.newaxis, :] has shape (1, n_samples_Y)
    # The subtraction and addition are then broadcasted element-wise.
    # quadratic_form_values_ij = term_X_C_X[i] - 2 * term_X_C_Y[i,j] + term_Y_C_Y[j]
    quadratic_form_values: NDArray[np.float64] = (
        term_X_C_X[:, np.newaxis] - 2 * term_X_C_Y + term_Y_C_Y[np.newaxis, :]
    )

    # 5. Apply the exponential and normalization factor
    pdf_matrix: NDArray[np.float64] = norm_factor * np.exp(-0.5 * quadratic_form_values)

    return pdf_matrix
    
def kernel_fit(X: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    """
    Fits the kernel by computing the inverse covariance matrix and normalization factor.

    Parameters:
    X : NDArray[np.float64]
        Data points (shape: [n_samples, n_features]).

    Returns:
    tuple[float, NDArray[np.float64]]
        Normalization factor and Inverse covariance matrix (shape: [n_features, n_features]).
    """
    
    # Compute the covariance matrix
    cov_matrix: NDArray[np.float64] = cast(NDArray[np.float64], np.cov(X, rowvar=False))

    cov_det = np.linalg.det(cov_matrix)
    if not cov_det > 0:
        raise ValueError("Covariance matrix is singular, ill-conditioned, or not positive definite.")

    # Compute the normalization factor
    _, n_features = X.shape
    norm_factor: float = 1 / ((2 * np.pi) ** (n_features / 2) * np.sqrt(cov_det))
    
    # Compute the inverse of the covariance matrix
    cov_inv: NDArray[np.float64] = cast(NDArray[np.float64], np.linalg.inv(cov_matrix))
    
    return norm_factor, cov_inv

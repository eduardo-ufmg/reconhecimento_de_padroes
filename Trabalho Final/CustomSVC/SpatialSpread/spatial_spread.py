import numpy as np


def objective_function_spatial_spread(K_matrix: np.ndarray, y: np.ndarray) -> float:
    """
    Objective function for spatial spread optimization.
    This function computes the spatial spread of the similarity space defined by the kernel matrix K_matrix.
    The spatial spread is defined as the average distance between points in the same class,
    minus the absolute difference between the average distance within classes,
    plus the average distance between points in different classes.

    Parameters:
    - K_matrix: np.ndarray, shape (n_samples, n_samples)
        The kernel matrix representing the similarity space.
    - y: np.ndarray, shape (n_samples,)
        The class labels for the samples.

    Returns:
    - float: The computed spatial spread value.
    """

    # Projects the samples into the similarity space
    Q0 = np.array([np.sum(K_matrix[i, y == 0]) for i in range(K_matrix.shape[0])])
    Q1 = np.array([np.sum(K_matrix[i, y == 1]) for i in range(K_matrix.shape[0])])

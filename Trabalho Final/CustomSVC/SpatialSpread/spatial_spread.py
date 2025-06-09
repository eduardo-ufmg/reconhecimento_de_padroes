import numpy as np


def intra_class_average_distance(C: np.ndarray) -> float:
    """
    Computes the average distance between points in the same class.

    Parameters:
    - C: np.ndarray, shape (n_samples,)
        The samples from the similarity space for a specific class.

    Returns:
    - float: The average distance between points in the same class.
    """
    if len(C) < 2:
        return 0.0  # No distance to compute if less than 2 points
    distances = np.linalg.norm(C[:, np.newaxis] - C[np.newaxis, :], axis=2)
    return np.mean(distances[distances > 0])  # Exclude self-distances


def inter_class_average_distance(C0: np.ndarray, C1: np.ndarray) -> float:
    """
    Computes the average distance between points in different classes.

    Parameters:
    - C0: np.ndarray, shape (n_samples_class_0,)
        The samples from the similarity space for class 0.
    - C1: np.ndarray, shape (n_samples_class_1,)
        The samples from the similarity space for class 1.

    Returns:
    - float: The average distance between points in different classes.
    """
    if len(C0) == 0 or len(C1) == 0:
        return 0.0  # No distance to compute if one class is empty
    distances = np.linalg.norm(C0[:, np.newaxis] - C1[np.newaxis, :], axis=2)
    return np.mean(distances)


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

    # Extracts the samples from the similarity space for each class
    C0 = np.hstack((Q0[y == 0], Q1[y == 0]))
    C1 = np.hstack((Q0[y == 1], Q1[y == 1]))

    # Computes the average distance between points in the same class
    intra_class_average_distance_0 = intra_class_average_distance(C0)
    intra_class_average_distance_1 = intra_class_average_distance(C1)

    intra_class_average_distance_mean = (
        intra_class_average_distance_0 + intra_class_average_distance_1
    ) / 2

    intra_class_average_distance_abs_diff = abs(
        intra_class_average_distance_0 - intra_class_average_distance_1
    )

    inter_class_average_distance_value = inter_class_average_distance(C0, C1)

    # Computes the spatial spread value
    spatial_spread_value = (
        intra_class_average_distance_mean
        - intra_class_average_distance_abs_diff
        + inter_class_average_distance_value
    )

    return spatial_spread_value

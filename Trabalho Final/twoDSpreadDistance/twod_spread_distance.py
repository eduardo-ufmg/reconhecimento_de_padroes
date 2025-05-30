import numpy as np
from numpy.typing import ArrayLike, NDArray

def mean_sample_dist(Q0: ArrayLike, Q1: ArrayLike) -> float | None:
    """
    Computes the mean distance between all pairs of points given by two 1D arrays (axes).
    
    Args:
        Q0 (ArrayLike): 1D array of x-coordinates.
        Q1 (ArrayLike): 1D array of y-coordinates.
    
    Returns:
        float: The mean distance between all pairs of points, or None if the input is invalid.
    """
    Q0 = np.asarray(Q0)
    Q1 = np.asarray(Q1)
    
    if Q0.ndim != 1 or Q1.ndim != 1 or Q0.shape[0] != Q1.shape[0]:
        return None
    
    n = Q0.shape[0]
    if n < 2:
        return None

    Q = np.stack((Q0, Q1), axis=1)
    dist_matrix = np.linalg.norm(Q[:, np.newaxis] - Q[np.newaxis, :], axis=-1)
    upper_triangle_indices = np.triu_indices(n, k=1)
    distances = dist_matrix[upper_triangle_indices]
    mean_distance = np.mean(distances)
    return mean_distance

def mean_class_dist(Q0: ArrayLike, Q1: ArrayLike, y: ArrayLike) -> float | None:
    """
    Computes the mean distance between all pairs of points in opposite classes.
    Args:
        Q0 (ArrayLike): 1D array of x-coordinates for class 0.
        Q1 (ArrayLike): 1D array of y-coordinates for class 1.
        y (ArrayLike): 1D array of class labels (0 or 1).
    Returns:
        float: The mean distance between all pairs of points in opposite classes, or None if the input is invalid.
    """
    Q0 = np.asarray(Q0)
    Q1 = np.asarray(Q1)
    y = np.asarray(y)

    if Q0.ndim != 1 or Q1.ndim != 1 or y.ndim != 1:
        return None

    if Q0.shape[0] != Q1.shape[0] or Q0.shape[0] != y.shape[0]:
        return None

    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]

    if len(class_0_indices) < 2 or len(class_1_indices) < 2:
        return None

    Q0_class_0 = Q0[class_0_indices]
    Q1_class_1 = Q1[class_1_indices]

    return mean_sample_dist(Q0_class_0, Q1_class_1)

def objective_function(Q0: NDArray[np.float64], Q1: NDArray[np.float64], y: NDArray[np.int32]) -> float:
    """
    The objective function is the sum of the mean distances between points in opposite classes and the mean distance
    between points in the same class. It is designed to encourage separation between classes while maintaining spread within classes.
    Args:
        Q0 (NDArray[np.float64]): 1D array of x-coordinates for class 0.
        Q1 (NDArray[np.float64]): 1D array of y-coordinates for class 1.
        y (NDArray[np.int32]): 1D array of class labels (0 or 1).
    Returns:
        float: The computed objective value, or NaN if the spread cannot be computed for either group.
    """
    mean_dist_classes = mean_class_dist(Q0, Q1, y)
    
    if mean_dist_classes is None:
        return np.nan
    
    # Calculate the mean distance within each class
    Q0C0, Q1C0 = Q0[y == 0], Q1[y == 0]
    Q0C1, Q1C1 = Q0[y == 1], Q1[y == 1]

    mean_dist_C0 = mean_sample_dist(Q0C0, Q1C0)
    mean_dist_C1 = mean_sample_dist(Q0C1, Q1C1)

    if mean_dist_C0 is None or mean_dist_C1 is None:
        return np.nan
    
    # Combine the mean distances
    mean_dist_combined = (mean_dist_C0 + mean_dist_C1) / 2.0

    # Compute the standard deviation of the mean distances
    std_dev = np.std([mean_dist_C0, mean_dist_C1])

    # Calculate the final objective value
    objective_value = mean_dist_classes + mean_dist_combined - std_dev

    return objective_value.astype(float)  # Ensure standard Python float
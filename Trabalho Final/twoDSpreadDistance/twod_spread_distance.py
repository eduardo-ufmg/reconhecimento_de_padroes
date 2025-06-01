import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, cdist

def mean_intra_class_dist(Q0: NDArray[np.float32], Q1: NDArray[np.float32]) -> np.float32:
    """
    Computes the mean distance between all pairs of points in a class, given by two 1D arrays.
    Args:
        Q0 (NDArray[np.float32]): 1D array of x-coordinates.
        Q1 (NDArray[np.float32]): 1D array of y-coordinates.
    Returns:
        np.float32: The mean distance of distances within the class.
    Raises:
        ValueError: If the input is invalid.
    """

    if Q0.ndim != 1 or Q1.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")
    if Q0.shape[0] != Q1.shape[0]:
        raise ValueError("Inputs must have the same length")
    num_points = Q0.shape[0]
    if num_points < 2:
        raise ValueError("Need at least two points to form a pair and calculate distance/std dev")

    # Combine Q0 and Q1 into a 2D array of points (N, 2)
    points = np.vstack((Q0, Q1)).T
    
    # pdist computes all pairwise distances within the set of points
    # It returns a 1D condensed distance matrix (upper triangle of the distance matrix)
    distances = pdist(points, metric='euclidean')
    
    # For num_points >= 2, 'distances' array will not be empty.
    # Example: 2 points yield 1 distance, 3 points yield 3 distances.
    
    mean_dist = np.mean(distances)
    
    return np.float32(mean_dist)

def mean_inter_class_dist(Q0: NDArray[np.float32], Q1: NDArray[np.float32], y: NDArray[np.int32]) -> np.float32:
    """
    Computes the mean distance between all pairs of points of opposite classes.
    Args:
        Q0 (NDArray[np.float32]): 1D array of x-coordinates
        Q1 (NDArray[np.float32]): 1D array of y-coordinates
        y (NDArray[np.int32]): 1D array of class labels (0 or 1).
    Returns:
        np.float32: The mean distance between all pairs of points of opposite classes.
    Raises:
        ValueError: If the input is invalid.
    """

    if not (Q0.ndim == 1 and Q1.ndim == 1 and y.ndim == 1):
        raise ValueError("All inputs must be 1D")
    if not (Q0.shape[0] == Q1.shape[0] == y.shape[0]):
        raise ValueError("All inputs must have the same length")
    if Q0.shape[0] == 0:
        raise ValueError("No points provided")

    # Combine Q0 and Q1 into a 2D array of all points
    points = np.vstack((Q0, Q1)).T
    
    # Separate points based on class labels
    points_c0 = points[y == 0]
    points_c1 = points[y == 1]

    if points_c0.shape[0] == 0 or points_c1.shape[0] == 0:
        raise ValueError("One or both classes are empty. Cannot compute inter-class distances.")

    # cdist computes distances between each point in points_c0 and each point in points_c1
    # Returns a 2D array of shape (num_points_c0, num_points_c1)
    inter_class_distances = cdist(points_c0, points_c1, metric='euclidean')
    
    # If both classes are non-empty, inter_class_distances will have size > 0.
    mean_dist = np.mean(inter_class_distances)
    
    return np.float32(mean_dist)

def objective_function(Q0: NDArray[np.float32], Q1: NDArray[np.float32], y: NDArray[np.int32]) -> np.float32:
    """
    The objective function is the mean distances between points in opposite classes plus the mean distance between points in the same class
    minus the standard deviation of the mean distances within each class.
    It is designed to encourage separation between classes while maintaining spread within classes.
    Args:
        Q0 (NDArray[np.float32]): 1D array of x-coordinates
        Q1 (NDArray[np.float32]): 1D array of y-coordinates
        y (NDArray[np.int32]): 1D array of class labels (0 or 1).
    Returns:
        np.float32: The computed objective value, or NaN if the spread cannot be computed for either group.
    Raises:
        ValueError: For input errors.
    """

    if not (Q0.ndim == 1 and Q1.ndim == 1 and y.ndim == 1):
        raise ValueError("All inputs must be 1D")
    if not (Q0.shape[0] == Q1.shape[0] == y.shape[0]):
        raise ValueError("All inputs must have the same length")
    if Q0.shape[0] == 0:
        raise ValueError("No points provided")

    # Calculate mean inter-class distance
    # This returns None if one or both classes are empty (e.g., y contains only one unique value).
    try:
        val_m_inter_cd = mean_inter_class_dist(Q0, Q1, y)
    except ValueError:
        return np.float32(np.nan)

    # Separate points by class for intra-class calculations
    q0_c0 = Q0[y == 0]
    q1_c0 = Q1[y == 0]
    
    q0_c1 = Q0[y == 1]
    q1_c1 = Q1[y == 1]

    # Calculate mean intra-class distance for class 0
    try:
        val_m_intra_c0 = mean_intra_class_dist(q0_c0, q1_c0)
    except ValueError:
        return np.float32(np.nan)  # Cannot compute spread for class 0
    
    # Calculate mean intra-class distance for class 1
    try:
        val_m_intra_c1 = mean_intra_class_dist(q0_c1, q1_c1)
    except ValueError:
        return np.float32(np.nan)  # Cannot compute spread for class 1
    
    # Calculate the mean of the intra-class distances
    val_m_intra = (val_m_intra_c0 + val_m_intra_c1) / 2.0

    # Calculate the standard deviation of the mean distances within each class.
    val_std_intra = np.std([val_m_intra_c0, val_m_intra_c1])

    # Objective value is the mean inter-class distance plus the mean intra-class distance minus the standard deviation of intra-class distances
    objective_value = val_m_inter_cd + val_m_intra - val_std_intra
    
    return np.float32(objective_value)

import numpy as np
from numpy.typing import ArrayLike, NDArray

def intern_sample_dist(Q0: ArrayLike, Q1: ArrayLike) -> tuple[np.float32, np.float32] | None:
    """
    Computes the mean distance between all pairs of points in a class, given by two 1D arrays.
    Args:
        Q0 (ArrayLike): 1D array of x-coordinates.
        Q1 (ArrayLike): 1D array of y-coordinates.
    Returns:
        tuple[np.float32, np.float32] | None: A tuple containing the mean distance and the standard deviation of distances within the class,
                                    or None if the input is invalid.
    """

def mean_class_dist(Q0: ArrayLike, Q1: ArrayLike, y: ArrayLike) -> np.float32 | None:
    """
    Computes the mean distance between all pairs of points in opposite classes.
    Args:
        Q0 (ArrayLike): 1D array of x-coordinates
        Q1 (ArrayLike): 1D array of y-coordinates
        y (ArrayLike): 1D array of class labels (0 or 1).
    Returns:
        np.float32: The mean distance between all pairs of points in opposite classes, or None if the input is invalid.
    """
    

def objective_function(Q0: NDArray[np.float32], Q1: NDArray[np.float32], y: NDArray[np.int32]) -> np.float32 | None:
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
    """

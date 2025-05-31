import numpy as np
from numpy.typing import NDArray

def vector_spread(Qx: NDArray[np.float32]) -> np.float32 | None:
    """
    Computes a score indicating how evenly spaced the values in a vector are.
    A higher score (closer to 1.0) means more evenly spaced (benefited).
    A lower score (closer to 0.0) means more clustered or unevenly spaced (penalized).

    The method involves:
    1. Sorting the vector.
    2. Calculating the differences between adjacent sorted elements.
    3. Computing the standard deviation of these differences (std_diff).
    4. The score is 1.0 / (1.0 + std_diff).

    Args:
        Qx (ndarray): A list, tuple, or NumPy array of numbers.

    Returns:
        np.float32: A score between 0.0 (exclusive, in practice) and 1.0 (inclusive).
               - 1.0 indicates perfect even spacing (or a single element vector).
               - Values closer to 0.0 indicate more clustered/uneven spacing.
               Returns None if the vector is empty, contains NaN values, cannot be processed,
               or if the internal standard deviation calculation results in NaN or Inf.
    """
    try:
        numeric_array = np.asarray(Qx, dtype=np.float32)
    except ValueError:
        # Handles cases where conversion to np.float32 array fails (e.g., list of strings)
        return None

    # Check for NaN values in the input array
    if np.any(np.isnan(numeric_array)):
        return None

    n = numeric_array.size

    if n == 0:
        # Empty vector, evenness undefined
        return None
    if n == 1:
        # Single element is considered perfectly evenly spaced
        return np.float32(1.0)

    # Sort the array to calculate differences between consecutive elements
    sorted_array = np.sort(numeric_array)

    # Calculate differences between adjacent elements
    # For n >= 2, 'differences' will have at least n-1 >= 1 elements.
    differences = np.diff(sorted_array)

    # Calculate the standard deviation of these differences.
    # Using ddof=0 (population standard deviation) because 'differences'
    # represents the complete set of spacings for the given vector.
    # If 'differences' has only one element (i.e., n=2 originally),
    # np.std([d], ddof=0) is 0.0, correctly indicating perfect evenness for two points.
    std_of_differences = np.std(differences, ddof=0)

    # Calculate the mean of the differences
    mean_of_differences = np.mean(differences)

    # If std_of_differences is NaN or Inf (e.g., if input had Inf values leading to this),
    # return None for a clean API.
    if np.isnan(std_of_differences) or np.isinf(std_of_differences):
        return None


    # Calculate the std_score
    # Since we want a score that is higher for more evenly spaced vectors,
    # we favor lower standard deviations.
    std_score = 1.0 / (1.0 + std_of_differences)

    # Calculate the mean score
    # We normalize the mean of differences by the maximum absolute difference
    # to ensure the score is between 0.0 and 1.0.
    max_diff = np.max(np.abs(differences))
    mean_score = mean_of_differences / max_diff if max_diff != 0 else 0.0
    
    # Combine the scores
    score = std_score * mean_score
    
    return np.float32(score) # Ensure standard Python np.float32

def objective_function(Q0: NDArray[np.float32], Q1: NDArray[np.float32], y: NDArray[np.int32]) -> np.float32 | float:
    """
    Calculates an objective value based on the spread of vectors in two groups, conditioned on class labels.
    This function computes the spread (using `vector_spread`) of vectors in `Q0` for class 0 and in `Q1` for class 1,
    then combines these spreads into a single objective value. If the spread cannot be computed for either group,
    the function returns NaN.
    Args:
        Q0 (NDArray[np.float32]): Array of vectors corresponding to the first set of features.
        Q1 (NDArray[np.float32]): Array of vectors corresponding to the second set of features.
        y (NDArray[np.int32]): Array of integer class labels (0 or 1) for each sample.
    Returns:
        np.float32: The computed objective value, or NaN if the spread cannot be computed for either group.
    """

    Q0C0, Q1C1 = Q0[y == 0], Q1[y == 1]

    spread_Q0C0 = vector_spread(Q0C0)
    spread_Q1C1 = vector_spread(Q1C1)

    if spread_Q0C0 is None or spread_Q1C1 is None:
        return np.nan
    
    mean = (spread_Q0C0 + spread_Q1C1) / 2.0
    diff = np.abs(spread_Q0C0 - spread_Q1C1)

    return mean - diff
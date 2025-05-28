import numpy as np
from numpy.typing import ArrayLike
import warnings

def vector_spread(data_vector: ArrayLike) -> float | None:
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
        data_vector (array-like): A list, tuple, or NumPy array of numbers.

    Returns:
        float: A score between 0.0 (exclusive, in practice) and 1.0 (inclusive).
               - 1.0 indicates perfect even spacing (or a single element vector).
               - Values closer to 0.0 indicate more clustered/uneven spacing.
               Returns None if the vector is empty, contains NaN values, cannot be processed,
               or if the internal standard deviation calculation results in NaN or Inf.
    """
    try:
        numeric_array = np.asarray(data_vector, dtype=float)
    except ValueError:
        # Handles cases where conversion to float array fails (e.g., list of strings)
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
        return 1.0

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
    
    return float(score) # Ensure standard Python float

def objective_function(score0: float, score1: float) -> float:
    """
    Computes the objective function value based on the scores of two classes.
    
    The objective function is defined as the mean of the two scores
    minus the absolute difference between the two scores.
    Parameters:
        score0 (float): Score for class 0.
        score1 (float): Score for class 1.
    Returns:
        float: The computed objective function value.
    """
    if np.isnan(score0) or np.isnan(score1):
        return np.nan
    
    mean = (score0 + score1) / 2.0
    abs_diff = abs(score0 - score1)

    return mean - abs_diff
    
if __name__ == "__main__":
    
    # Ordened scores from higher to lower
    v0 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    v1 = [0.0, 0.0, 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8]
    v2 = [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
    v3 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    print("Spread of v0:", vector_spread(v0))
    print("Spread of v1:", vector_spread(v1))
    print("Spread of v2:", vector_spread(v2))
    print("Spread of v3:", vector_spread(v3))
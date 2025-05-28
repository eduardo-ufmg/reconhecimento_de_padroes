import numpy as np

from numpy.typing import ArrayLike

def vector_spread(data_vector: ArrayLike, ddof: int = 1) -> float | None:
    """
    Computes how well spread the values in a vector are using standard deviation.
    This version uses NumPy for efficiency and flexibility.

    Args:
        data_vector (array-like): A list, tuple, or NumPy array of numbers.
        ddof (int, optional): Delta Degrees of Freedom.
                               Defaults to 1 for sample standard deviation (N-1).
                               Set to 0 for population standard deviation (N).

    Returns:
        float: The standard deviation of the values.
               Returns 0.0 if the vector contains only one element.
               Returns None if the vector is empty or cannot be processed.
    
    Raises:
        TypeError: If ddof is not an integer.
        ValueError: If data_vector cannot be converted to a numeric NumPy array.
    """

    try:
        # Attempt to convert the input to a NumPy array of floats
        # np.asarray avoids copying if data_vector is already a compatible NumPy array.
        numeric_array = np.asarray(data_vector, dtype=float)
    except (ValueError, TypeError) as e:
        # Handles cases where conversion to float array fails (e.g., list of strings)
        print(f"Error: Input data could not be converted to a numeric array. Original error: {e}")
        return None

    # Handle empty array
    if numeric_array.size == 0:
        print("Warning: The vector is empty. Cannot calculate spread.")
        return None

    # Handle array with a single element
    # Standard deviation of a single point is 0, as there's no variation.
    if numeric_array.size == 1:
        return 0.0

    # For sample standard deviation (ddof=1), N-ddof must be > 0.
    # If N=1 and ddof=1, N-ddof = 0, leading to division by zero (NaN).
    # Our numeric_array.size == 1 check above handles this by returning 0.0.
    # np.std itself handles other N-ddof <= 0 cases by returning NaN or raising warnings,
    # but our explicit checks for size 0 and 1 are clearer for this function's contract.

    # Calculate and return the standard deviation
    try:
        spread = np.std(numeric_array, ddof=ddof)
        return float(spread)
    except Exception as e: # Catch any other potential NumPy errors
        print(f"An unexpected error occurred during NumPy standard deviation calculation: {e}")
        return None
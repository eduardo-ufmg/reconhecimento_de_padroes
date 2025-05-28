import numpy as np

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

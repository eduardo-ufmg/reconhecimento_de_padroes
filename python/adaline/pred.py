import numpy as np

def pred(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Predict the target values using the weights and input observations.

    Parameters:
        X (np.ndarray): Input observations (n_samples, n_features)
        w (np.ndarray): Weights (n_features + 1)

    Returns:
        Y_pred (np.ndarray): Predicted target values (n_samples)
    """
    # Add a bias term (column of ones) to the input matrix
    augX = np.hstack((X, np.ones((X.shape[0], 1))))
    
    # Compute the predicted target values
    Y_pred = augX @ w
    
    return Y_pred

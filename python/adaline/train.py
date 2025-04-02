import numpy as np

def train_pseudoinv(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
  """
  Compute the weights that better fit the known data using the pseudoinverse.
  
  Parameters:
    X (np.ndarray): Input observations (n_samples, n_features)
    Y (np.ndarray): Target values (n_samples)
    
  Returns:
    w (np.ndarray): Weights (n_features + 1)
  """
  # Add a bias term (column of ones) to the input matrix
  augX = np.hstack((X, np.ones((X.shape[0], 1))))
  
  # Compute the pseudoinverse of the augmented matrix
  pseudoinv = np.linalg.pinv(augX)
  
  # Compute the weights
  w = pseudoinv @ Y
  
  return w

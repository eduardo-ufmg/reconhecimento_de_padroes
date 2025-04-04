import numpy as np

def train(X1: np.ndarray, y1: int, X2: np.ndarray, y2: int
          ) -> tuple[float, np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, int]:
  """
  Train a Gaussian mixture model for two classes.
  
  Parameters:
    X1 (np.ndarray): The data points for class 1.
    y1 (int): The label for class 1.
    X2 (np.ndarray): The data points for class 2.
    y2 (int): The label for class 2.
  
  Returns:
    tuple: A tuple containing the parameters of the Gaussian mixture model:
      - float: Prior probability of class 1.
      - np.ndarray: Mean vector of class 1.
      - np.ndarray: Covariance matrix of class 1.
      - int: Number of data points in class 1.
      - float: Prior probability of class 2.
      - np.ndarray: Mean vector of class 2.
      - np.ndarray: Covariance matrix of class 2.
      - int: Number of data points in class 2.
  """
  N1 = X1.shape[0]
  N2 = X2.shape[0]
  N = N1 + N2

  P1 = N1 / N
  P2 = N2 / N

  M1 = np.mean(X1, axis=0)
  M2 = np.mean(X2, axis=0)

  S1 = np.cov(X1.T)
  S2 = np.cov(X2.T)

  return P1, M1, S1, N1, P2, M2, S2, N2

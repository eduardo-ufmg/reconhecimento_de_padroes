import numpy as np
from scipy.stats import multivariate_normal

def gaussian_multivariate(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
  """
  Compute the multivariate Gaussian probability density function for a given set of data points.

  Parameters:
    X (np.ndarray): The data points for which to compute the multivariate Gaussian PDF.
    mu (np.ndarray): The mean vector of the multivariate Gaussian distribution.
    sigma (np.ndarray): The covariance matrix of the multivariate Gaussian distribution.

  Returns:
    np.ndarray: The multivariate Gaussian PDF values for each data point in X.
  """
  return multivariate_normal.pdf(X, mean=mu, cov=sigma)

def pred_multivariate_normal(X: np.ndarray,
                             y1: int, P1: float, M1: np.ndarray, S1: np.ndarray,
                             y2: int, P2: float, M2: np.ndarray, S2: np.ndarray) -> np.ndarray:
  """
  Predict the class of a given set of data points based on two multivariate Gaussian distributions.
  The function computes the multivariate Gaussian probability density functions for two classes
  and returns the predicted class for each data point.

  Parameters:
    X (np.ndarray): The data points to predict.
    y1 (int): The label for class 1.
    P1 (float): The prior probability of class 1.
    M1 (np.ndarray): The mean vector of class 1.
    S1 (np.ndarray): The covariance matrix of class 1.
    y2 (int): The label for class 2.
    P2 (float): The prior probability of class 2.
    M2 (np.ndarray): The mean vector of class 2.
    S2 (np.ndarray): The covariance matrix of class 2.

  Returns:
    np.ndarray: The predicted class for each data point in X.
  """

  G1 = gaussian_multivariate(X, M1, S1)
  G2 = gaussian_multivariate(X, M2, S2)

  PC1 = P1 * G1
  PC2 = P2 * G2

  # Predict the class based on the maximum posterior probability
  pred = np.empty(X.shape[0])
  pred[PC1 > PC2] = y1
  pred[PC1 < PC2] = y2

  return pred

def pred_multivariate_gausmix(X: np.ndarray,
                             X1: np.ndarray, y1: int, P1: float, S1: np.ndarray, N1: int,
                             X2: np.ndarray, y2: int, P2: float, S2: np.ndarray, N2: int,
                             params: tuple) -> np.ndarray:
  """
  Predict the class of a given set of data points based on two Gaussian mixture models.
  Each class is modeled as a mixture of Gaussians centered at each training data point with the class's covariance matrix.
  
  Parameters:
    X (np.ndarray): The data points to predict.
    X1 (np.ndarray): The data points for class 1.
    y1 (int): The label for class 1.
    P1 (float): The prior probability of class 1.
    S1 (np.ndarray): The covariance matrix of class 1.
    N1 (int): The number of data points in class 1.
    X2 (np.ndarray): The data points for class 2.
    y2 (int): The label for class 2.
    P2 (float): The prior probability of class 2.
    S2 (np.ndarray): The covariance matrix of class 2.
    N2 (int): The number of data points in class 2.
    params (tuple): 
    
  Returns:
    np.ndarray: The predicted class for each data point in X.
  """

  # Calculate Gaussian mixture PDF for class 1
  G1 = np.zeros(X.shape[0])
  for x in X1:
    G1 += multivariate_normal.pdf(X, mean=x, cov=S1)
  G1 /= N1  # Average the contributions

  # Calculate Gaussian mixture PDF for class 2
  G2 = np.zeros(X.shape[0])
  for x in X2:
    G2 += multivariate_normal.pdf(X, mean=x, cov=S2)
  G2 /= N2  # Average the contributions

  # Compute posterior probabilities
  PC1 = P1 * G1
  PC2 = P2 * G2

  # Predict the class with higher posterior probability
  pred = np.where(PC1 > PC2, y1, y2)
  return pred
  

def pred_multivariate(X: np.ndarray, 
                      X1: np.ndarray, y1: int, P1: float, M1: np.ndarray, S1: np.ndarray, N1: int,
                      X2: np.ndarray, y2: int, P2: float, M2: np.ndarray, S2: np.ndarray, N2: int,
                      type: str, params: tuple=None) -> np.ndarray:
  if type == "normal":
    return pred_multivariate_normal(X, y1, P1, M1, S1, y2, P2, M2, S2)
  elif type == "gaussian_mix":
    return pred_multivariate_gausmix(X, X1, y1, P1, S1, N1, X2, y2, P2, S2, N2, params)
  elif type == "kde":
    raise NotImplementedError("KDE prediction is not implemented yet.")
  else:
    raise ValueError(f"Unknown prediction type: {type}. Supported types are 'normal', 'sum_of_gaussians', and 'kde'.")

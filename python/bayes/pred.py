import numpy as np
from scipy.stats import norm, multivariate_normal

def pred(X: np.ndarray, X1: np.ndarray, y1: int, X2: np.ndarray, y2: int) -> np.ndarray:
  """
  Predict the class of a given set of data points based on two Gaussian distributions.
  The function computes the Gaussian probability density functions for two classes
  and returns the predicted class for each data point.

  Parameters:
    X (np.ndarray): The data points to predict.
    X1 (np.ndarray): The data points for class 1.
    y1 (int): The label for class 1.
    X2 (np.ndarray): The data points for class 2.
    y2 (int): The label for class 2.

  Returns:
    np.ndarray: The predicted class for each data point in X.
  """

  N1 = X1.shape[0]
  N2 = X2.shape[0]
  N = N1 + N2

  P1 = N1 / N
  P2 = N2 / N

  M1 = np.mean(X1)
  M2 = np.mean(X2)

  S1 = np.std(X1)
  S2 = np.std(X2)

  G1 = gaussian(X, M1, S1)
  G2 = gaussian(X, M2, S2)

  PC1 = P1 * G1
  PC2 = P2 * G2

  # Predict the class based on the maximum posterior probability
  pred = np.empty(X.shape[0])
  pred[PC1 > PC2] = y1
  pred[PC1 < PC2] = y2

  return pred

def gaussian(X: np.ndarray, mu: float, sigma: float) -> np.ndarray:
  """
  Compute the Gaussian probability density function for a given set of data points.

  Parameters:
    X (np.ndarray): The data points for which to compute the Gaussian PDF.
    mu (float): The mean of the Gaussian distribution.
    sigma (float): The standard deviation of the Gaussian distribution.

  Returns:
    np.ndarray: The Gaussian PDF values for each data point in X.
  """
  return norm.pdf(X, loc=mu, scale=sigma)

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

def pred_multivariate(X: np.ndarray, X1: np.ndarray, y1: int, X2: np.ndarray, y2: int) -> np.ndarray:
  """
  Predict the class of a given set of data points based on two multivariate Gaussian distributions.
  The function computes the multivariate Gaussian probability density functions for two classes
  and returns the predicted class for each data point.

  Parameters:
    X (np.ndarray): The data points to predict.
    X1 (np.ndarray): The data points for class 1.
    y1 (int): The label for class 1.
    X2 (np.ndarray): The data points for class 2.
    y2 (int): The label for class 2.

  Returns:
    np.ndarray: The predicted class for each data point in X.
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

  G1 = gaussian_multivariate(X, M1, S1)
  G2 = gaussian_multivariate(X, M2, S2)

  PC1 = P1 * G1
  PC2 = P2 * G2

  # Predict the class based on the maximum posterior probability
  pred = np.empty(X.shape[0])
  pred[PC1 > PC2] = y1
  pred[PC1 < PC2] = y2

  return pred

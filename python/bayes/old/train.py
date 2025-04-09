import numpy as np
from typing import NamedTuple

class GaussianParams(NamedTuple):
  prior: float
  mean: np.ndarray
  cov: np.ndarray
  n_samples: int
  label: int

def train(X1: np.ndarray, y1: int, X2: np.ndarray, y2: int) -> tuple[GaussianParams, GaussianParams]:
  """
  Train Gaussian models for two classes with regularized covariance matrices.
  
  Parameters:
    X1 (np.ndarray): Data points for class 1.
    y1 (int): Label for class 1.
    X2 (np.ndarray): Data points for class 2.
    y2 (int): Label for class 2.
  
  Returns:
    tuple: GaussianParams for both classes.
  """
  def compute_params(X: np.ndarray, y: int) -> GaussianParams:
    n = X.shape[0]
    mean = np.mean(X, axis=0)
    cov = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])  # Regularization
    label = y
    return GaussianParams(prior=n/(n + X2.shape[0]), mean=mean, cov=cov, n_samples=n, label=label)

  return compute_params(X1, y1), compute_params(X2, y2)

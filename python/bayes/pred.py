import numpy as np
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity
from typing import Literal

from bayes.train import GaussianParams

def _vectorized_gaussian_mixture(X: np.ndarray, components: np.ndarray, cov: np.ndarray) -> np.ndarray:
  """Vectorized computation of Gaussian mixture PDF."""
  k = X.shape[1]
  cov_inv = np.linalg.inv(cov)
  coeff = 1 / ((2 * np.pi) ** (k/2) * np.sqrt(np.linalg.det(cov)))
  
  # Compute all pairwise differences
  diffs = X[:, np.newaxis, :] - components  # Shape (n_test, n_components, n_features)
  
  # Compute quadratic terms using einsum
  quad_terms = np.einsum('...ki,ij,...kj->...k', diffs, cov_inv, diffs)
  exponents = np.exp(-0.5 * quad_terms)
  
  return coeff * exponents.mean(axis=1)

def pred_multivariate_normal(X: np.ndarray, params1: GaussianParams, params2: GaussianParams) -> np.ndarray:
  """Predict using multivariate Gaussian distributions."""
  pdf1 = multivariate_normal(params1.mean, params1.cov).pdf(X)
  pdf2 = multivariate_normal(params2.mean, params2.cov).pdf(X)
  posterior1 = params1.prior * pdf1
  posterior2 = params2.prior * pdf2
  return np.where(posterior1 > posterior2, params1.label, params2.label)

def pred_gaussian_mixture(X: np.ndarray, params1: GaussianParams, X1: np.ndarray, 
             params2: GaussianParams, X2: np.ndarray) -> np.ndarray:
  """Predict using Gaussian mixture models."""
  pdf1 = _vectorized_gaussian_mixture(X, X1, params1.cov)
  pdf2 = _vectorized_gaussian_mixture(X, X2, params2.cov)
  posterior1 = params1.prior * pdf1
  posterior2 = params2.prior * pdf2
  return np.where(posterior1 > posterior2, params1.label, params2.label)

def pred_kde(X: np.ndarray, params1: GaussianParams, X1: np.ndarray,
      params2: GaussianParams, X2: np.ndarray, bandwidth: float = 0.2) -> np.ndarray:
  """Predict using Kernel Density Estimation."""
  kde1 = KernelDensity(bandwidth=bandwidth).fit(X1)
  kde2 = KernelDensity(bandwidth=bandwidth).fit(X2)
  log_prob1 = kde1.score_samples(X) + np.log(params1.prior)
  log_prob2 = kde2.score_samples(X) + np.log(params2.prior)
  return np.where(log_prob1 > log_prob2, params1.label, params2.label)

def predict(
  X: np.ndarray,
  X1: np.ndarray, params1: GaussianParams,
  X2: np.ndarray, params2: GaussianParams,
  method: Literal["normal", "gaussian_mix", "kde"] = "normal"
) -> np.ndarray:
  """Unified prediction interface."""
  if method == "normal":
    return pred_multivariate_normal(X, params1, params2)
  elif method == "gaussian_mix":
    return pred_gaussian_mixture(X, params1, X1, params2, X2)
  elif method == "kde":
    return pred_kde(X, params1, X1, params2, X2)
  else:
    raise ValueError(f"Invalid method: {method}. Choose 'normal', 'gaussian_mix', or 'kde'.")

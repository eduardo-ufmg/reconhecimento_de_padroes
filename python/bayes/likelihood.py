import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal

def likelihood(X, args0, args1, method):
  """
  Compute the likelihood to each class for each sample.
  Args:
    X: Input features (n_samples, n_features)
    args0: Model parameters for class 0
    args1: Model parameters for class 1
    method: 'normal', 'gaussian_mix', or 'kde'
  Returns:
    Q0: Likelihood for class 0 (n_samples,)
    Q1: Likelihood for class 1 (n_samples,)
  """

  methods = {
      'normal': likelihood_normal,
      'gaussian_mix': likelihood_gaussian_mix,
      'kde': likelihood_kde
  }

  if method not in methods:
    raise ValueError(f"Unknown method: {method}")
    
  return methods[method](X, args0, args1)

def likelihood_normal(X, params0, params1):
  """Compute likelihood using Gaussian distributions."""
  mean0, cov0, _ = params0
  mean1, cov1, _ = params1

  # Compute the likelihood for each class
  Q0 = multivariate_normal.pdf(X, mean=mean0, cov=cov0)
  Q1 = multivariate_normal.pdf(X, mean=mean1, cov=cov1)

  return Q0, Q1

def likelihood_gaussian_mix(X, params0, params1):
  raise NotImplementedError("Gaussian Mixture likelihood not implemented.")

def likelihood_kde(X, params0, params1):
  raise NotImplementedError("Kernel Density Estimation likelihood not implemented.")

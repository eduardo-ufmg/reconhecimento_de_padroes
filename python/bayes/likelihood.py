import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

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
  mean0, cov0, prior0 = params0
  mean1, cov1, prior1 = params1
  
  log_likelihood0 = multivariate_normal.logpdf(X, mean=mean0, cov=cov0, allow_singular=True)
  log_likelihood1 = multivariate_normal.logpdf(X, mean=mean1, cov=cov1, allow_singular=True)
  
  log_posterior0 = np.log(prior0) + log_likelihood0
  log_posterior1 = np.log(prior1) + log_likelihood1

  return log_posterior0, log_posterior1

def likelihood_gaussian_mix(X, params0, params1):
  """Compute likelihood using Gaussian Mixture Models."""
  gmm0, prior0 = params0
  gmm1, prior1 = params1
  
  log_likelihood0 = gmm0.score_samples(X)
  log_likelihood1 = gmm1.score_samples(X)
  
  log_posterior0 = np.log(prior0) + log_likelihood0
  log_posterior1 = np.log(prior1) + log_likelihood1

  return log_posterior0, log_posterior1

def likelihood_kde(X, params0, params1):
  """Compute likelihood using Kernel Density Estimators."""
  kde0, prior0 = params0
  kde1, prior1 = params1
  
  log_likelihood0 = kde0.score_samples(X)
  log_likelihood1 = kde1.score_samples(X)
  
  log_posterior0 = np.log(prior0) + log_likelihood0
  log_posterior1 = np.log(prior1) + log_likelihood1

  return log_posterior0, log_posterior1

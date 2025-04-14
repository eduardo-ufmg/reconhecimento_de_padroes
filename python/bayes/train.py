import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

def train(X0, X1, method):
  """
  Train a Bayesian classifier using the specified method.
  
  Args:
    X0: Features for class 0 (n_samples, n_features)
    X1: Features for class 1 (n_samples, n_features)
    method: 'normal', 'gaussian_mix', or 'kde'
  
  Returns:
    Tuple of model parameters for each class.
  """
  methods = {
    'normal': train_normal,
    'gaussian_mix': train_gaussian_mix,
    'kde': train_kde
  }
  if method not in methods:
    raise ValueError(f"Unknown method: {method}")
  return methods[method](X0, X1)

def train_normal(X0, X1):
  """Train Gaussian classifiers with mean and covariance."""
  # Input validation
  if len(X0) == 0 or len(X1) == 0:
    raise ValueError("Training data cannot be empty.")
  
  mean0 = np.mean(X0, axis=0)
  cov0 = np.cov(X0, rowvar=False)
  prior0 = len(X0) / (len(X0) + len(X1))
  
  mean1 = np.mean(X1, axis=0)
  cov1 = np.cov(X1, rowvar=False)
  prior1 = len(X1) / (len(X0) + len(X1))
  
  return (mean0, cov0, prior0), (mean1, cov1, prior1)

def train_gaussian_mix(X0, X1):
  """Train Gaussian Mixture Models with 2 components."""
  # Ensure data is sufficient for components
  n_components = 2
  if len(X0) < n_components:
    raise ValueError(f"X0 has insufficient samples for {n_components} components.")
  if len(X1) < n_components:
    raise ValueError(f"X1 has insufficient samples for {n_components} components.")
  
  gmm0 = GaussianMixture(n_components=n_components).fit(X0)
  gmm1 = GaussianMixture(n_components=n_components).fit(X1)
  prior0 = len(X0) / (len(X0) + len(X1))
  prior1 = len(X1) / (len(X0) + len(X1))
  
  return (gmm0, prior0), (gmm1, prior1)

def train_kde(X0, X1):
  """Train Kernel Density Estimators with adaptive bandwidth."""
  def silverman_bandwidth(X):
    n = len(X)
    if n == 0:
      return 0.5  # Fallback
    std = np.mean(np.std(X, axis=0))
    return 1.06 * std * (n ** (-1/5))
  
  bw0 = silverman_bandwidth(X0) if len(X0) > 0 else 0.5
  bw1 = silverman_bandwidth(X1) if len(X1) > 0 else 0.5
  
  kde0 = KernelDensity(kernel='gaussian', bandwidth=bw0).fit(X0)
  kde1 = KernelDensity(kernel='gaussian', bandwidth=bw1).fit(X1)
  prior0 = len(X0) / (len(X0) + len(X1))
  prior1 = len(X1) / (len(X0) + len(X1))
  
  return (kde0, prior0), (kde1, prior1)
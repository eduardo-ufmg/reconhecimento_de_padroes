import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

def train(X0, y0, X1, y1, method):
  if method == 'normal':
    return train_normal(X0, y0, X1, y1)
  elif method == 'gaussian_mix':
    return train_gaussian_mix(X0, y0, X1, y1)
  elif method == 'kde':
    return train_kde(X0, y0, X1, y1)
  else:
    raise ValueError("Unknown method: {}".format(method))

def train_normal(X0, y0, X1, y1):
  mean0 = np.mean(X0, axis=0)
  cov0 = np.cov(X0, rowvar=False)
  prior0 = X0.shape[0] / (X0.shape[0] + X1.shape[0])
  mean1 = np.mean(X1, axis=0)
  cov1 = np.cov(X1, rowvar=False)
  prior1 = X1.shape[0] / (X0.shape[0] + X1.shape[0])
  return (mean0, cov0, prior0), (mean1, cov1, prior1)

def train_gaussian_mix(X0, y0, X1, y1):
  gmm0 = GaussianMixture(n_components=1).fit(X0)
  gmm1 = GaussianMixture(n_components=1).fit(X1)
  prior0 = X0.shape[0] / (X0.shape[0] + X1.shape[0])
  prior1 = X1.shape[0] / (X0.shape[0] + X1.shape[0])
  return (gmm0, prior0), (gmm1, prior1)

def train_kde(X0, y0, X1, y1):
  kde0 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X0)
  kde1 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X1)
  prior0 = X0.shape[0] / (X0.shape[0] + X1.shape[0])
  prior1 = X1.shape[0] / (X0.shape[0] + X1.shape[0])
  return (kde0, prior0), (kde1, prior1)

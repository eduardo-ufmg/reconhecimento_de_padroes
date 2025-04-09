import numpy as np
from typing import NamedTuple
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

def train(X1, y1, X2, y2, method):
  if method == 'normal':
    return train_normal(X1, y1, X2, y2)
  elif method == 'gaussian_mix':
    return train_gaussian_mix(X1, y1, X2, y2)
  elif method == 'kde':
    return train_kde(X1, y1, X2, y2)
  else:
    raise ValueError("Unknown method: {}".format(method))
  
def train_normal(X1, y1, X2, y2):
  raise NotImplementedError("Normal distribution training is not implemented yet.")

def train_gaussian_mix(X1, y1, X2, y2):
  raise NotImplementedError("Gaussian mixture model training is not implemented yet.")

def train_kde(X1, y1, X2, y2):
  raise NotImplementedError("Kernel density estimation training is not implemented yet.")

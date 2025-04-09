import numpy as np
from typing import NamedTuple
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

def pred(X, args1, args2, method):
  if method == 'normal':
    return predict_normal(X, args1, args2)
  elif method == 'gaussian_mix':
    return predict_gaussian_mix(X, args1, args2)
  elif method == 'kde':
    return predict_kde(X, args1, args2)
  else:
    raise ValueError("Unknown method: {}".format(method))
  
def predict_normal(X, normalParams1, normalParams2):
  raise NotImplementedError("Normal distribution prediction is not implemented yet.")

def predict_gaussian_mix(X, gmixParams1, gmixParams2):
  raise NotImplementedError("Gaussian mixture model prediction is not implemented yet.")

def predict_kde(X, kdeParams1, kdeParams2):
  raise NotImplementedError("Kernel density estimation prediction is not implemented yet.")


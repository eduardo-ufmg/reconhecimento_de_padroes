import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

def pred(X, args0, args1, method):
  if method == 'normal':
    return predict_normal(X, args0, args1)
  elif method == 'gaussian_mix':
    return predict_gaussian_mix(X, args0, args1)
  elif method == 'kde':
    return predict_kde(X, args0, args1)
  else:
    raise ValueError("Unknown method: {}".format(method))

def predict_normal(X, normalParams0, normalParams1):
  mean0, cov0, prior0 = normalParams0
  mean1, cov1, prior1 = normalParams1
  log_likelihood0 = multivariate_normal.logpdf(X, mean=mean0, cov=cov0)
  log_likelihood1 = multivariate_normal.logpdf(X, mean=mean1, cov=cov1)
  log_posterior0 = np.log(prior0) + log_likelihood0
  log_posterior1 = np.log(prior1) + log_likelihood1
  predictions = np.argmax(np.vstack((log_posterior0, log_posterior1)), axis=0)
  return predictions

def predict_gaussian_mix(X, gmixParams0, gmixParams1):
  gmm0, prior0 = gmixParams0
  gmm1, prior1 = gmixParams1
  log_likelihood0 = gmm0.score_samples(X)
  log_likelihood1 = gmm1.score_samples(X)
  log_posterior0 = np.log(prior0) + log_likelihood0
  log_posterior1 = np.log(prior1) + log_likelihood1
  predictions = np.argmax(np.vstack((log_posterior0, log_posterior1)), axis=0)
  return predictions

def predict_kde(X, kdeParams0, kdeParams1):
  kde0, prior0 = kdeParams0
  kde1, prior1 = kdeParams1
  log_likelihood0 = kde0.score_samples(X)
  log_likelihood1 = kde1.score_samples(X)
  log_posterior0 = np.log(prior0) + log_likelihood0
  log_posterior1 = np.log(prior1) + log_likelihood1
  predictions = np.argmax(np.vstack((log_posterior0, log_posterior1)), axis=0)
  return predictions

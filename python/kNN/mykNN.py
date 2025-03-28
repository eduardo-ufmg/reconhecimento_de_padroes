import numpy as np

from scipy.spatial.distance import cdist
from typing import Tuple

def mykNN_batch(Xt: np.ndarray, X: np.ndarray, Y: np.ndarray, k: int, h: float=None) -> np.ndarray:
  """
    Weighted k-nearest neighbors classifier
    Xt: target data, column vector of row vectors (samples, features)
    X: known data, column vector of row vectors (samples, features)
    Y: known labels, raw vector of integer labels (samples) expected labels are -1 and 1
    k: number of neighbors
    h: covariance factor
    returns
      Yt: predicted labels, raw vector of integer labels (samples) -1 or 1
    Weights are given by the multivariate normal distribution pdf
    with mean at the neighbor and bandwidth h
  """

  distances_sq = cdist(Xt, X, 'sqeuclidean')
  
  # Find indices of the k nearest neighbors for each target point
  indices = np.argpartition(distances_sq, k, axis=1)[:, :k]
  
  # Extract squared distances of the k neighbors using advanced indexing
  rows = np.arange(Xt.shape[0])[:, np.newaxis]
  knn_dist_sq = distances_sq[rows, indices]
  
  if h is None:
    weights = np.ones_like(knn_dist_sq)
  else:
    h = max(h, 1e-10)  # Ensure h is not zero to avoid division by zero
    # Calculate weights using the kernel (multivariate normal PDF without normalization)
    weights = np.exp(-knn_dist_sq / (2 * h ** 2))
  
  # Retrieve the labels of the k nearest neighbors
  knn_labels = Y[indices]
  
  # Compute the weighted sum of labels
  weighted_sums = np.sum(weights * knn_labels, axis=1)
  
  # Determine the predicted labels based on the sign of the weighted sum
  Yt = np.where(weighted_sums >= 0, 1, -1)
  
  return Yt

def to_characteristic_space(Xt: np.ndarray, X: np.ndarray, Y: np.ndarray, k: int, h: float) -> Tuple[np.ndarray, np.ndarray]:
  """
    Convert data to characteristic space
    Xt: target data, column vector of row vectors (samples, features)
    X: known data, column vector of row vectors (samples, features)
    Y: known labels, raw vector of integer labels (samples) expected labels are -1 and 1
    k: number of neighbors
    h: covariance factor
    returns
      Q1: for each target sample, the weighted sum of the k nearest neighbors of class 1 (label 1)
      Q2: for each target sample, the weighted sum of the k nearest neighbors of class 2 (label -1)
    Q1 and Q2 are the characteristic space coordinates
  """

  # Compute squared Euclidean distances between all target and training points
  distances_sq = cdist(Xt, X, 'sqeuclidean')
  
  # Find indices of the k nearest neighbors for each target point
  indices = np.argpartition(distances_sq, k, axis=1)[:, :k]
  
  # Extract squared distances of the k neighbors using advanced indexing
  rows = np.arange(Xt.shape[0])[:, np.newaxis]
  knn_dist_sq = distances_sq[rows, indices]
  
  if h is None:
    weights = np.ones_like(knn_dist_sq)
  else:
    h = max(h, 1e-10) # Ensure h is not zero to avoid division by zero
    # Calculate weights using the kernel (multivariate normal PDF)
    weights = np.exp(-knn_dist_sq / (2 * h ** 2))
  
  # Retrieve the labels of the k nearest neighbors
  knn_labels = Y[indices]
  
  # Create masks for class 1 and -1
  mask_class1 = (knn_labels == 1)
  mask_class2 = (knn_labels == -1)
  
  # Compute weighted sums for each class
  Q1 = (weights * mask_class1).sum(axis=1)
  Q2 = (weights * mask_class2).sum(axis=1)
  
  return Q1, Q2

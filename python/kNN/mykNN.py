import numpy as np
from scipy.spatial.distance import cdist

def mykNN_batch(Xt: np.ndarray, X: np.ndarray, Y: np.ndarray, k: int, h: float) -> np.ndarray:
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
  
  # Calculate weights using the kernel (multivariate normal PDF without normalization)
  weights = np.exp(-knn_dist_sq / (2 * h ** 2))
  
  # Retrieve the labels of the k nearest neighbors
  knn_labels = Y[indices]
  
  # Compute the weighted sum of labels
  weighted_sums = np.sum(weights * knn_labels, axis=1)
  
  # Determine the predicted labels based on the sign of the weighted sum
  Yt = np.where(weighted_sums >= 0, 1, -1)
  
  return Yt
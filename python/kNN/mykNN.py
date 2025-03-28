import numpy as np

def mykNN_batch(Xt: np.ndarray, X: np.ndarray, Y: np.ndarray, k: int, h: float) -> np.ndarray:
  """
  Weighted k-nearest neighbors classifier
  
  Parameters:
    Xt: target data, numpy array of shape (n_samples_t, n_features)
    X: known data, numpy array of shape (n_samples, n_features)
    Y: known labels, numpy array of shape (n_samples, 1) or (n_samples,)
    k: number of neighbors to use
    h: bandwidth of the kernel
    
  Returns:
    Yt: predicted labels, numpy array of shape (n_samples_t, 1)
    
  Weights are computed using the multivariate normal distribution PDF
  (ignoring the constant factor) as:
    weight = exp(-d^2 / (2 * h^2))
  where d is the Euclidean distance between the target and neighbor.
  """
  # Ensure Y is a 1D array for easier indexing if needed
  Y = Y.flatten()
  
  n_t = Xt.shape[0]
  
  # Compute squared Euclidean distances between each Xt and each X
  # Using the identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
  Xt_norm_sq = np.sum(Xt**2, axis=1, keepdims=True)  # (n_t, 1)
  X_norm_sq = np.sum(X**2, axis=1, keepdims=True).T     # (1, n)
  dists_sq = Xt_norm_sq + X_norm_sq - 2 * Xt.dot(X.T)    # (n_t, n)
  
  # Find the indices of the k nearest neighbors for each target sample.
  # argpartition gives k smallest entries but not sorted.
  neighbor_idx = np.argpartition(dists_sq, kth=k-1, axis=1)[:, :k]
  
  # Gather the corresponding squared distances and labels for these neighbors.
  neighbor_dists_sq = np.take_along_axis(dists_sq, neighbor_idx, axis=1)
  neighbor_labels = Y[neighbor_idx]  # shape (n_t, k)
  
  # Compute weights using the kernel function (ignoring constant factor).
  weights = np.exp(- neighbor_dists_sq / (2 * h**2))  # shape (n_t, k)
  
  # For each target sample, accumulate the weights for each label.
  pred_labels = np.empty(n_t, dtype=Y.dtype)
  for i in range(n_t):
    # Get labels and weights for the i-th target sample.
    labels_i = neighbor_labels[i]
    weights_i = weights[i]
    
    # Use np.unique to sum weights for each distinct label.
    unique_labels, inverse = np.unique(labels_i, return_inverse=True)
    weighted_sums = np.zeros_like(unique_labels, dtype=np.float64)
    # Sum weights for each unique label.
    np.add.at(weighted_sums, inverse, weights_i)
    
    # Predict the label with the maximum weight.
    pred_labels[i] = unique_labels[np.argmax(weighted_sums)]
      
  return pred_labels.reshape(-1, 1)

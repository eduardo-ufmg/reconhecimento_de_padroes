import numpy as np

def preparekNN(x, complete_set, k, h):
  features = complete_set[:, :-1]
  squared_distances = np.sum((features - x) ** 2, axis=1)
  sorted_indices = np.argsort(squared_distances)
  top_k_indices = sorted_indices[:k]
  squared_distances_top_k = squared_distances[top_k_indices]
  class_labels = complete_set[top_k_indices, -1]
  n_features = x.shape[0]
  normalization = 1.0 / np.sqrt((2 * np.pi * h) ** n_features)
  weights = normalization * np.exp(-0.5 * squared_distances_top_k / h)
  return weights, class_labels

def mykNN(x, complete_set, k, h):
  weights, class_labels = preparekNN(x, complete_set, k, h)
  weighted_sum = np.dot(weights, class_labels)
  return np.sign(weighted_sum)

def to_characteristic_space(x, complete_set, k, h):
  kernel_vals, class_labels = preparekNN(x, complete_set, k, h)
  Q1 = np.sum(kernel_vals[class_labels == 1])
  Q2 = np.sum(kernel_vals[class_labels != 1])
  return Q1, Q2

def preparekNN_batch(X, complete_set, k, h):
  features = complete_set[:, :-1]
  squared_distances = np.sum((X[:, np.newaxis] - features) ** 2, axis=2)
  top_k_indices = np.argpartition(squared_distances, k, axis=1)[:, :k]
  batch_distances = np.take_along_axis(squared_distances, top_k_indices, axis=1)
  batch_labels = complete_set[top_k_indices, -1]
  n_features = X.shape[1]
  normalization = 1.0 / np.sqrt((2 * np.pi * h) ** n_features)
  batch_weights = normalization * np.exp(-0.5 * batch_distances / h)
  return batch_weights, batch_labels

def mykNN_batch(X, complete_set, k, h):
  batch_weights, batch_labels = preparekNN_batch(X, complete_set, k, h)
  weighted_sums = np.sum(batch_weights * batch_labels, axis=1)
  return np.sign(weighted_sums)

def to_characteristic_space_batch(X, complete_set, k, h):
  batch_kernel_vals, batch_labels = preparekNN_batch(X, complete_set, k, h)
  Q1 = np.sum(batch_kernel_vals * (batch_labels == 1), axis=1)
  Q2 = np.sum(batch_kernel_vals * (batch_labels != 1), axis=1)
  return Q1, Q2

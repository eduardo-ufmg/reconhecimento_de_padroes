import numpy as np

from common.distance import euclidean_distance, kernel_function

def preparekNN(x, complete_set, k):
  _, n_features = complete_set.shape
  n_features -= 1  # Last column is label
  
  distances = np.array([euclidean_distance(x, point[:n_features]) for point in complete_set])
  sorted_indices = np.argsort(distances)
  sorted_distances = distances[sorted_indices][:k]
  class_labels = complete_set[sorted_indices[:k], -1]
  
  return sorted_distances, sorted_indices[:k], class_labels

def to_characteristic_space(x, complete_set, k, h):
  _, sorted_indices, class_labels = preparekNN(x, complete_set, k)
  Q1, Q2 = 0.0, 0.0
  
  for idx, label in zip(sorted_indices, class_labels):
    mu = complete_set[idx, :-1]
    kernel_val = kernel_function(x, h, mu)
    if label == 1:
      Q1 += kernel_val
    else:
      Q2 += kernel_val
  
  return Q1, Q2

def mykNN(x, complete_set, k, h):
  _, sorted_indices, class_labels = preparekNN(x, complete_set, k)
  weights = np.zeros(k)
  
  for i, idx in enumerate(sorted_indices):
    mu = complete_set[idx, :-1]
    weights[i] = kernel_function(x, h, mu)
  
  return np.sign(np.sum(weights * class_labels))

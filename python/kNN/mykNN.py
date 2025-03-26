import numpy as np

# Function to prepare k-Nearest Neighbors (kNN) for a single query point
def preparekNN(x, complete_set, k, h):
  # Extract features (all columns except the last one) from the dataset
  features = complete_set[:, :-1]
  # Compute squared Euclidean distances between the query point and all points in the dataset
  squared_distances = np.sum((features - x) ** 2, axis=1)
  # Find the indices of the k nearest neighbors
  top_k_indices = np.argpartition(squared_distances, k)[:k]
  # Get the squared distances of the k nearest neighbors
  squared_distances_top_k = squared_distances[top_k_indices]
  # Extract the class labels of the k nearest neighbors
  class_labels = complete_set[top_k_indices, -1]
  # Calculate the dimensionality of the feature space
  n_features = x.shape[0]
  # Compute the normalization factor for the Gaussian kernel
  normalization = 1.0 / ((2 * np.pi * h**2) ** (n_features / 2))
  # Compute the weights using the Gaussian kernel
  weights = normalization * np.exp(-0.5 * squared_distances_top_k / (h**2))
  return weights, class_labels

# Function to classify a single query point using kNN
def mykNN(x, complete_set, k, h):
  # Prepare kNN by computing weights and class labels
  weights, class_labels = preparekNN(x, complete_set, k, h)
  # Compute the weighted sum of class labels
  weighted_sum = np.dot(weights, class_labels)
  # Return the sign of the weighted sum as the predicted class
  return np.sign(weighted_sum)

# Function to map a single query point to a characteristic space
def to_characteristic_space(x, complete_set, k, h):
  # Prepare kNN by computing kernel values and class labels
  kernel_vals, class_labels = preparekNN(x, complete_set, k, h)
  # Compute the sum of kernel values for class 1
  Q1 = np.sum(kernel_vals[class_labels == 1])
  # Compute the sum of kernel values for other classes
  Q2 = np.sum(kernel_vals[class_labels != 1])
  return Q1, Q2

# Function to prepare kNN for a batch of query points
def preparekNN_batch(X, complete_set, k, h):
  # Extract features (all columns except the last one) from the dataset
  features = complete_set[:, :-1]
  # Compute squared Euclidean distances between each query point and all points in the dataset
  squared_distances = np.sum((X[:, np.newaxis] - features) ** 2, axis=2)
  # Find the indices of the k nearest neighbors for each query point
  top_k_indices = np.argpartition(squared_distances, k, axis=1)[:, :k]
  # Get the squared distances of the k nearest neighbors for each query point
  batch_distances = np.take_along_axis(squared_distances, top_k_indices, axis=1)
  # Extract the class labels of the k nearest neighbors for each query point
  batch_labels = complete_set[top_k_indices, -1]
  # Calculate the dimensionality of the feature space
  n_features = X.shape[1]
  # Compute the normalization factor for the Gaussian kernel
  normalization = 1.0 / ((2 * np.pi * h**2) ** (n_features / 2))
  # Compute the weights using the Gaussian kernel for each query point
  batch_weights = normalization * np.exp(-0.5 * batch_distances / (h**2))
  return batch_weights, batch_labels

# Function to classify a batch of query points using kNN
def mykNN_batch(X, complete_set, k, h):
  # Prepare kNN by computing weights and class labels for the batch
  batch_weights, batch_labels = preparekNN_batch(X, complete_set, k, h)
  # Compute the weighted sum of class labels for each query point
  weighted_sums = np.sum(batch_weights * batch_labels, axis=1)
  # Return the sign of the weighted sums as the predicted classes
  return np.sign(weighted_sums)

# Function to map a batch of query points to a characteristic space
def to_characteristic_space_batch(X, complete_set, k, h):
  # Prepare kNN by computing kernel values and class labels for the batch
  batch_kernel_vals, batch_labels = preparekNN_batch(X, complete_set, k, h)
  # Compute the sum of kernel values for class 1 for each query point
  Q1 = np.sum(batch_kernel_vals * (batch_labels == 1), axis=1)
  # Compute the sum of kernel values for other classes for each query point
  Q2 = np.sum(batch_kernel_vals * (batch_labels != 1), axis=1)
  return Q1, Q2

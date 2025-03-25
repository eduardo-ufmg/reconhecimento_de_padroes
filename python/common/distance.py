import numpy as np

def euclidean_distance(x, y):
  return np.linalg.norm(x - y)

def kernel_function(x, h, mu):
  n = x.shape[0]
  K = h * np.eye(n)
  diff = x - mu
  value = (1 / (np.sqrt((2 * np.pi)**n * np.linalg.det(K)))) * np.exp(-0.5 * diff.T @ np.linalg.inv(K) @ diff)
  return value

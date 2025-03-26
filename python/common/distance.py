import numpy as np

# Function to calculate the Euclidean distance between two points x and y
def euclidean_distance(x, y):
  return np.linalg.norm(x - y)

# Function to compute a kernel function value
# Parameters:
# - x: Input vector
# - h: Bandwidth parameter (scalar)
# - mu: Mean vector
def kernel_function(x, h, mu):
  n = x.shape[0]  # Dimension of the input vector
  inv_h2 = 1 / (h**2)  # Precompute 1/h^2 for efficiency
  diff = x - mu  # Difference between input vector and mean
  # Compute the kernel value using the multivariate Gaussian formula
  value = (inv_h2**(n / 2) / (np.sqrt((2 * np.pi)**n))) * np.exp(-0.5 * inv_h2 * np.dot(diff, diff))
  return value

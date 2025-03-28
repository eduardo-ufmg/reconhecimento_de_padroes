import numpy as np

from sklearn.datasets import make_blobs, make_moons
from typing import Tuple

def generate_dataset(dataset_type: str, n_samples: int, noise: float, random_state: int=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """
    Generate a dataset of type given by dataset_type with n_samples total samples and added noise
    dataset_type: type of dataset to generate (blobs, spirals, xor, moons)
    n_samples: total number of samples to generate
    noise: amount of noise to add to the dataset
    random_state: random state for reproducibility
    return: tuple of (X1, Y1, X2, Y2)
      X1: matrix of input features for class 1 (n_samples, n_features)
      Y1: vector of target classes for class 1 (n_samples)
      X2: matrix of input features for class 2 (n_samples, n_features)
      Y2: vector of target classes for class 2 (n_samples)
    Classes are -1 and 1
    Input is a column vector of row vectors
  """

  if dataset_type == "blobs":
    # Generate a dataset of blobs
    X, Y = make_blobs(n_samples=n_samples, centers=2, cluster_std=noise, random_state=random_state)

    # Convert the target classes to -1 and 1
    Y[Y == 0] = -1
    Y[Y == 1] = 1

    # Split the dataset into two classes
    X1 = X[Y == -1]
    Y1 = Y[Y == -1]
    X2 = X[Y == 1]
    Y2 = Y[Y == 1]

  elif dataset_type == "spirals":
    raise NotImplementedError("Spiral dataset generation is not implemented yet.")

  elif dataset_type == "xor":
    # Generate the XOR dataset
    centers1 = np.array([[-1, -1], [1, 1]])
    X1, Y1 = make_blobs(n_samples=n_samples // 2, centers=centers1, cluster_std=noise, random_state=random_state)
    Y1.fill(-1)

    centers2 = np.array([[-1, 1], [1, -1]])
    X2, Y2 = make_blobs(n_samples=n_samples // 2, centers=centers2, cluster_std=noise, random_state=random_state)
    Y2.fill(1)

  elif dataset_type == "moons":
    # Generate the moons dataset
    X, Y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    # Convert the target classes to -1 and 1
    Y[Y == 0] = -1
    Y[Y == 1] = 1

    # Split the dataset into two classes
    X1 = X[Y == -1]
    Y1 = Y[Y == -1]
    X2 = X[Y == 1]
    Y2 = Y[Y == 1]

  else:
    raise ValueError("Invalid dataset type. Choose from 'blobs', 'spirals', or 'xor'.")

  return X1, Y1, X2, Y2

def generate_grid(X: np.ndarray, step: float=0.01) -> np.ndarray:
  """
    Generate a grid of points for plotting decision boundaries.
    X: matrix of input features (n_samples, n_features)
    step: step size for the grid
    return
      G: column vector of row vectors
  """

  # Create a grid of points
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
  G = np.c_[xx.ravel(), yy.ravel()]

  return G

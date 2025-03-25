import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.neighbors import NearestNeighbors
from matplotlib.colors import ListedColormap
import os

# ---------------------------- Core Algorithms ----------------------------
def euclidean_distance(x, y):
  return np.linalg.norm(x - y)

def kernel_function(x, h, mu):
  n = x.shape[0]
  K = h * np.eye(n)
  diff = x - mu
  value = (1 / (np.sqrt((2 * np.pi)**n * np.linalg.det(K)))) * np.exp(-0.5 * diff.T @ np.linalg.inv(K) @ diff)
  return value

def preparekNN(x, complete_set, k):
  n_samples, n_features = complete_set.shape
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

# ---------------------------- Dataset Generation ----------------------------
def generate_dataset(dataset_type, n_samples, noise):
  np.random.seed(0)
  n_per_class = n_samples // 2
  
  if dataset_type == 'blobs':
    data1 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], n_per_class) + noise * np.random.randn(n_per_class, 2)
    data2 = np.random.multivariate_normal([-2, -2], [[0.5, 0], [0, 0.5]], n_per_class) + noise * np.random.randn(n_per_class, 2)
    labels1 = np.ones(n_per_class)
    labels2 = -np.ones(n_per_class)
  
  elif dataset_type == 'spirals':
    t = np.linspace(0, 4*np.pi, n_per_class)
    data1 = np.column_stack([t*np.cos(t), t*np.sin(t)]) + noise * np.random.randn(n_per_class, 2)
    data2 = np.column_stack([-t*np.cos(t), -t*np.sin(t)]) + noise * np.random.randn(n_per_class, 2)
    labels1 = np.ones(n_per_class)
    labels2 = -np.ones(n_per_class)
  
  elif dataset_type == 'moons':
    data, labels = make_moons(n_samples, noise=noise, random_state=0)
    data1 = data[labels == 1]
    data2 = data[labels == 0]
    labels1 = np.ones(n_per_class)
    labels2 = -np.ones(n_per_class)
  
  elif dataset_type == 'circles':
    data, labels = make_circles(n_samples, noise=noise, factor=0.5, random_state=0)
    data1 = data[labels == 1]
    data2 = data[labels == 0]
    labels1 = np.ones(n_per_class)
    labels2 = -np.ones(n_per_class)
  
  elif dataset_type == 'xor':
    data1 = np.vstack([
      np.random.randn(n_per_class//2, 2) + [2, 2],
      np.random.randn(n_per_class//2, 2) - [2, 2]
    ]) + noise * np.random.randn(n_per_class, 2)
    data2 = np.vstack([
      np.random.randn(n_per_class//2, 2) + [-2, 2],
      np.random.randn(n_per_class//2, 2) - [-2, 2]
    ]) + noise * np.random.randn(n_per_class, 2)
    labels1 = np.ones(n_per_class)
    labels2 = -np.ones(n_per_class)
  
  else:
    raise ValueError("Unknown dataset type")
  
  complete_set = np.vstack([
    np.hstack([data1, labels1.reshape(-1, 1)]),
    np.hstack([data2, labels2.reshape(-1, 1)])
  ])
  return data1, labels1, data2, labels2, complete_set

# ---------------------------- Visualization ----------------------------
def generate_grid(complete_set, resolution=100):
  x_min, x_max = complete_set[:, 0].min() - 1, complete_set[:, 0].max() + 1
  y_min, y_max = complete_set[:, 1].min() - 1, complete_set[:, 1].max() + 1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
             np.linspace(y_min, y_max, resolution))
  X = np.c_[xx.ravel(), yy.ravel()]
  return X, xx, yy

def plot_boundaries(ax, X, xx, yy, data1, data2):
  Z = X[:, 2].reshape(xx.shape)
  cmap = ListedColormap(['#FFFF80', '#9999FF'])
  ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)
  ax.scatter(data1[:, 0], data1[:, 1], c='b', edgecolors='k')
  ax.scatter(data2[:, 0], data2[:, 1], c='y', edgecolors='k')
  ax.set_title('Decision boundaries')

def plot_charspace(ax, Q1s, Q2s, labels):
  ax.scatter(Q1s[labels == 1], Q2s[labels == 1], c='b')
  ax.scatter(Q1s[labels == -1], Q2s[labels == -1], c='y')
  max_val = max(np.max(Q1s), np.max(Q2s))
  ax.plot([0, max_val], [0, max_val], 'k--')
  ax.set_title('Characteristic space')
  ax.legend()

def plot_results(X_grid, xx, yy, data1, data2, Q1s, Q2s, labels, k, h, noise, which, save=False):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
  
  # Plot decision boundaries
  plot_boundaries(ax1, X_grid, xx, yy, data1, data2)
  
  # Plot characteristic space
  plot_charspace(ax2, Q1s, Q2s, labels)
  
  # Add parameter info
  param_val = h if which == 'h' else noise
  fig.suptitle(f'k = {k}, {which} = {param_val:.2f}')
  
  if save:
    os.makedirs('output', exist_ok=True)
    plt.savefig(f'output/kNN_{k}_{which}{param_val:.2f}.png')
    plt.close()
  else:
    plt.show()

# ---------------------------- Evaluation ----------------------------
def generate_and_evaluate(dataset_type, samples, k, h, noise, which, save=False):
  data1, labels1, data2, labels2, complete_set = generate_dataset(dataset_type, samples, noise)
  labels = np.hstack([labels1, labels2])
  
  # Generate grid
  X_grid, xx, yy = generate_grid(complete_set)
  X_with_pred = np.zeros((X_grid.shape[0], 3))
  X_with_pred[:, :2] = X_grid
  
  # Classify grid points
  for i in range(X_grid.shape[0]):
    X_with_pred[i, 2] = mykNN(X_grid[i], complete_set, k, h)
  
  # Transform to characteristic space
  Q1s = np.zeros(samples)
  Q2s = np.zeros(samples)
  for i in range(samples):
    Q1s[i], Q2s[i] = to_characteristic_space(complete_set[i, :2], complete_set, k, h)
  
  # Plot results
  plot_results(X_with_pred, xx, yy, data1, data2, Q1s, Q2s, labels, k, h, noise, which, save)

# ---------------------------- Main Execution ----------------------------
if __name__ == "__main__":
  DATASET_TYPE = 'blobs'
  SAMPLES = 200
  
  # Evaluate bandwidth (h)
  for k in [1, 25, 50]:
    for h in [0.0001, 0.01, 1]:
      generate_and_evaluate(DATASET_TYPE, SAMPLES, k, h, noise=1, which='h', save=False)
  
  # Evaluate noise
  for k in [1, 25, 50]:
    for noise in [1, 2, 3]:
      generate_and_evaluate(DATASET_TYPE, SAMPLES, k, h=1, noise=noise, which='noise', save=False)
      
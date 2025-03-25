import numpy as np

from sklearn.datasets import make_moons, make_circles

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

def generate_grid(complete_set, resolution=100):
  x_min, x_max = complete_set[:, 0].min() - 1, complete_set[:, 0].max() + 1
  y_min, y_max = complete_set[:, 1].min() - 1, complete_set[:, 1].max() + 1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
             np.linspace(y_min, y_max, resolution))
  X = np.c_[xx.ravel(), yy.ravel()]
  return X, xx, yy

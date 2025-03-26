import numpy as np

from sklearn.datasets import make_moons, make_circles

# Function to generate synthetic datasets based on the specified type
def generate_dataset(dataset_type, n_samples, noise):
  # Set random seed for reproducibility
  np.random.seed(0)
  # Calculate the number of samples per class
  n_per_class = n_samples // 2
  
  # Generate data for 'blobs' dataset
  if dataset_type == 'blobs':
    # Create two Gaussian blobs with added noise
    data1 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], n_per_class) + noise * np.random.randn(n_per_class, 2)
    data2 = np.random.multivariate_normal([-2, -2], [[0.5, 0], [0, 0.5]], n_per_class) + noise * np.random.randn(n_per_class, 2)
    # Assign labels to the blobs
    labels1 = np.ones(n_per_class)
    labels2 = -np.ones(n_per_class)
  
  # Generate data for 'spirals' dataset
  elif dataset_type == 'spirals':
    # Create two spirals with added noise
    t = np.linspace(0, 4 * np.pi, n_per_class)
    data1 = np.column_stack([t * np.cos(t), t * np.sin(t)]) + noise * np.random.randn(n_per_class, 2)
    data2 = np.column_stack([-t * np.cos(t), -t * np.sin(t)]) + noise * np.random.randn(n_per_class, 2)
    # Assign labels to the spirals
    labels1 = np.ones(n_per_class)
    labels2 = -np.ones(n_per_class)
  
  # Generate data for 'moons' dataset
  elif dataset_type == 'moons':
    # Use sklearn's make_moons function to generate data
    data, labels = make_moons(n_samples, noise=noise, random_state=0)
    # Separate data into two classes
    data1 = data[labels == 1]
    data2 = data[labels == 0]
    # Assign labels to the classes
    labels1 = np.ones(n_per_class)
    labels2 = -np.ones(n_per_class)
  
  # Generate data for 'circles' dataset
  elif dataset_type == 'circles':
    # Use sklearn's make_circles function to generate data
    data, labels = make_circles(n_samples, noise=noise, factor=0.5, random_state=0)
    # Separate data into two classes
    data1 = data[labels == 1]
    data2 = data[labels == 0]
    # Assign labels to the classes
    labels1 = np.ones(n_per_class)
    labels2 = -np.ones(n_per_class)
  
  # Generate data for 'xor' dataset
  elif dataset_type == 'xor':
    # Create XOR pattern with added noise
    data1 = np.vstack([
      np.random.randn(n_per_class // 2, 2) + [2, 2],
      np.random.randn(n_per_class // 2, 2) - [2, 2]
    ]) + noise * np.random.randn(n_per_class, 2)
    data2 = np.vstack([
      np.random.randn(n_per_class // 2, 2) + [-2, 2],
      np.random.randn(n_per_class // 2, 2) - [-2, 2]
    ]) + noise * np.random.randn(n_per_class, 2)
    # Assign labels to the XOR pattern
    labels1 = np.ones(n_per_class)
    labels2 = -np.ones(n_per_class)
  
  # Raise an error if the dataset type is unknown
  else:
    raise ValueError("Unknown dataset type")
  
  # Combine data and labels into a complete dataset
  complete_set = np.vstack([
    np.hstack([data1, labels1.reshape(-1, 1)]),
    np.hstack([data2, labels2.reshape(-1, 1)])
  ])
  return data1, labels1, data2, labels2, complete_set

# Function to generate a grid for visualization
def generate_grid(complete_set, resolution=100):
  # Determine the range of the grid based on the dataset
  x_min, x_max = complete_set[:, 0].min() - 1, complete_set[:, 0].max() + 1
  y_min, y_max = complete_set[:, 1].min() - 1, complete_set[:, 1].max() + 1
  # Create a mesh grid with the specified resolution
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
             np.linspace(y_min, y_max, resolution))
  # Flatten the grid into a 2D array of points
  X = np.c_[xx.ravel(), yy.ravel()]
  return X, xx, yy

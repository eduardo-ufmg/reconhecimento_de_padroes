import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, Y, G, P, dataset_name):
  """Plot the decision boundary of a classifier."""
  fig, ax = plt.subplots(figsize=(10, 6))
  
  # Extract unique grid coordinates
  x_coords = np.unique(G[:, 0])
  y_coords = np.unique(G[:, 1])
  nx, ny = len(x_coords), len(y_coords)
  Z = P.reshape(ny, nx)
  
  # Create a colormap for the decision regions
  cmap = ListedColormap(['#FFAAAA', '#AAAAFF'])
  ax.contourf(x_coords, y_coords, Z, cmap=cmap, alpha=0.3)
  
  # Plot the data points with original styling
  X1 = X[Y == -1]
  X2 = X[Y == 1]
  ax.scatter(X1[:, 0], X1[:, 1], c='red', marker='o', edgecolor='k')
  ax.scatter(X2[:, 0], X2[:, 1], c='blue', marker='x')
  
  ax.set_title(f'Decision Boundary for {dataset_name} Dataset')

  # Create output directory for saving the plot
  output_dir = './output'
  os.makedirs(output_dir, exist_ok=True)
  
  # Save the plot to a file
  output_path = os.path.join(output_dir, f'{dataset_name}_decision_boundary.png')
  plt.savefig(output_path)


def plot_errors(errors, dataset_name):
  """Plot the training errors over iterations."""
  plt.figure(figsize=(10, 6))
  plt.plot(errors, marker='o', linestyle='-', color='b')
  plt.title(f'Training Errors ({dataset_name} Dataset)')
  plt.xlabel('Iteration')
  plt.ylabel('Number of Errors')
  plt.grid(True)

  # Create output directory for saving the plot
  output_dir = './output'
  os.makedirs(output_dir, exist_ok=True)

  # Save the plot to a file
  output_path = os.path.join(output_dir, f'{dataset_name}_errors.png')
  plt.savefig(output_path)


def plot_accuracyk(k_values, average_accuracies, dataset_name, num_instances, num_features, kfold_rd_state):
  """Plot the average accuracy vs k for a given dataset."""

  # Plot the average accuracy vs k
  plt.figure(figsize=(10, 6))
  plt.plot(k_values, average_accuracies, marker='o', linestyle='-', color='b')
  plt.title(f'Average Accuracy vs k ({dataset_name} Dataset)')
  plt.xlabel('k')
  plt.ylabel('Average Accuracy')
  plt.legend([f'{dataset_name}: {num_instances} instances, {num_features} features'])
  plt.grid(True)

  # Create output directory for saving the plot
  output_dir = './results'
  os.makedirs(output_dir, exist_ok=True)

  # Save the plot to a file
  output_path = os.path.join(output_dir, f'{dataset_name}_{kfold_rd_state}.png')
  plt.savefig(output_path)


def plot_characteristic_space(QX, YX, QG, YG, dataset_name):
  """Plot the characteristic space of the kNN classifier."""

  fig, ax = plt.subplots(figsize=(10, 6))
  
  # Plot grid points based on their predicted labels with lower alpha
  # Assuming QG is of shape (2, n_samples)
  ax.scatter(QG[0, YG == 1], QG[1, YG == 1], c='blue', marker='x', alpha=0.1)
  ax.scatter(QG[0, YG == -1], QG[1, YG == -1], c='red', marker='x', alpha=0.1)
  
  # Plot training points based on their true labels
  ax.scatter(QX[0, YX == 1], QX[1, YX == 1], c='blue', marker='o', edgecolor='k')
  ax.scatter(QX[0, YX == -1], QX[1, YX == -1], c='red', marker='o', edgecolor='k')
  
  ax.set_title(f'Characteristic Space for {dataset_name} Dataset')
  
  # Create output directory for saving the plot
  output_dir = './output'
  os.makedirs(output_dir, exist_ok=True)
  
  # Save the plot to a file
  output_path = os.path.join(output_dir, f'{dataset_name}_characteristic_space.png')
  plt.savefig(output_path)
  plt.close(fig)


def plot_kNN_results(X, Y, G, P, QX, QG, dataset_name):
  """Plot the kNN decision boundary and characteristic space."""

  # Plot the decision boundary
  plot_decision_boundary(X, Y, G, P, dataset_name)

  # Plot the characteristic space
  plot_characteristic_space(QX, Y, QG, P, dataset_name)

  plt.show()

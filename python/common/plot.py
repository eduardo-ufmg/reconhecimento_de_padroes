import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_dataset(X1, X2, dataset_name):
  """Plot the dataset with two classes."""

  # Create a figure and axis
  fig, ax = plt.subplots(figsize=(10, 6))
  # Create a custom color map
  cmap = ListedColormap(['#FFAAAA', '#AAAAFF'])
  # Plot the first class
  ax.scatter(X1[:, 0], X1[:, 1], c='red', marker='o', edgecolor='k')
  # Plot the second class
  ax.scatter(X2[:, 0], X2[:, 1], c='blue', marker='x')
  # Set the title
  ax.set_title(f'{dataset_name} Dataset')


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
  class_minus1 = X[Y == -1]
  class1 = X[Y == 1]
  ax.scatter(class_minus1[:, 0], class_minus1[:, 1], c='red', marker='o', edgecolor='k')
  ax.scatter(class1[:, 0], class1[:, 1], c='blue', marker='x')
  
  ax.set_title(f'Decision Boundary for {dataset_name} Dataset')


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

  # Display the plot
  # plt.show() # Uncomment this line to display the plot in the console

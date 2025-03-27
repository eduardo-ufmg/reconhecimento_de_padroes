import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Function to plot decision boundaries
def plot_boundaries(ax, X, xx, yy, data1, data2):
  # Reshape the third column of X to match the grid shape
  Z = X[:, 2].reshape(xx.shape)
  # Define a colormap for the decision regions
  cmap = ListedColormap(['#FFFF80', '#9999FF'])
  # Plot the decision boundaries as a filled contour plot
  ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)
  # Scatter plot for the first dataset
  ax.scatter(data1[:, 0], data1[:, 1], c='b', edgecolors='k')
  # Scatter plot for the second dataset
  ax.scatter(data2[:, 0], data2[:, 1], c='y', edgecolors='k')
  # Set the title for the plot
  ax.set_title('Decision boundaries')

# Function to plot the characteristic space
def plot_charspace(ax, Q1s, Q2s, labels):
  # Scatter plot for points with label 1
  ax.scatter(Q1s[labels == 1], Q2s[labels == 1], c='b')
  # Scatter plot for points with label -1
  ax.scatter(Q1s[labels == -1], Q2s[labels == -1], c='y')
  # Calculate the maximum value for the diagonal line
  max_val = max(np.max(Q1s), np.max(Q2s))
  # Plot a dashed diagonal line
  ax.plot([0, max_val], [0, max_val], 'k--')
  # Set the title for the plot
  ax.set_title('Characteristic space')

# Function to plot the results, including decision boundaries and characteristic space
def plot_results(X_grid, xx, yy, data1, data2, Q1s, Q2s, labels, k, h, noise, which, save=False):
  # Create a figure with two subplots side by side
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
  # Plot the decision boundaries on the first subplot
  plot_boundaries(ax1, X_grid, xx, yy, data1, data2)
  # Plot the characteristic space on the second subplot
  plot_charspace(ax2, Q1s, Q2s, labels)
  # Determine the parameter value to display in the title
  param_val = h if which == 'h' else noise
  # Set the overall title for the figure
  fig.suptitle(f'k = {k}, {which} = {param_val:.2f}')
  
  if save:
    # If save is True, create an output directory and save the plot as a PNG file
    os.makedirs('output', exist_ok=True)
    plt.savefig(f'output/kNN_{k}_{which}{param_val:.2f}.png')
    plt.close()
  else:
    # Otherwise, display the plot
    plt.show()

def plot_accuracyk(k_values, average_accuracies, dataset_name, num_instances, num_features):
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
  output_path = os.path.join(output_dir, f'{dataset_name}.png')
  plt.savefig(output_path)

  # Display the plot
  # plt.show() # Uncomment this line to display the plot in the console

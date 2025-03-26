import numpy as np

# Import necessary functions from other modules
from common.generate import generate_dataset, generate_grid
from kNN.mykNN import mykNN, to_characteristic_space
from common.plot import plot_results

# Function to generate a dataset, classify points, and evaluate results
def generate_and_evaluate(dataset_type, samples, k, h, noise, which, save=False):
  # Generate dataset with specified parameters
  data1, labels1, data2, labels2, complete_set = generate_dataset(dataset_type, samples, noise)
  labels = np.hstack([labels1, labels2])  # Combine labels from both classes
  
  # Generate a grid of points for classification
  X_grid, xx, yy = generate_grid(complete_set)
  X_with_pred = np.zeros((X_grid.shape[0], 3))  # Initialize array to store grid points and predictions
  X_with_pred[:, :2] = X_grid  # Store grid points (x, y) coordinates
  
  # Classify each grid point using kNN
  for i in range(X_grid.shape[0]):
    X_with_pred[i, 2] = mykNN(X_grid[i], complete_set, k, h)  # Store predicted class in the third column
  
  # Transform dataset points to characteristic space
  Q1s = np.zeros(samples)  # Initialize array for Q1 values
  Q2s = np.zeros(samples)  # Initialize array for Q2 values
  for i in range(samples):
    Q1s[i], Q2s[i] = to_characteristic_space(complete_set[i, :2], complete_set, k, h)  # Compute Q1 and Q2
  
  # Plot the results, including grid predictions and characteristic space
  plot_results(X_with_pred, xx, yy, data1, data2, Q1s, Q2s, labels, k, h, noise, which, save)
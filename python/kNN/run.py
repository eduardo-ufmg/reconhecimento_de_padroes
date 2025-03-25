import numpy as np

from common.generate import generate_dataset, generate_grid
from kNN.mykNN import mykNN, to_characteristic_space
from common.plot import plot_results

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
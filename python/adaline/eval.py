import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from adaline.train import train_pseudoinv
from adaline.pred import pred

if __name__ == '__main__':
  # Load the Boston housing dataset
  data_url = "http://lib.stat.cmu.edu/datasets/boston"
  raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
  X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
  y = raw_df.values[1::2, 2]

  # Initialize K-Fold cross-validation
  kf = KFold(n_splits=10, shuffle=True)

  error_stats = []
  fig, axes = plt.subplots(5, 2, figsize=(15, 20))  # Create a 5x2 grid of subplots
  axes = axes.flatten()  # Flatten the axes array for easy indexing

  for fold, (train_index, test_index) in enumerate(kf.split(X)):
    # Split the data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model using the pseudoinverse method
    w = train_pseudoinv(X_train, y_train)

    # Predict the target values for the test set
    y_pred = pred(X_test, w)

    # Calculate the mean squared error for the fold
    mse = np.mean((y_test - y_pred)**2)

    # Append the error statistics for the fold
    error_stats.append({'fold': fold + 1, 'mse': mse})

    # Plot the true vs predicted values for the current fold
    ax = axes[fold]
    ax.plot(y_test, marker='o', linestyle='-', alpha=0.7, label='True Values' if fold == 0 else "")
    ax.plot(y_pred, marker='x', linestyle='--', alpha=0.7, label='Predicted Values' if fold == 0 else "")
    ax.set_title(f'Fold {fold + 1}')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Target Value')

  # Add a single legend for the entire figure
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper center', ncol=2, fontsize='large')

  # Adjust layout for better visualization
  plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend

  # Save the plot to a file
  os.makedirs('./adaline/output/', exist_ok=True)
  plt.savefig('./adaline/output/fold_predictions.png')

  # Convert error statistics to a DataFrame
  error_stats_df = pd.DataFrame(error_stats)

  # Save the error statistics to a CSV file
  error_stats_df.to_csv('./adaline/output/error_stats.csv', index=False)

  print("Error statistics saved to 'error_stats.csv'.")
  print("Prediction plots saved to 'fold_predictions.png'.")

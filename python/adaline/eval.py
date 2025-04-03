import os
import numpy as np
import pandas as pd
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
  kf = KFold(n_splits=10, shuffle=True, random_state=42)

  error_stats = []

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

  # Convert error statistics to a DataFrame
  error_stats_df = pd.DataFrame(error_stats)

  # Save the error statistics to a CSV file
  os.makedirs('./adaline/output/', exist_ok=True)
  error_stats_df.to_csv('./adaline/output/error_stats.csv', index=False)

  print("Error statistics saved to 'error_stats.csv'.")

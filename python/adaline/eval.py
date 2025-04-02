import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from common.generate import generate_reg
from adaline.train import train_pseudoinv
from adaline.pred import pred

def linear(X: np.ndarray) -> np.ndarray:
  """
  Generate a linear target function.
  f(x, y) = 2*x + 3*y
  """
  return 2 * X[:, 0] + 3 * X[:, 1]

if __name__ == '__main__':
  # Generate a dataset using the parabolic function
  X, y = generate_reg(linear, n_samples=100, noise=1)

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
  error_stats_df.to_csv('error_stats.csv', index=False)

  print("Error statistics saved to 'error_stats.csv'.")





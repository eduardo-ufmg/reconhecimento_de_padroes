import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.animation as animation

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from scipy.stats import multivariate_normal

from bayes.train import train
from bayes.pred import pred
from bayes.likelihood import likelihood
from selection.featsel import drop_highcorr

# Constants
OUTPUT_DIR = Path("bayes/breastcancer/output")
TEST_SIZE = 0.3
RANDOM_STATE = 0
BANDWIDTH_RANGE = np.linspace(1e-2, 1e2, 1000)
N_BEST_WORST = 3
SELECTION_STEP = 10


def plot_likelihoods(
  data: tuple,
  accuracy: float,
  filename: str,
  show: bool = False
) -> None:
  """Plot likelihoods for training and test data with accuracy annotation."""
  Q0_train, Q0_test, Q1_train, Q1_test, y_train, y_test = data
  
  max_Q0 = max(np.max(Q0_train), np.max(Q0_test))
  max_Q1 = max(np.max(Q1_train), np.max(Q1_test))
  
  plt.figure()
  plt.scatter(Q0_train, Q1_train, c=y_train, marker='o', edgecolors='k')
  plt.scatter(Q0_test, Q1_test, c=y_test, marker='x')
  plt.plot([0, max_Q0], [0, max_Q1], 'k--')
  plt.xlabel("Q0")
  plt.ylabel("Q1")
  plt.title(f"acc: {accuracy:.2f}")
  
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  plt.savefig(OUTPUT_DIR / f"{filename}.png")
  if show:
    plt.show()
  plt.close()


def custom_bayes_gaussian(
  X_train: np.ndarray,
  X_test: np.ndarray,
  y_train: np.ndarray,
  bandwidth_matrix: np.ndarray
) -> tuple:
  """
  Bayesian classifier with custom diagonal covariance matrix.
  
  Args:
    X_train: Training features
    X_test: Test features
    y_train: Training labels
    bandwidth_matrix: Diagonal covariance matrix components
    
  Returns:
    Tuple containing predictions and likelihood values
  """
  # Validate bandwidth matrix
  if not np.allclose(bandwidth_matrix, bandwidth_matrix.T):
    raise ValueError("Bandwidth matrix must be symmetric")
  if np.any(np.linalg.eigvals(bandwidth_matrix) <= 0):
    raise ValueError("Bandwidth matrix must be positive definite")

  # Class separation
  X0_train = X_train[y_train == 0]
  X1_train = X_train[y_train == 1]
  
  # Prior probabilities
  prior0 = len(X0_train) / len(X_train)
  prior1 = 1 - prior0
  
  # Gaussian parameters
  mean0 = X0_train.mean(axis=0)
  mean1 = X1_train.mean(axis=0)
  
  # Calculate likelihoods
  Q0_train = multivariate_normal.pdf(X_train, mean=mean0, cov=bandwidth_matrix)
  Q1_train = multivariate_normal.pdf(X_train, mean=mean1, cov=bandwidth_matrix)
  Q0_test = multivariate_normal.pdf(X_test, mean=mean0, cov=bandwidth_matrix)
  Q1_test = multivariate_normal.pdf(X_test, mean=mean1, cov=bandwidth_matrix)
  
  # Posterior predictions
  y_pred = np.argmax(np.vstack([Q0_test*prior0, Q1_test*prior1]).T, axis=1)
  
  return y_pred, Q0_train, Q0_test, Q1_train, Q1_test


def run_default_model(
  X_train: pd.DataFrame,
  X_test: pd.DataFrame,
  y_train: pd.Series,
  y_test: pd.Series
) -> None:
  """Run standard Gaussian Naive Bayes classifier."""
  # Separate classes
  X0_train = X_train[y_train == 0]
  X1_train = X_train[y_train == 1]
  
  # Train and predict
  gaussian0, gaussian1 = train(X0_train, X1_train, method='normal')
  y_pred = pred(X_test, gaussian0, gaussian1, method='normal')
  
  # Calculate metrics
  accuracy = accuracy_score(y_test, y_pred)
  Q0_train, Q1_train = likelihood(X_train, gaussian0, gaussian1, method='normal')
  Q0_test, Q1_test = likelihood(X_test, gaussian0, gaussian1, method='normal')
  
  # Plot results
  plot_data = (Q0_train, Q0_test, Q1_train, Q1_test, y_train, y_test)
  plot_likelihoods(plot_data, accuracy, "default_likelihood")


def analyze_bandwidth(
  X_train: pd.DataFrame,
  X_test: pd.DataFrame,
  y_train: pd.Series,
  y_test: pd.Series,
  step: int = SELECTION_STEP
) -> None:
  """Analyze bandwidth with both accuracy plot and animation."""
  n_features = X_train.shape[1]
  bandwidths = BANDWIDTH_RANGE[::step]
  accuracies = []
  frame_data = []

  # Precompute all data
  for h in bandwidths:
    H = np.eye(n_features) * h
    y_pred, Q0_tr, Q0_te, Q1_tr, Q1_te = custom_bayes_gaussian(
      X_train.values, X_test.values, y_train.values, H
    )
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    frame_data.append((h, acc, Q0_tr, Q0_te, Q1_tr, Q1_te))

  # 1. Plot accuracy vs bandwidth
  plt.figure()
  plt.plot(bandwidths, accuracies)
  plt.xlabel("bandwidth")
  plt.ylabel("accuracy")
  plt.savefig(OUTPUT_DIR / "accuracy_vs_bandwidth.png")
  plt.close()

  # 2. Create animation
  fig, ax = plt.subplots(figsize=(8, 6))
  plt.xlabel("Q0")
  plt.ylabel("Q1")

  def update(frame):
    h, acc, Q0_tr, Q0_te, Q1_tr, Q1_te = frame_data[frame]
    ax.clear()
    
    # Plot training and test data
    ax.scatter(Q0_tr, Q1_tr, c=y_train, marker='o', edgecolors='k')
    ax.scatter(Q0_te, Q1_te, c=y_test, marker='x')
    max_Q0 = max(Q0_tr.max(), Q0_te.max())
    max_Q1 = max(Q1_tr.max(), Q1_te.max())
    ax.plot([0, max_Q0], [0, max_Q1], 'k--')
    ax.set_title(f"Bandwidth: {h:.2f}, Accuracy: {acc:.2f}")
    return ax

  ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(bandwidths),
    interval=100
  )
  
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  ani.save(OUTPUT_DIR / "bandwidth_animation.gif", writer='pillow', fps=10)
  plt.close()


def main():
  # Load and prepare data
  X, y = load_breast_cancer(return_X_y=True, as_frame=True)
  X = X.dropna().astype(float)
  X = drop_highcorr(X, y)
  
  # Split data
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
  )
  
  # Run analyses
  run_default_model(X_train, X_test, y_train, y_test)
  analyze_bandwidth(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
  main()
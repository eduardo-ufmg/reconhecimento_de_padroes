import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from scipy.stats import multivariate_normal

from bayes.train import train
from bayes.pred import pred
from bayes.likelihood import likelihood
from selection.featsel import drop_highcorr

def plot(data, acc, name, show=False):
  Q0train, Q0test, Q1train, Q1test, ytrain, ytest = data
  maxQ0, maxQ1 = np.max(np.hstack([Q0train, Q0test])), np.max(np.hstack([Q1train, Q1test]))
  plt.figure()
  plt.scatter(Q0train, Q1train, c=ytrain, marker='o', edgecolors='k')
  plt.scatter(Q0test, Q1test, c=ytest, marker='x')
  plt.plot([0, maxQ0], [0, maxQ1], 'k--')
  plt.text(maxQ0 * 0.9, maxQ1 * 0.8, f'acc: {acc:.2f}')
  plt.savefig(f'bayes/breastcancer/output/{name}.png')
  if show:
    plt.show()
  plt.close()

def custom_bayes_gauss(Xtrain, Xtest, ytrain, ytest, H):
  """
  Instead of using the covariance matrix, this function
  uses a custom bandwidth for each feature
  Args:
    Xtrain: training data
    Xtest: test data
    ytrain: training labels
    ytest: test labels
    H: bandwidth list
  Returns:
    ypred: predicted labels
    Q0train: likelihood for class 0 on training data
    Q0test: likelihood for class 0 on test data
    Q1train: likelihood for class 1 on training data
    Q1test: likelihood for class 1 on test data
  """

  # Ensure H is symmetric positive definite
  if not np.allclose(H, H.T):
    raise ValueError("H must be symmetric.")
  if np.any(np.linalg.eigvals(H) <= 0):
    raise ValueError("H must be positive definite.")

  X0train, X1train = Xtrain[ytrain == 0], Xtrain[ytrain == 1]

  mean0 = np.mean(X0train, axis=0)
  mean1 = np.mean(X1train, axis=0)

  prior0 = len(X0train) / (len(X0train) + len(X1train))
  prior1 = len(X1train) / (len(X0train) + len(X1train))

  Q0train = multivariate_normal.pdf(Xtrain, mean=mean0, cov=H)
  Q1train = multivariate_normal.pdf(Xtrain, mean=mean1, cov=H)

  Q0test = multivariate_normal.pdf(Xtest, mean=mean0, cov=H)
  Q1test = multivariate_normal.pdf(Xtest, mean=mean1, cov=H)

  post0 = Q0test * prior0
  post1 = Q1test * prior1

  ypred = np.where(post0 > post1, 0, 1)

  return ypred, Q0train, Q0test, Q1train, Q1test

def run_default(X0train, X1train, Xtest, ytrain, ytest):
  # Train the model
  gaussian0, gaussian1 = train(X0train, X1train, method='normal')

  # Predict on the test set
  ypred = pred(Xtest, gaussian0, gaussian1, method='normal')

  # Calculate accuracy
  accuracy = accuracy_score(ytest, ypred)

  # Calculate likelihood for each class
  Q0train, Q1train = likelihood(Xtrain, gaussian0, gaussian1, method='normal')
  Q0test, Q1test = likelihood(Xtest, gaussian0, gaussian1, method='normal')

  plot((Q0train, Q0test, Q1train, Q1test, ytrain, ytest), accuracy, 'def_likelihood', show=False)

def run_bandwidth(X0train, X1train, Xtest, ytrain, ytest):
  # Custom bandwidth for Gaussian
  hs = np.linspace(1e-3, 3e2, 1000)
  accs = []

  for h in hs:
    H = np.diag([h] * nfeat)
    ypred_custom, Q0train_custom, Q0test_custom, Q1train_custom, Q1test_custom = custom_bayes_gauss(Xtrain, Xtest, ytrain, ytest, H)
    accuracy_custom = accuracy_score(ytest, ypred_custom)
    accs.append(accuracy_custom)

  plt.figure()
  plt.plot(hs, accs)
  plt.xlabel('Bandwidth (h)')
  plt.ylabel('Accuracy')
  plt.savefig('bayes/breastcancer/output/accuracy_vs_bandwidth.png')
  plt.close()

  # Sort bandwidths by accuracy
  sorted_hs_best = [h for _, h in sorted(zip(accs, hs), reverse=True)]
  sorted_hs_worst = [h for _, h in sorted(zip(accs, hs))]

  fig, axes = plt.subplots(2, 3, figsize=(15, 10))

  for i, h in enumerate(sorted_hs_best[:3]):
    H = np.diag([h] * nfeat)
    ypred_custom, Q0train_custom, Q0test_custom, Q1train_custom, Q1test_custom = custom_bayes_gauss(Xtrain, Xtest, ytrain, ytest, H)
    accuracy_custom = accuracy_score(ytest, ypred_custom)
    
    ax = axes[0, i]
    maxQ0, maxQ1 = np.max(np.hstack([Q0train_custom, Q0test_custom])), np.max(np.hstack([Q1train_custom, Q1test_custom]))
    ax.scatter(Q0train_custom, Q1train_custom, c=ytrain, marker='o', edgecolors='k')
    ax.scatter(Q0test_custom, Q1test_custom, c=ytest, marker='x')
    ax.plot([0, maxQ0], [0, maxQ1], 'k--')
    ax.text(maxQ0 * 0.7, maxQ1 * 0.9, f'acc: {accuracy_custom:.2f}\nh: {h:.2f}')

  # Plot the worst 3 bandwidths with their corresponding accuracies (lower row)
  for i, h in enumerate(sorted_hs_worst[:3]):
    H = np.diag([h] * nfeat)
    ypred_custom, Q0train_custom, Q0test_custom, Q1train_custom, Q1test_custom = custom_bayes_gauss(Xtrain, Xtest, ytrain, ytest, H)
    accuracy_custom = accuracy_score(ytest, ypred_custom)
    
    ax = axes[1, i]
    maxQ0, maxQ1 = np.max(np.hstack([Q0train_custom, Q0test_custom])), np.max(np.hstack([Q1train_custom, Q1test_custom]))
    ax.scatter(Q0train_custom, Q1train_custom, c=ytrain, marker='o', edgecolors='k')
    ax.scatter(Q0test_custom, Q1test_custom, c=ytest, marker='x')
    ax.plot([0, maxQ0], [0, maxQ1], 'k--')
    ax.text(maxQ0 * 0.7, maxQ1 * 0.1, f'acc: {accuracy_custom:.2f}\nh: {h:.2f}')

  plt.tight_layout()
  plt.savefig('bayes/breastcancer/output/custom_likelihoods.png')
  plt.close()



# Load the breast cancer dataset
(X, y) = load_breast_cancer(return_X_y=True, as_frame=True)

# Preprocess X to remove invalid data and ensure all values are float
X = X.dropna()  # Remove rows with missing values
X = X.astype(float)  # Convert all values to float
X = drop_highcorr(X, y)

nfeat = X.shape[1]

# Perform a random split of the data into train and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

X0train, X1train = Xtrain[ytrain == 0], Xtrain[ytrain == 1]

run_default(X0train, X1train, Xtest, ytrain, ytest)

run_bandwidth(X0train, X1train, Xtest, ytrain, ytest)

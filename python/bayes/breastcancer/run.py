import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

from bayes.train import train
from bayes.pred import pred
from bayes.likelihood import likelihood
from selection.featsel import drop_highcorr

# Load the breast cancer dataset
(X, y) = load_breast_cancer(return_X_y=True, as_frame=True)

# Preprocess X to remove invalid data and ensure all values are float
X = X.dropna()  # Remove rows with missing values
X = X.astype(float)  # Convert all values to float

X = drop_highcorr(X, y, threshold=0.8)  # Drop highly correlated features

# Perform a random split of the data into train and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5)

X0train, X1train = Xtrain[ytrain == 0], Xtrain[ytrain == 1]
X0test, X1test = Xtest[ytest == 0], Xtest[ytest == 1]

# Train the model
gaussian0, gaussian1 = train(X0train, X1train, method='normal')

# Predict on the test set
ypred = pred(Xtest, gaussian0, gaussian1, method='normal')

# Calculate accuracy
accuracy = accuracy_score(ytest, ypred)

# Calculate likelihood for each class
Q0train, Q1train = likelihood(Xtrain, gaussian0, gaussian1, method='normal')
Q0test, Q1test = likelihood(Xtest, gaussian0, gaussian1, method='normal')
maxQ0, maxQ1 = np.max(np.hstack([Q0train, Q0test])), np.max(np.hstack([Q1train, Q1test]))

# Scatter plot of the likelihoods
plt.figure()
plt.scatter(Q0train, Q1train, c=ytrain, marker='o', edgecolors='k')
plt.scatter(Q0test, Q1test, c=ytest, marker='x')
plt.plot([0, maxQ0], [0, maxQ1], 'k--')
plt.text(maxQ0 * 0.9, maxQ1 * 0.8, f'acc: {accuracy:.2f}')
plt.show()

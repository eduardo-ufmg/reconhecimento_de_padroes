import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from bayes.train import train
from bayes.pred import pred
from bayes.likelihood import likelihood

# Load the breast cancer dataset
(X, y) = load_breast_cancer(return_X_y=True, as_frame=True)

# Preprocess X to remove invalid data and ensure all values are float
X = X.dropna()  # Remove rows with missing values
X = X.astype(float)  # Convert all values to float

# Perform k-fold cross-validation
kf = KFold(n_splits=10, shuffle=True)

# Initialize lists to store results
accuracies = []

# Initialize the plot for likelihood spaces
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
plt.tight_layout(pad=3.0)

for fold, (train_index, test_index) in enumerate(kf.split(X)):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y[train_index], y[test_index]

  # Split training data into classes
  X0_train = X_train[y_train == 0]
  X1_train = X_train[y_train == 1]
  X0_test = X_test[y_test == 0]
  X1_test = X_test[y_test == 1]

  # Train the model
  model_args0, model_args1 = train(X0_train, X1_train, method='normal')

  # Predict using the model
  y_pred = pred(X_test, model_args0, model_args1, method='normal')

  # Calculate the accuracy
  accuracy = accuracy_score(y_test, y_pred)
  accuracies.append(accuracy)
  print(f"Fold {fold + 1}: Accuracy = {accuracy:.4f}")

  # Transform to likelihood space
  Q0train, Q1train = likelihood(X_train, model_args0, model_args1, method='normal')
  Q0test, Q1test = likelihood(X_test, model_args0, model_args1, method='normal')

  # Plot the likelihood space for the current fold
  ax = axes[fold // 5, fold % 5]

  # Colormap the points based on their class
  ax.scatter(Q0train, Q1train, c=y_train, marker='o', edgecolor='k')
  ax.scatter(Q0test, Q1test, c=y_test, marker='s', edgecolor='k')

  ax.set_title(f"Fold {fold + 1}")
  ax.set_xlabel("Q0")
  ax.set_ylabel("Q1")

# Calculate the average accuracy
average_accuracy = np.mean(accuracies)
print(f"Average Accuracy = {average_accuracy:.4f}")

# Calculate the standard deviation of the accuracies
std_accuracy = np.std(accuracies)
print(f"Standard Deviation of Accuracy = {std_accuracy:.4f}")

# Save the results to a table
results = {
  "Fold": [f"{i + 1}" for i in range(len(accuracies))],
  "Accuracy": [round(acc, 2) for acc in accuracies]
}
results["Fold"].append("Average")
results["Accuracy"].append(round(average_accuracy, 2))
results["Fold"].append("Standard Deviation")
results["Accuracy"].append(round(std_accuracy, 2))

# Create a DataFrame and save it as a CSV file
results_df = pd.DataFrame(results)
os.makedirs("./bayes/breastcancer/output", exist_ok=True)
results_df.to_csv("./bayes/breastcancer/output/kfold_results.csv", index=False)

# Save and show the plot
plt.savefig("./bayes/breastcancer/output/kfold_likelihood_space.png")
plt.show()

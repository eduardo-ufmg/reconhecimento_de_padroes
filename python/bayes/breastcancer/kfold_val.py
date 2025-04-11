import numpy as np
import pandas as pd
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from bayes.train import train
from bayes.pred import pred

# Load the breast cancer dataset
(X, y) = load_breast_cancer(return_X_y=True, as_frame=True)

# Preprocess X to remove invalid data and ensure all values are float
X = X.dropna()  # Remove rows with missing values
X = X.astype(float)  # Convert all values to float

# Perform 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True)

# Initialize lists to store results
accuracies = []

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

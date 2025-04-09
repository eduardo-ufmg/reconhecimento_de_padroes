import numpy as np
import pandas as pd
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from bayes.old.train import train
from bayes.old.pred import predict

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

  # Train the bayes model
  class_1_data = X_train[y_train == 0]
  class_2_data = X_train[y_train == 1]
  params_class_1, params_class_2 = train(class_1_data.values, 0, class_2_data.values, 1)

  # Predict the classes for the test set
  y_pred = predict(
    X_test.values,
    class_1_data.values, params_class_1,
    class_2_data.values, params_class_2,
    method="normal"
  )

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
os.makedirs("./bayes/old/output", exist_ok=True)
results_df.to_csv("./bayes/old/output/kfold_results.csv", index=False)

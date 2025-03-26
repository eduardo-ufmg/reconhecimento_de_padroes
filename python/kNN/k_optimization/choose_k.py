import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from kNN.mykNN import mykNN_batch
from kNN.k_optimization.load_prepare_occupancy import load_prepare_occup
from kNN.k_optimization.prepare_set import prepare_set

X, y = load_prepare_occup()

kf = KFold(n_splits=10, shuffle=True)

k_values = range(10, 100, 10)
average_accuracies = []

total_iterations = len(k_values) * kf.get_n_splits()
iteration_counter = 0

for k in k_values:
  accuracies = []
  for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    complete_set, X_test, y_pred = prepare_set(X_train, y_train, X_test, y_test)

    y_pred = mykNN_batch(X_test, complete_set, k=k, h=1.0)

    accuracy = np.mean(y_pred == y_test.values.ravel())
    accuracies.append(accuracy)

    iteration_counter += 1
    print(f"Progress: {iteration_counter}/{total_iterations} iterations completed", end='\r')
  
  average_accuracies.append(np.mean(accuracies))

# Get dataset name and its dimensions
dataset_name = "Occupancy"
num_instances, num_features = X.shape

# Update the plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, average_accuracies, marker='o', linestyle='-', color='b')
plt.title(f'Average Accuracy vs k ({dataset_name} Dataset)')
plt.xlabel('k')
plt.ylabel('Average Accuracy')
plt.legend([f'{dataset_name}: {num_instances} instances, {num_features} features'])
plt.grid(True)

# Ensure the directory exists
output_dir = './results'
os.makedirs(output_dir, exist_ok=True)

# Save the plot
output_path = os.path.join(output_dir, 'average_accuracy_vs_k.png')
plt.savefig(output_path)

# Show the plot
plt.show()

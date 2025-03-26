import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Import custom modules for kNN implementation and data preparation
from kNN.mykNN import mykNN_batch
from kNN.k_optimization.load_prepare_occupancy import load_prepare_occup
from kNN.k_optimization.prepare_set import prepare_set

# Load and prepare the Occupancy dataset
X, y = load_prepare_occup()

# Initialize K-Fold cross-validation with 10 splits and shuffling
kf = KFold(n_splits=10, shuffle=True)

# Define the range of k values to test
k_values = range(10, 100, 10)
average_accuracies = []  # List to store average accuracies for each k

# Calculate the total number of iterations for progress tracking
total_iterations = len(k_values) * kf.get_n_splits()
iteration_counter = 0  # Counter to track progress

# Loop through each k value
for k in k_values:
  accuracies = []  # List to store accuracies for each fold
  # Perform K-Fold cross-validation
  for train_index, test_index in kf.split(X):
    # Split the dataset into training and testing sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Prepare the complete set for kNN and the test set
    complete_set, X_test, y_pred = prepare_set(X_train, y_train, X_test, y_test)

    # Perform kNN classification
    y_pred = mykNN_batch(X_test, complete_set, k=k, h=1.0)

    # Calculate accuracy for the current fold
    accuracy = np.mean(y_pred == y_test.values.ravel())
    accuracies.append(accuracy)

    # Update progress counter and print progress
    iteration_counter += 1
    print(f"Progress: {iteration_counter}/{total_iterations} iterations completed", end='\r')
  
  # Calculate and store the average accuracy for the current k
  average_accuracies.append(np.mean(accuracies))

# Dataset metadata for visualization
dataset_name = "Occupancy"
num_instances, num_features = X.shape

# Plot the average accuracy vs k
plt.figure(figsize=(10, 6))
plt.plot(k_values, average_accuracies, marker='o', linestyle='-', color='b')
plt.title(f'Average Accuracy vs k ({dataset_name} Dataset)')
plt.xlabel('k')
plt.ylabel('Average Accuracy')
plt.legend([f'{dataset_name}: {num_instances} instances, {num_features} features'])
plt.grid(True)

# Create output directory for saving the plot
output_dir = './results'
os.makedirs(output_dir, exist_ok=True)

# Save the plot to a file
output_path = os.path.join(output_dir, 'average_accuracy_vs_k.png')
plt.savefig(output_path)

# Display the plot
plt.show()

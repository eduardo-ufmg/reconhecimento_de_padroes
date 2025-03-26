import multiprocessing
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from collections import defaultdict

# Import custom modules
from kNN.mykNN import mykNN_batch
from kNN.k_optimization.load_prepare import load_prepare_uciset
from kNN.k_optimization.prepare_set import prepare_set
from common.plot import plot_accuracyk

def process_task(args):
  """Process a single task of evaluating k for a specific fold."""
  k, train_index, test_index = args
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]
  
  # Prepare sets and run kNN
  complete_set, X_test_prepared, _ = prepare_set(X_train, y_train, X_test, y_test)
  y_pred = mykNN_batch(X_test_prepared, complete_set, k, 1.0)
  
  accuracy = np.mean(y_pred == y_test.values.ravel())
  return (k, accuracy)

def plot_k_for_set(setname, X, y):
  """plot average accuracy vs k for a given dataset."""
  
  # Initialize KFold and precompute splits
  kf = KFold(n_splits=10, shuffle=True)
  folds = list(kf.split(X))
  
  # Define k values and generate tasks
  k_values = range(10, 100, 10)
  tasks = [(k, train_idx, test_idx) for k in k_values for (train_idx, test_idx) in folds]
  
  # Configure parallelism (limit processes to avoid overload)
  n_procs = min(4, os.cpu_count())  # Adjust based on system capabilities
  
  # Process tasks in parallel
  accuracies_dict = defaultdict(list)
  total_tasks = len(tasks)
  
  with multiprocessing.Pool(processes=n_procs) as pool:
    results = []
    for i, result in enumerate(pool.imap_unordered(process_task, tasks), 1):
      results.append(result)
      print(f"Progress: {i}/{total_tasks} tasks completed", end='\r')
  
  # Aggregate results
  for k, accuracy in results:
    accuracies_dict[k].append(accuracy)
  
  # Calculate average accuracies
  average_accuracies = [np.mean(accuracies_dict[k]) for k in k_values]
  
  # Plot results
  plot_accuracyk(X, k_values, average_accuracies, setname)

if __name__ == "__main__":

  X, y = load_prepare_uciset(891) 

  plot_k_for_set('cdc_diabetes_health_indicators', X, y)

  X, y = load_prepare_uciset(329) 

  plot_k_for_set('diabetic_retinopathy_debrecen', X, y)

  X, y = load_prepare_uciset(357)

  plot_k_for_set('occupancy', X, y)

import multiprocessing
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from collections import defaultdict
from kNN.mykNN import mykNN_batch
from kNN.k_optimization.load_prepare import load_prepare_uciset, window_bigset
from kNN.k_optimization.prepare_set import prepare_set
from common.plot import plot_accuracyk

def k_range(ref):
  return list(range(5, ref//5, ref//200))

def process_task_set(args):
  """Process a single task for plot_k_for_set (uses global X, y set by init_pool_global)."""
  k, train_idx, test_idx = args
  global X, y
  X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
  y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
  
  complete_set, X_test_prepared, _ = prepare_set(X_train, y_train, X_test, y_test)
  y_pred = mykNN_batch(X_test_prepared, complete_set, k, 1.0)
  return (k, np.mean(y_pred == y_test.values.ravel()))

def process_task_window(args):
  """Process a single task for plot_k_for_windows (window data passed in args)."""
  k, X_window, y_window, train_idx, test_idx = args
  X_train = X_window.iloc[train_idx]
  X_test = X_window.iloc[test_idx]
  y_train = y_window.iloc[train_idx]
  y_test = y_window.iloc[test_idx]
  
  complete_set, X_test_prepared, _ = prepare_set(X_train, y_train, X_test, y_test)
  y_pred = mykNN_batch(X_test_prepared, complete_set, k, 1.0)
  return (k, np.mean(y_pred == y_test.values.ravel()))

def init_pool_global(X_shared, y_shared):
  """Initialize global X and y for child processes."""
  global X, y
  X = X_shared
  y = y_shared

def compute_average_accuracies(tasks, process_func, k_values, n_procs=4, initializer=None, initargs=None):
  """Helper function to compute average accuracies using multiprocessing."""
  accuracies_dict = defaultdict(list)
  total_tasks = len(tasks)
  
  with multiprocessing.Pool(processes=n_procs, initializer=initializer, initargs=initargs) as pool:
    results = []
    for i, result in enumerate(pool.imap_unordered(process_func, tasks), 1):
      results.append(result)
      print(f"Progress: {i}/{total_tasks} tasks completed", end='\r')
  
  for k, accuracy in results:
    accuracies_dict[k].append(accuracy)
  
  return [np.mean(accuracies_dict[k]) for k in k_values]

def plot_k_for_set(setname, X, y, kfold_rd_state):
  """Plot average accuracy vs k for a dataset using multiprocessing."""

  kf = KFold(n_splits=10, shuffle=True, random_state=kfold_rd_state)
  folds = list(kf.split(X))

  num_instances, num_features = X.shape
  k_values = list(k_range(num_instances))
  
  tasks = [(k, train_idx, test_idx) for k in k_values for (train_idx, test_idx) in folds]
  average_accuracies = compute_average_accuracies(
    tasks, process_task_set, k_values,
    initializer=init_pool_global, initargs=(X, y)
  )
  
  plot_accuracyk(k_values, average_accuracies, setname, num_instances, num_features, kfold_rd_state)
  print()

def plot_k_for_windows(setname, windows, num_instances, num_features, kfold_rd_state):
  """Plot average accuracy vs k for windowed dataset using multiprocessing."""
  
  window_size = windows[0][0].shape[0] if windows else 0
  k_values = list(k_range(window_size))
  
  tasks = []
  for k in k_values:
    for X_window, y_window in windows:
      kf = KFold(n_splits=10, shuffle=True, random_state=kfold_rd_state)
      for train_idx, test_idx in kf.split(X_window):
        tasks.append((k, X_window, y_window, train_idx, test_idx))
  
  average_accuracies = compute_average_accuracies(tasks, process_task_window, k_values)
  plot_accuracyk(k_values, average_accuracies, setname, num_instances, num_features, kfold_rd_state)
  print()

if __name__ == "__main__":
  
  rdgen = np.random.Generator(np.random.PCG64())


  for _ in range(5):
    kfold_rd_state = rdgen.integers(65536)

    # # Dataset 1: Process with windowing
    # X, y = load_prepare_uciset(891)
    # windows, num_inst, num_feat = window_bigset(X, y, window_size=10000)
    # plot_k_for_windows('cdc_diabetes_health_indicators', windows, num_inst, num_feat, kfold_rd_state)
    
    # # Dataset 2: Direct processing
    # X, y = load_prepare_uciset(329)
    # plot_k_for_set('diabetic_retinopathy_debrecen', X, y, kfold_rd_state)
    
    # Dataset 3: Direct processing
    X, y = load_prepare_uciset(357)
    plot_k_for_set('occupancy', X, y, kfold_rd_state)
  
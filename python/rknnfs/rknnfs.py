"""
Enhanced RKNN-FS implementation with type hints, parallel processing, and improved error handling.
"""

import os
import time
import logging
from typing import Tuple, Dict, Union, List
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
  accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MIN_FEATURES = 4  # Minimum features to maintain during elimination
DEFAULT_TEST_SIZE = 0.3
DEFAULT_RANDOM_STATE = 42

def compute_supports(
  X: np.ndarray,
  y: np.ndarray,
  r: int,
  k: int,
  n_jobs: int = -1
) -> Tuple[np.ndarray, float]:
  """
  Compute feature supports using parallelized bidirectional voting.

  Args:
    X: Input data matrix (n_samples, n_features)
    y: Target labels (n_samples,)
    r: Number of KNN models to train
    k: Number of neighbors for KNN
    n_jobs: Number of parallel jobs (-1 for all cores)

  Returns:
    Tuple containing:
    - support_values: Normalized support scores for each feature
    - mean_accuracy: Average accuracy of all KNN models
  """
  n_samples, p_current = X.shape
  m = int(np.sqrt(p_current))
  supports = np.zeros(p_current)
  counts = np.zeros(p_current)
  accuracies = []

  def process_iteration():
    nonlocal supports, counts
    selected = np.random.choice(p_current, size=m, replace=False)
    indices = np.random.permutation(n_samples)
    split = n_samples // 2
    
    X_base = X[indices[:split]][:, selected]
    y_base = y[indices[:split]]
    X_query = X[indices[split:]][:, selected]
    y_query = y[indices[split:]]

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_base, y_base)
    pred = knn.predict(X_query)
    acc = accuracy_score(y_query, pred)
    
    return selected, acc

  # Parallel execution
  results = Parallel(n_jobs=n_jobs)(
    delayed(process_iteration)() for _ in range(r)
  )

  # Aggregate results
  for selected, acc in results:
    accuracies.append(acc)
    for f in selected:
      supports[f] += acc
      counts[f] += 1

  # Calculate normalized supports
  support_values = np.divide(
    supports, counts, 
    out=np.zeros_like(supports), 
    where=counts != 0
  )
  return support_values, np.mean(accuracies)

def rknn_feature_selection(
  X: np.ndarray,
  y: np.ndarray,
  k: int = 3,
  r: int = 100,
  q: float = 0.5,
  d: int = 1,
  min_features: int = MIN_FEATURES
) -> np.ndarray:
  """
  Enhanced RKNN-FS with improved peak detection and early stopping.

  Args:
    X: Input data matrix
    y: Target labels
    k: KNN neighbors parameter
    r: Models per iteration
    q: Feature elimination ratio (Stage 1)
    d: Features to remove per iteration (Stage 2)
    min_features: Minimum features to preserve

  Returns:
    Selected feature indices
  """
  p = X.shape[1]
  remaining = np.arange(p)
  stage1_acc, stage1_features = [], []

  # Stage 1: Geometric Elimination
  while len(remaining) > min_features:
    supports, acc = compute_supports(X[:, remaining], y, r, k)
    stage1_acc.append(acc)
    stage1_features.append(remaining.copy())
    
    keep = max(int(len(remaining) * (1 - q)), 1)
    remaining = remaining[np.argsort(-supports)[:keep]]

  # Improved peak detection
  if stage1_acc:
    peak_idx = np.argmax(stage1_acc)
    # Look backward for first non-increasing point
    for i in range(peak_idx, 0, -1):
      if stage1_acc[i-1] < stage1_acc[i]:
        remaining = stage1_features[i]
        break
    else:
      remaining = stage1_features[0]

  # Stage 2: Linear Reduction
  stage2_acc, stage2_features = [], []
  while len(remaining) > min_features:
    supports, acc = compute_supports(X[:, remaining], y, r, k)
    stage2_acc.append(acc)
    stage2_features.append(remaining.copy())
    
    if len(remaining) <= d:
      break
    remaining = remaining[np.argsort(-supports)[:-d]]

  return stage2_features[np.argmax(stage2_acc)] if stage2_acc else remaining

def evaluate_model(
  model,
  X_train: np.ndarray,
  y_train: np.ndarray,
  X_test: np.ndarray,
  y_test: np.ndarray
) -> Dict[str, Union[float, np.ndarray]]:
  """Enhanced evaluation with class count awareness."""
  start = time.perf_counter()
  model.fit(X_train, y_train)
  train_time = time.perf_counter() - start
  
  start = time.perf_counter()
  y_pred = model.predict(X_test)
  pred_time = time.perf_counter() - start
  
  metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision_macro': precision_score(y_test, y_pred, average='macro'),
    'recall_macro': recall_score(y_test, y_pred, average='macro'),
    'f1_macro': f1_score(y_test, y_pred, average='macro'),
    'train_time': train_time,
    'pred_time': pred_time,
    'num_features': X_train.shape[1],
    'confusion_matrix': confusion_matrix(y_test, y_pred)
  }
  
  # Handle binary classification metrics
  unique_classes = np.unique(y_test)
  if len(unique_classes) == 2:
    tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
  else:
    metrics['specificity'] = np.nan
  
  return metrics

def compare_models(
  X_train: np.ndarray,
  y_train: np.ndarray,
  X_test: np.ndarray,
  y_test: np.ndarray,
  k: int = 3,
  r: int = 200
) -> Tuple[Dict, np.ndarray]:
  """
  Comprehensive model comparison with additional feature selectors.
  """
  # RKNN-FS
  selected = rknn_feature_selection(X_train, y_train, k=k, r=r)
  X_train_rknn = X_train[:, selected]
  X_test_rknn = X_test[:, selected]
  
  # KNN Models
  results = {}
  knn = KNeighborsClassifier(n_neighbors=k)
  results['RKNN-FS'] = evaluate_model(knn, X_train_rknn, y_train, X_test_rknn, y_test)
  results['KNN'] = evaluate_model(knn, X_train, y_train, X_test, y_test)
  
  # Random Forest with Feature Importance
  rf = RandomForestClassifier(n_estimators=100, random_state=DEFAULT_RANDOM_STATE)
  rf.fit(X_train, y_train)
  sfm = SelectFromModel(rf, threshold='median')
  X_train_rf = sfm.transform(X_train)
  X_test_rf = sfm.transform(X_test)
  results['RF-FS'] = evaluate_model(rf, X_train_rf, y_train, X_test_rf, y_test)
  
  return results, selected

def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
  """
  Unified dataset loader with error handling and path normalization.
  """
  data_dir = os.path.join(os.path.dirname(__file__), 'data')
  
  try:
    if name == 'Leukemia':
      url = "https://web.stanford.edu/~hastie/CASI_files/DATA/leukemia_big.csv"
      df = pd.read_csv(url, index_col=0).T
      y = pd.Series(df.index).str.contains('ALL').astype(int).values
      return df.values.astype(float), y
    
    elif name == 'Gastrointestinal':
      path = os.path.join(data_dir, 'gastrointestinal+lesions+in+regular+colonoscopy', 'data.txt')
      df = pd.read_csv(path, header=None).T
      y = df[1].astype(int).values
      return df.drop([0, 1], axis=1).astype(float).values, y
    
    elif name == 'Period Changer':
      path = os.path.join(data_dir, 'period+changer-2', 'data.csv')
      df = pd.read_csv(path)
      return df.iloc[:, :-1].values, df.iloc[:, -1].values
    
    elif name == 'Toxicity':
      data = fetch_ucirepo(id=728)
      return data.data.features.values, data.data.targets.values.ravel()
    
  except Exception as e:
    logging.error(f"Error loading {name}: {str(e)}")
    raise

def run_experiments() -> pd.DataFrame:
  """
  Main experiment runner with improved result handling and visualization.
  """
  datasets = ['Toxicity', 'Period Changer', 'Gastrointestinal', 'Leukemia']
  results = []

  for name in datasets:
    logging.info(f"Processing {name}")
    try:
      X, y = load_dataset(name)
      X = StandardScaler().fit_transform(X)
      X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE, stratify=y
      )
      
      model_results, selected = compare_models(X_train, y_train, X_test, y_test)
      
      for model_name, metrics in model_results.items():
        results.append({
          'Dataset': name,
          'Model': model_name,
          **metrics,
          'Feature Reduction %': (
            (X.shape[1] - metrics['num_features']) / X.shape[1] * 100
            if model_name == 'RKNN-FS' else 0
          )
        })
        
    except Exception as e:
      logging.error(f"Skipping {name} due to error: {str(e)}")
      continue

  # Create comprehensive report
  report = pd.DataFrame(results)
  report.to_csv(
    os.path.join('rknnfs', 'output', 'experiment_results.csv'), 
    index=False
  )
  
  # Generate visual report
  fig, ax = plt.subplots(figsize=(14, 6))
  for model in report.Model.unique():
    subset = report[report.Model == model]
    ax.plot(subset.Dataset, subset.accuracy, 'o-', label=model)
  
  ax.set_title('Model Accuracy Across Datasets')
  ax.set_ylabel('Accuracy')
  ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.tight_layout()
  plt.savefig(
    os.path.join('rknnfs', 'output', 'performance_comparison.png'),
    bbox_inches='tight', dpi=300
  )
  
  return report

if __name__ == "__main__":
  report = run_experiments()
  print("\nExperiment Summary:")
  print(report.groupby(['Dataset', 'Model']).mean(numeric_only=True))
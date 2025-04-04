import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, LabelEncoder
from urllib.request import urlopen
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix)
import time
import matplotlib.pyplot as plt

def compute_supports(X, y, r, k):
  """
  Compute feature supports using bidirectional voting (Table 1).
  
  Args:
    X (np.ndarray): Input data (n_samples, n_features).
    y (np.ndarray): Target labels (n_samples,).
    r (int): Number of KNN models to train.
    k (int): Number of neighbors for KNN.
  
  Returns:
    np.ndarray: Support scores for each feature.
    float: Average accuracy of all KNN models.
  """
  n_samples, p_current = X.shape
  m = int(np.sqrt(p_current))  # Features per KNN
  supports = np.zeros(p_current)
  counts = np.zeros(p_current)
  accuracies = []

  for _ in range(r):
    # Randomly select m features
    selected = np.random.choice(p_current, size=m, replace=False)
    # Dynamic partition: split data into base and query
    indices = np.random.permutation(n_samples)
    split = n_samples // 2
    base_idx, query_idx = indices[:split], indices[split:]
    
    X_base = X[base_idx][:, selected]
    y_base = np.array(y)[base_idx]
    X_query = X[query_idx][:, selected]
    y_query = np.array(y)[query_idx]

    # Train KNN and predict
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_base, y_base)
    pred = knn.predict(X_query)
    acc = accuracy_score(y_query, pred)
    accuracies.append(acc)

    # Update supports for selected features
    for f in selected:
      supports[f] += acc
      counts[f] += 1

  # Compute average support (avoid division by zero)
  support_values = np.zeros(p_current)
  for f in range(p_current):
    if counts[f] > 0:
      support_values[f] = supports[f] / counts[f]
    else:
      support_values[f] = 0.0  # Unselected features

  return support_values, np.mean(accuracies)

def rknn_feature_selection(X, y, k=3, r=100, q=0.5, d=1):
  """
  RKNN-FS two-stage backward elimination (Table 2).
  
  Args:
    X (np.ndarray): Input data (n_samples, n_features).
    y (np.ndarray): Target labels (n_samples,).
    k (int): Number of neighbors for KNN.
    r (int): Number of KNN models per iteration.
    q (float): Proportion of features to drop in Stage 1.
    d (int): Number of features to drop per iteration in Stage 2.
  
  Returns:
    np.ndarray: Indices of selected features.
  """
  n_samples, p = X.shape
  remaining_features = np.arange(p)
  accuracies_stage1, feature_subsets_stage1 = [], []

  # Stage 1: Geometric Elimination
  current_p = p
  while current_p > 4:  # Minimum 4 features
    current_X = X[:, remaining_features]
    support_values, avg_acc = compute_supports(current_X, y, r, k)
    accuracies_stage1.append(avg_acc)
    feature_subsets_stage1.append(remaining_features.copy())

    # Keep top (1-q) features
    num_keep = max(int(current_p * (1 - q)), 1)
    sorted_indices = np.argsort(-support_values)
    remaining_features = remaining_features[sorted_indices[:num_keep]]
    current_p = len(remaining_features)

  # Find pre-max iteration (best before peak)
  if accuracies_stage1:
    max_idx = np.argmax(accuracies_stage1)
    pre_max_idx = max(0, max_idx - 1)
    remaining_features = feature_subsets_stage1[pre_max_idx]
  else:
    remaining_features = np.arange(p)  # Fallback

  # Stage 2: Linear Reduction
  accuracies_stage2, feature_subsets_stage2 = [], []
  current_p = len(remaining_features)
  num_iterations = (current_p - 4) // d

  for _ in range(num_iterations):
    current_X = X[:, remaining_features]
    support_values, avg_acc = compute_supports(current_X, y, r, k)
    accuracies_stage2.append(avg_acc)
    feature_subsets_stage2.append(remaining_features.copy())

    # Remove d features with lowest support
    sorted_indices = np.argsort(-support_values)
    remaining_features = remaining_features[sorted_indices[:-d]]
    current_p = len(remaining_features)
    if current_p <= 4:
      break

  # Select best from Stage 2
  if accuracies_stage2:
    best_idx = np.argmax(accuracies_stage2)
    best_features = feature_subsets_stage2[best_idx]
  else:
    best_features = remaining_features  # Fallback

  return best_features

def evaluate_model(model, X_train, y_train, X_test, y_test):
  """Helper function to train and evaluate a model"""
  start_time = time.time()
  model.fit(X_train, y_train)
  train_time = time.time() - start_time
  
  start_pred = time.time()
  y_pred = model.predict(X_test)
  pred_time = time.time() - start_pred
  
  return {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, average='macro'),
    'recall': recall_score(y_test, y_pred, average='macro'),
    'f1': f1_score(y_test, y_pred, average='macro'),
    'confusion_matrix': confusion_matrix(y_test, y_pred),
    'train_time': train_time,
    'pred_time': pred_time
  }

def compare_models(X, y, test_size=0.3, random_state=42, k=3, r=100):
  """Compare performance of different models"""
  # Split data
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
  )
  
  # 1. RKNN-FS + kNN
  selected_features = rknn_feature_selection(X_train, y_train, k=k, r=r)
  X_train_selected = X_train[:, selected_features]
  X_test_selected = X_test[:, selected_features]
  
  # 2. kNN with all features
  results = {}
  
  # Model 1: kNN with selected features
  knn_selected = KNeighborsClassifier(n_neighbors=k)
  results['kNN (RKNN-FS)'] = evaluate_model(
    knn_selected, X_train_selected, y_train, X_test_selected, y_test
  )
  
  # Model 2: kNN with all features
  knn_all = KNeighborsClassifier(n_neighbors=k)
  results['kNN (All Features)'] = evaluate_model(
    knn_all, X_train, y_train, X_test, y_test
  )
  
  # Model 3: Random Forest with all features
  rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
  results['Random Forest'] = evaluate_model(
    rf, X_train, y_train, X_test, y_test
  )
  
  return results, selected_features

def load_leukemia():
  """Golub et al. (1999) Leukemia dataset (n=72, p=7129)"""
  url = "https://web.stanford.edu/~hastie/CASI_files/DATA/leukemia_big.csv"
  df = pd.read_csv(url, index_col=0).T
  y = pd.Series(df.index).str.contains('ALL').astype(int).values
  X = df.values
  return X, y

def load_gastrointestinal():
  """Load the gastrointestinal lesions dataset.
  
  Returns:
      X (pd.DataFrame): Features including light type and raw features.
      y (pd.Series): Target labels indicating lesion type (3, 1, 2).
  """
  # Read the dataset without headers
  df = pd.read_csv("./rknnfs/data/gastrointestinal+lesions+in+regular+colonoscopy/data.txt", header=None)
  
  # Transpose the DataFrame to have samples as rows
  df_transposed = df.T
  
  # Extract the target variable y (class labels) from the second column
  y = df_transposed[1].astype(int)
  
  # Extract features X by dropping the lesion name and class label columns
  X = df_transposed.drop(columns=[0, 1])
  
  # Convert all feature columns to numeric (assuming no non-numeric values in features)
  X = X.apply(pd.to_numeric, errors='coerce')

  return X, y

def load_periodchanger():
  """Gül, Ş. & RAHIM, F. (2021). Period Changer [Dataset].
  UCI Machine Learning Repository. https://doi.org/10.24432/C5B31D"""
  
  df = pd.read_csv("./rknnfs/data/period+changer-2/data.csv")
  X = df.iloc[:, :-1]  # All columns except the last
  y = df.iloc[:, -1]   # Last column 'Class' as the target

  return X, y

def load_toxicity():
  "Gül, Ş. & RAHIM, F. (2021). Toxicity [Dataset]."
  "UCI Machine Learning Repository. https://doi.org/10.24432/C59313."
  
  toxicity = fetch_ucirepo(id=728)
  X = toxicity.data.features
  y = toxicity.data.targets.to_numpy().ravel()

  return X, y

def run_real_world_tests():
  """Run comparison on real 'small n, large p' datasets"""
  datasets = {
    'Toxicity': load_toxicity,
    'Period Changer': load_periodchanger,
    'Gastrointestinal': load_gastrointestinal,
    'Leukemia': load_leukemia
  }
  
  results = {}
  
  for name, loader in datasets.items():
    print(f"\n{'='*40}\nProcessing {name}\n{'='*40}")
    X, y = loader()
    
    # Preprocessing
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Run comparisons
    dataset_results, selected = compare_models(
      X_train, y_train, 
      test_size=0.3,
      k=3, 
      r=200,
      random_state=42
    )
    
    results[name] = {
      'metrics': dataset_results,
      'selected_features': len(selected),
      'total_features': X.shape[1]
    }
  
  # Collect all results into a DataFrame
  rows = []
  for dataset_name, res in results.items():
    total_features = res['total_features']
    selected_features = res['selected_features']
    for model_name, metrics in res['metrics'].items():
      num_features = selected_features if model_name == 'kNN (RKNN-FS)' else total_features
      rows.append({
          'Dataset': dataset_name,
          'Model': model_name,
          'Accuracy': metrics['accuracy'],
          'Precision': metrics['precision'],
          'Recall': metrics['recall'],
          'F1': metrics['f1'],
          'Train Time (s)': metrics['train_time'],
          'Prediction Time (s)': metrics['pred_time'],
          'Num Features': num_features,
      })
  
  results_df = pd.DataFrame(rows)
  
  # Save to CSV
  results_df.to_csv('./rknnfs/output/model_comparison_results.csv', index=False)
  
  # Create and save plotted table
  plt.figure(figsize=(12, 8))
  ax = plt.gca()
  ax.axis('off')
  plt.title("Model Performance Comparison")
  table = ax.table(
      cellText=results_df.round(3).values,
      colLabels=results_df.columns,
      loc='center',
      cellLoc='center'
  )
  table.auto_set_font_size(False)
  table.set_fontsize(10)
  table.scale(1.2, 1.2)
  plt.savefig('./rknnfs/output/model_comparison_table.png', bbox_inches='tight', dpi=300)
  plt.close()

if __name__ == "__main__":
  run_real_world_tests()
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load dataset
X, y = load_breast_cancer(return_X_y=True, as_frame=True)

# Preprocess
X = X.dropna().astype(float)
y = np.where(y == 0, -1, 1)  # Convert to -1 and 1

# 10-Fold CV
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []

for fold, (train_index, test_index) in enumerate(kf.split(X)):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y[train_index], y[test_index]
  
  # Inner loop to find optimal K
  best_k = 2
  best_score = -np.inf
  for k in range(2, 21):
    inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)
    inner_scores = []
    for inner_train_idx, inner_val_idx in inner_kf.split(X_train):
      # Inner training/validation split
      X_in_tr = X_train.iloc[inner_train_idx]
      y_in_tr = y_train[inner_train_idx]
      X_in_val = X_train.iloc[inner_val_idx]
      y_in_val = y_train[inner_val_idx]
      
      # Scale data
      scaler = StandardScaler()
      X_in_tr_scaled = scaler.fit_transform(X_in_tr)
      X_in_val_scaled = scaler.transform(X_in_val)
      
      # Cluster and compute RBF features
      kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
      kmeans.fit(X_in_tr_scaled)
      centers = kmeans.cluster_centers_
      
      # Calculate adaptive gamma
      labels = kmeans.labels_
      distances = np.linalg.norm(X_in_tr_scaled - centers[labels], axis=1)
      sigma = np.mean(distances)
      sigma = sigma if sigma > 0 else 1e-6
      gamma = 1 / (2 * sigma ** 2)
      
      # Compute RBF activations
      rbf_tr = np.exp(-gamma * pairwise_distances(X_in_tr_scaled, centers, metric='euclidean')**2)
      rbf_val = np.exp(-gamma * pairwise_distances(X_in_val_scaled, centers, metric='euclidean')**2)
      
      # Train classifier
      clf = LogisticRegression(max_iter=1000)
      clf.fit(rbf_tr, y_in_tr)
      y_pred = clf.predict(rbf_val)
      inner_scores.append(accuracy_score(y_in_val, y_pred))
    
    avg_score = np.mean(inner_scores)
    if avg_score > best_score:
      best_score = avg_score
      best_k = k
  
  # Train with best_k on entire training set
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  
  kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
  kmeans.fit(X_train_scaled)
  centers = kmeans.cluster_centers_
  
  # Compute gamma
  labels = kmeans.labels_
  distances = np.linalg.norm(X_train_scaled - centers[labels], axis=1)
  sigma = np.mean(distances)
  sigma = sigma if sigma > 0 else 1e-6
  gamma = 1 / (2 * sigma ** 2)
  
  # RBF features
  rbf_train = np.exp(-gamma * pairwise_distances(X_train_scaled, centers, metric='euclidean')**2)
  rbf_test = np.exp(-gamma * pairwise_distances(X_test_scaled, centers, metric='euclidean')**2)
  
  # Final classifier
  clf = LogisticRegression(max_iter=1000)
  clf.fit(rbf_train, y_train)
  y_pred = clf.predict(rbf_test)
  
  # Store results
  accuracy = accuracy_score(y_test, y_pred)
  accuracies.append(accuracy)
  print(f"Fold {fold+1}: Accuracy = {accuracy:.4f}, Best K = {best_k}")

# Calculate metrics
average_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
print(f"\nAverage Accuracy = {average_accuracy:.4f}")
print(f"Standard Deviation = {std_accuracy:.4f}")

# Save results
results = {
  "Fold": [f"{i+1}" for i in range(10)] + ["Average", "Std Deviation"],
  "Accuracy": [f"{acc:.4f}" for acc in accuracies] + [f"{average_accuracy:.4f}", f"{std_accuracy:.4f}"]
}
results_df = pd.DataFrame(results)
os.makedirs("./rbf/output", exist_ok=True)
results_df.to_csv("./rbf/output/kfold_results.csv", index=False)
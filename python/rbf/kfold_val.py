import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# Load and prepare data
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X = X.dropna().astype(float)
y = np.where(y == 0, -1, 1)  # Convert to -1/1 labels

# Configuration
k_values = range(2, 21)  # Test cluster quantities from 2 to 20
n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store results for each K
k_results = {k: [] for k in k_values}

# Outer loop: Evaluate each K candidate
for k in k_values:
  print(f"\n{'='*30}\nEvaluating K={k}\n{'='*30}")
  
  fold_accuracies = []
  
  # Inner loop: Cross-validation
  for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    # Data splitting
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Cluster training data
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_train_scaled)
    centers = kmeans.cluster_centers_
    
    # Calculate adaptive gamma
    labels = kmeans.labels_
    cluster_distances = np.linalg.norm(X_train_scaled - centers[labels], axis=1)
    sigma = np.mean(cluster_distances)
    gamma = 1 / (2 * (sigma ** 2)) if sigma > 0 else 1.0
    
    # Generate RBF features
    rbf_train = np.exp(-gamma * pairwise_distances(X_train_scaled, centers, metric='euclidean')**2)
    rbf_test = np.exp(-gamma * pairwise_distances(X_test_scaled, centers, metric='euclidean')**2)
    
    # Train and evaluate
    model = LogisticRegression(max_iter=1000)
    model.fit(rbf_train, y_train)
    y_pred = model.predict(rbf_test)
    
    # Record accuracy
    acc = accuracy_score(y_test, y_pred)
    fold_accuracies.append(acc)
    print(f"Fold {fold+1}/{n_folds} | Accuracy: {acc:.4f}")
  
  k_results[k] = fold_accuracies

# Determine optimal K
avg_accuracies = {k: np.mean(v) for k, v in k_results.items()}
best_k = max(avg_accuracies, key=avg_accuracies.get)
best_accuracies = k_results[best_k]

# Compile final results
results = {
  "Fold": [f"Fold {i+1}" for i in range(n_folds)] + ["Average", "Std Dev"],
  "Accuracy": [f"{acc:.4f}" for acc in best_accuracies] + 
        [f"{np.mean(best_accuracies):.4f}", f"{np.std(best_accuracies):.4f}"]
}

# Save and display
results_df = pd.DataFrame(results)
os.makedirs("./rbf/output", exist_ok=True)
results_df.to_csv("./rbf/output/kfold_results.csv", index=False)

print(f"\n{'#'*40}")
print(f"Optimal Cluster Quantity: K = {best_k}")
print(f"Cross-Validation Accuracy: {np.mean(best_accuracies):.4f} ± {np.std(best_accuracies):.4f}")
print(f"{'#'*40}")

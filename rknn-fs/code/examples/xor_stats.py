import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate XOR dataset
def generate_xor_data_with_blobs(n_samples=200, noise=0.5):
  centers = [(-1, -1), (1, 1), (-1, 1), (1, -1)]
  X, labels = make_blobs(n_samples=n_samples, centers=centers, cluster_std=noise)
  y = np.logical_xor(labels == 0, labels == 1).astype(int)
  return X, y

X, y = generate_xor_data_with_blobs()

# Create a single figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot original data
axes[0, 0].scatter(X[y == 0, 0], X[y == 0, 1], color='red')
axes[0, 0].scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
axes[0, 0].set_title("XOR Dataset")

# Plot feature distributions
axes[0, 1].hist(X[y == 0, 0], alpha=0.5, color='red')
axes[0, 1].hist(X[y == 1, 0], alpha=0.5, color='blue')
axes[0, 1].set_title("Histogram for x1")

axes[1, 0].hist(X[y == 0, 1], alpha=0.5, color='red')
axes[1, 0].hist(X[y == 1, 1], alpha=0.5, color='blue')
axes[1, 0].set_title("Histogram for x2")

# Evaluate single-feature classifiers
# Using X1
model_x1 = LogisticRegression()
model_x1.fit(X[:, 0].reshape(-1, 1), y)
acc_x1 = accuracy_score(y, model_x1.predict(X[:, 0].reshape(-1, 1)))

# Using X2
model_x2 = LogisticRegression()
model_x2.fit(X[:, 1].reshape(-1, 1), y)
acc_x2 = accuracy_score(y, model_x2.predict(X[:, 1].reshape(-1, 1)))

# Evaluate sign-based rule
pred_rule = (np.sign(X[:, 0]) == np.sign(X[:, 1])).astype(int)
acc_rule = accuracy_score(y, pred_rule)

print(f"Accuracy using only X1: {acc_x1:.2f}")
print(f"Accuracy using only X2: {acc_x2:.2f}")
print(f"Accuracy using sign relationship: {acc_rule:.2f}")

# Plot decision boundary based on sign relationship
xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 500), np.linspace(-2.5, 2.5, 500))
Z = (np.sign(xx) != np.sign(yy)).astype(int)
Z = Z.reshape(xx.shape)

axes[1, 1].contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
axes[1, 1].scatter(X[y == 0, 0], X[y == 0, 1], color='red')
axes[1, 1].scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
axes[1, 1].set_title("Decision Boundary Based on Sign Relationship")

# Adjust layout
plt.tight_layout()
plt.show()

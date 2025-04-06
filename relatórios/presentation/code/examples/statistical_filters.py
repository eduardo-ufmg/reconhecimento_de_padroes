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

# Plot original data
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
plt.title("XOR Dataset")
plt.show()

# Plot feature distributions
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(X[y == 0, 0], alpha=0.5, color='red')
plt.hist(X[y == 1, 0], alpha=0.5, color='blue')
plt.title("Histogram for x1")

plt.subplot(1, 2, 2)
plt.hist(X[y == 0, 1], alpha=0.5, color='red')
plt.hist(X[y == 1, 1], alpha=0.5, color='blue')
plt.title("Histogram for x2")
plt.show()

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
xx, yy = np.meshgrid(np.linspace(-4, 4, 500), np.linspace(-4, 4, 500))
Z = (np.sign(xx) != np.sign(yy)).astype(int)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
plt.title("Decision Boundary Based on Sign Relationship")
plt.show()

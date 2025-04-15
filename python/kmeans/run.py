import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def xor(n_samples: int, noise: float) -> np.ndarray:
  """
  Generate a dataset of XOR-like points with added noise.
  Args:
    n_samples: total number of samples to generate
    noise: amount of noise to add to the dataset
  Returns:
    X: matrix of input features (n_samples, 2)
  """

  centers = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])

  return np.random.randn(n_samples, 2) * noise + centers[np.random.randint(0, 4, n_samples)]

X = xor(1000, 0.1)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=4)
Y_pred = kmeans.fit_predict(X)

# Sort the samples to group them by cluster
sort_idx = np.argsort(Y_pred)
X = X[sort_idx]
Y_pred = Y_pred[sort_idx]

# Compute the distance matrix
distance_matrix = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Scatter plot of the clustering results
axes[0].scatter(X[:, 0], X[:, 1], c=Y_pred, cmap='viridis', marker='o', edgecolor='k', s=50)
axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200)
axes[0].set_title("KMeans Clustering")

# Plot the distance matrix
im = axes[1].imshow(distance_matrix, cmap='viridis', interpolation='nearest')
axes[1].set_title("Distance Matrix")
fig.colorbar(im, ax=axes[1])

# Show the plots
plt.tight_layout()
plt.show()
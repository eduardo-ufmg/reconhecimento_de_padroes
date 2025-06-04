import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

# Generate synthetic data
X, y = make_moons(n_samples=50, noise=0.1)

# Create two different bootstrap samples from the data
X1, y1 = resample(X, y, random_state=0)
X2, y2 = resample(X, y, random_state=1)

# Train two decision trees on the different samples
tree1 = DecisionTreeClassifier(random_state=2)
tree1.fit(X1, y1)

tree2 = DecisionTreeClassifier(random_state=2)
tree2.fit(X2, y2)

# Create a mesh grid to plot decision boundaries
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Plot the decision boundaries for each tree
plt.figure(figsize=(12, 5))

# Plot for Tree 1
plt.subplot(1, 2, 1)
Z1 = tree1.predict(np.c_[xx.ravel(), yy.ravel()])
Z1 = Z1.reshape(xx.shape)
plt.contourf(xx, yy, Z1, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
plt.title("Decision Tree 1 (Trained on Sample 1)")

# Plot for Tree 2
plt.subplot(1, 2, 2)
Z2 = tree2.predict(np.c_[xx.ravel(), yy.ravel()])
Z2 = Z2.reshape(xx.shape)
plt.contourf(xx, yy, Z2, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
plt.title("Decision Tree 2 (Trained on Sample 2)")

plt.tight_layout()
plt.show()

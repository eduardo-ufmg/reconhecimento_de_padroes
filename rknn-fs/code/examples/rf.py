import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

# Generate a challenging classification dataset
X, y = make_moons(n_samples=1000, noise=0.3)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a Random Forest with few trees (2 in this case)
rf = RandomForestClassifier(n_estimators=2, max_depth=1, bootstrap=True)
rf.fit(X_train, y_train)

# Create a figure to display the forest
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
fig.suptitle("Random Forest Tree Structures", fontsize=16, y=1.05)

# Plot each tree in the forest
for index, (tree, ax) in enumerate(zip(rf.estimators_, axes)):
    plot_tree(
        tree,
        ax=ax,
        feature_names=["x1", "x2"],
        class_names=["0", "1"],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
    )

# Add spacing between subplots
plt.tight_layout()

# Show feature importances
print("Feature importances:", rf.feature_importances_)

# Evaluate performance
train_acc = rf.score(X_train, y_train)
test_acc = rf.score(X_test, y_test)
print(f"\nTraining accuracy: {train_acc:.2f}")
print(f"Test accuracy: {test_acc:.2f}")

plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data with one feature
X = np.random.rand(1000, 1)  # 1000 samples, 1 feature
y_clean = (X[:, 0] < 0.5).astype(int)  # True boundary at X=0.5

# Introduce label noise near the decision boundary
y_noisy = y_clean.copy()
noise_region = (X[:, 0] >= 0.4) & (X[:, 0] <= 0.6)
noise_indices = np.where(noise_region)[0]
np.random.shuffle(noise_indices)
num_noise = int(0.3 * len(noise_indices))  # 30% of boundary region flipped
y_noisy[noise_indices[:num_noise]] = 1 - y_noisy[noise_indices[:num_noise]]

# Split data into training and test sets
X_train, X_test, y_train_clean, y_test_clean = train_test_split(
    X, y_clean, test_size=0.2, random_state=42
)
_, _, y_train_noisy, y_test_noisy = train_test_split(
    X, y_noisy, test_size=0.2, random_state=42
)

# Train decision trees
tree_clean = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clean.fit(X_train, y_train_clean)

tree_noisy = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_noisy.fit(X_train, y_train_noisy)

# Visualize the trees
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plot_tree(tree_clean, feature_names=["X"], class_names=["0", "1"], filled=True)
plt.title("Decision Tree Trained on Clean Data")
plt.subplot(1, 2, 2)
plot_tree(tree_noisy, feature_names=["X"], class_names=["0", "1"], filled=True)
plt.title("Decision Tree Trained on Noisy Data")
plt.show()

# Evaluate performance
clean_accuracy = tree_clean.score(X_test, y_test_clean)
noisy_accuracy = tree_noisy.score(X_test, y_test_clean)  # Test on clean labels

print(f"Clean tree test accuracy: {clean_accuracy:.2f}")
print(f"Noisy tree test accuracy: {noisy_accuracy:.2f}")

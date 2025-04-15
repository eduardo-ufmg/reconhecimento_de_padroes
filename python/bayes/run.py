import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from common.generate import generate_dataset
from bayes.train import train
from bayes.pred import pred

N_SAMPLES = 250

def xor(n_samples, noise):
  X0, Y0, X1, Y1 = generate_dataset('xor', n_samples, noise)
  X = np.vstack((X0, X1))
  Y = np.hstack((Y0, Y1))

  return X, Y

# Generate synthetic datasets with reproducibility
datasets = [
  ('Linear', make_classification(
    n_samples=N_SAMPLES, n_features=2, n_redundant=0,
    n_clusters_per_class=1, class_sep=2
  )),
  ('Moons', make_moons(n_samples=N_SAMPLES, noise=0.3)),
  ('Circles', make_circles(n_samples=N_SAMPLES, noise=0.2, factor=0.5)),
  ('XOR', xor(N_SAMPLES, 0.1))
]

methods = ['normal', 'gaussian_mix', 'kde']
results = {}

# Create output directory
os.makedirs('bayes/output', exist_ok=True)

# Create a single figure with subplots
fig, axes = plt.subplots(len(datasets), len(methods))

for row, (dataset_name, (X, y)) in enumerate(datasets):
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
  )
  
  # Split training data into classes
  X0_train = X_train[y_train == 0]
  X1_train = X_train[y_train == 1]
  
  # Prepare mesh grid
  h = 0.02
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
             np.arange(y_min, y_max, h))
  mesh_points = np.c_[xx.ravel(), yy.ravel()]
  
  dataset_results = {}
  
  for col, method in enumerate(methods):
    # Train and predict
    model_args0, model_args1 = train(X0_train, X1_train, method)
    y_pred = pred(X_test, model_args0, model_args1, method)
    acc = accuracy_score(y_test, y_pred)
    dataset_results[method] = acc
    
    # Decision boundary
    Z = pred(mesh_points, model_args0, model_args1, method)
    Z = Z.reshape(xx.shape)
    
    # Plotting
    ax = axes[row, col]
    ax.contourf(xx, yy, Z, alpha=0.8, cmap='RdYlBu')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolor='k', s=20)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(method.lower())
  
  results[dataset_name] = dataset_results

plt.suptitle('Decision Boundaries')
plt.savefig('bayes/output/boundaries.png', bbox_inches='tight')
plt.show()
plt.close()

# Print results
print("Evaluation Results:")
for dataset, res in results.items():
  print(f"\nDataset: {dataset}")
  for method, acc in res.items():
    print(f"{method.upper():<12}: {acc:.4f}")
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

from bayes.pred import pred, pred_multivariate

def unidimensional(samples: int=100):
  X1 = np.random.normal(loc=-np.random.uniform(0, 5), scale=np.random.uniform(0, 2), size=samples)
  X2 = np.random.normal(loc=+np.random.uniform(0, 5), scale=np.random.uniform(0, 2), size=samples)

  X = np.linspace(min(X1) - 1, max(X2) + 1, 1000)

  y = pred(X, X1, 0, X2, 1)

  plt.figure(figsize=(10, 6))
  plt.plot(X, y, label="Decision Boundary", color="blue")
  plt.scatter(X1, np.zeros_like(X1), color="red", alpha=0.6)
  plt.scatter(X2, np.zeros_like(X2), color="blue", alpha=0.6)
  plt.xlabel("X")
  plt.ylabel("Class Label")
  plt.title("Decision Boundary and Data Distribution")
  plt.grid(True)
  plt.show()

def bidimensional(samples: int=100):
  X1 = np.random.multivariate_normal(mean=[-np.random.uniform(0, 5), -np.random.uniform(0, 5)],
                                     cov=np.eye(2) * np.random.uniform(0, 2),
                                     size=samples)
  X2 = np.random.multivariate_normal(mean=[+np.random.uniform(0, 5), +np.random.uniform(0, 5)],
                                     cov=np.eye(2) * np.random.uniform(0, 2),
                                     size=samples)

  X = np.linspace(min(X1[:, 0]) - 1, max(X2[:, 0]) + 1, 100)
  Y = np.linspace(min(X1[:, 1]) - 1, max(X2[:, 1]) + 1, 100)

  XX, YY = np.meshgrid(X, Y)
  grid = np.c_[XX.ravel(), YY.ravel()]

  Z = pred_multivariate(grid, X1, 0, X2, 1)
  Z = Z.reshape(XX.shape)

  fig, axes = plt.subplots(1, 2, figsize=(20, 8))

  # Subfigure 1: Decision Boundary
  axes[0].contourf(XX, YY, Z, levels=1, alpha=0.6, colors=["red", "blue"])
  axes[0].scatter(X1[:, 0], X1[:, 1], color="red", alpha=0.6)
  axes[0].scatter(X2[:, 0], X2[:, 1], color="blue", alpha=0.6)
  axes[0].set_xlabel("X1")
  axes[0].set_ylabel("X2")
  axes[0].set_title("Decision Boundary and Data Distribution (2D)")
  axes[0].grid(True)

  # Estimate PDFs for X1 and X2
  pdf_X1 = multivariate_normal(mean=np.mean(X1, axis=0), cov=np.cov(X1, rowvar=False))
  pdf_X2 = multivariate_normal(mean=np.mean(X2, axis=0), cov=np.cov(X2, rowvar=False))

  # Compute PDF values for the grid
  pdf1 = pdf_X1.pdf(grid).reshape(XX.shape)
  pdf2 = pdf_X2.pdf(grid).reshape(XX.shape)

  # Subfigure 2: 3D PDFs
  ax = fig.add_subplot(122, projection='3d')
  ax.plot_surface(XX, YY, pdf1, cmap="Reds")
  ax.plot_surface(XX, YY, pdf2, cmap="Blues")
  ax.set_xlabel("X1")
  ax.set_ylabel("X2")
  ax.set_zlabel("PDF")
  ax.set_title("3D PDFs of Class Distributions")

  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
  SAMPLES = 1000

  bidimensional(SAMPLES)

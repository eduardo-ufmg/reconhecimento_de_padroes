import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from bayes.train import train, GaussianParams
from bayes.pred import predict, multivariate_normal

def visualize_results(
  X1: np.ndarray, X2: np.ndarray,
  params1: GaussianParams, params2: GaussianParams,
  grid: np.ndarray, predictions: np.ndarray,
  xx: np.ndarray, yy: np.ndarray,
  method: str
):
  """Enhanced visualization matching classification method."""
  from bayes.pred import _vectorized_gaussian_mixture
  from sklearn.neighbors import KernelDensity

  fig = plt.figure(figsize=(20, 8))
  
  # Decision Boundary
  ax1 = fig.add_subplot(121)
  ax1.contourf(xx, yy, predictions.reshape(xx.shape), alpha=0.3, levels=1, colors=['#FFAAAA', '#AAAAFF'])
  ax1.scatter(X1[:, 0], X1[:, 1], c='red', edgecolors='k', alpha=0.6)
  ax1.scatter(X2[:, 0], X2[:, 1], c='blue', edgecolors='k', alpha=0.6)
  ax1.set_title(f'Decision Boundary ({method.replace("_", " ").title()})')

  # 3D Density Plot
  ax2 = fig.add_subplot(122, projection='3d')
  
  # Compute densities based on classification method
  if method == "normal":
    Z1 = multivariate_normal(params1.mean, params1.cov).pdf(grid)
    Z2 = multivariate_normal(params2.mean, params2.cov).pdf(grid)
  elif method == "gaussian_mix":
    Z1 = _vectorized_gaussian_mixture(grid, X1, params1.cov)
    Z2 = _vectorized_gaussian_mixture(grid, X2, params2.cov)
  elif method == "kde":
    kde1 = KernelDensity(bandwidth=0.2).fit(X1)
    Z1 = np.exp(kde1.score_samples(grid))
    kde2 = KernelDensity(bandwidth=0.2).fit(X2)
    Z2 = np.exp(kde2.score_samples(grid))
  
  # Reshape and plot
  Z1 = Z1.reshape(xx.shape)
  Z2 = Z2.reshape(xx.shape)
  ax2.plot_surface(xx, yy, Z1, cmap='Reds', alpha=0.5, antialiased=False)
  ax2.plot_surface(xx, yy, Z2, cmap='Blues', alpha=0.5, antialiased=False)
  ax2.set_title(f'Class Densities ({method.replace("_", " ").title()})')
  ax2.set_zlabel('Probability Density')

  plt.tight_layout()
  plt.show()

def evaluate_model(X_test: np.ndarray, y_test: np.ndarray, predictions: np.ndarray):
  """Display classification metrics."""
  cm = confusion_matrix(y_test, predictions)
  disp = ConfusionMatrixDisplay(cm)
  disp.plot(cmap='Blues')
  plt.title('Confusion Matrix')
  plt.show()

def bidimensional(dataset_type: str, method: str, samples: int = 1000, noise: float = 0.1):
  """Enhanced evaluation with visualization and metrics."""
  from common.generate import generate_dataset  # Assume this exists
  X1, y1, X2, y2 = generate_dataset(dataset_type, samples, noise)
  params1, params2 = train(X1, 0, X2, 1)
  
  # Create prediction grid
  x_min, x_max = min(X1[:,0].min(), X2[:,0].min()) - 1, max(X1[:,0].max(), X2[:,0].max()) + 1
  y_min, y_max = min(X1[:,1].min(), X2[:,1].min()) - 1, max(X1[:,1].max(), X2[:,1].max()) + 1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
  grid = np.c_[xx.ravel(), yy.ravel()]
  
  # Predict using Gaussian mixture
  preds = predict(grid, X1, params1, X2, params2, method)
  
  # Visualize
  visualize_results(X1, X2, params1, params2, grid, preds, xx, yy, method)
  
  # For demonstration
  X = np.vstack((X1, X2))
  Xt = X + np.random.normal(0, 0.5, X.shape)
  yt = np.hstack((y1, y2))
  yt[yt == -1] = 0

  # Predict on test set
  yp = predict(Xt, X1, params1, X2, params2, method)

  # Evaluate model
  evaluate_model(Xt, yt, yp)

if __name__ == "__main__":
  bidimensional('spirals', 'kde', 1000, 0.25)

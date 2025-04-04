import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from bayes.train import train, GaussianParams
from bayes.pred import predict, multivariate_normal

def visualize_results(
  X1: np.ndarray, X2: np.ndarray,
  params1: GaussianParams, params2: GaussianParams,
  grid: np.ndarray, predictions: np.ndarray,
  xx: np.ndarray, yy: np.ndarray
):
  """Create enhanced visualizations."""
  fig, axs = plt.subplots(1, 2, figsize=(20, 8))
  
  # Decision boundary plot
  axs[0].contourf(xx, yy, predictions.reshape(xx.shape), alpha=0.3, levels=1, colors=['#FFAAAA', '#AAAAFF'])
  axs[0].scatter(X1[:, 0], X1[:, 1], c='red', edgecolors='k', alpha=0.6)
  axs[0].scatter(X2[:, 0], X2[:, 1], c='blue', edgecolors='k', alpha=0.6)
  axs[0].set_title('Decision Boundary')
  
  # 3D PDF plot
  X_plot = np.linspace(grid[:,0].min(), grid[:,0].max(), 100)
  Y_plot = np.linspace(grid[:,1].min(), grid[:,1].max(), 100)
  XX, YY = np.meshgrid(X_plot, Y_plot)
  Z1 = multivariate_normal(params1.mean, params1.cov).pdf(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
  Z2 = multivariate_normal(params2.mean, params2.cov).pdf(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
  
  ax = fig.add_subplot(122, projection='3d')
  ax.plot_surface(XX, YY, Z1, cmap='Reds', alpha=0.5)
  ax.plot_surface(XX, YY, Z2, cmap='Blues', alpha=0.5)
  ax.set_title('Class Probability Densities')
  
  plt.tight_layout()
  plt.show()

def evaluate_model(X_test: np.ndarray, y_test: np.ndarray, predictions: np.ndarray):
  """Display classification metrics."""
  cm = confusion_matrix(y_test, predictions)
  disp = ConfusionMatrixDisplay(cm)
  disp.plot(cmap='Blues')
  plt.title('Confusion Matrix')
  plt.show()

def bidimensional(dataset_type: str = 'moons', samples: int = 1000, noise: float = 0.1):
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
  preds = predict(grid, X1, params1, X2, params2, method="kde")
  
  # Visualize
  visualize_results(X1, X2, params1, params2, grid, preds, xx, yy)
  
  # For demonstration, assume test data is available
  # evaluate_model(X_test, y_test, preds)

if __name__ == "__main__":
  bidimensional('moons', 1000, 0.1)

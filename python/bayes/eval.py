import numpy as np
import matplotlib.pyplot as plt

from bayes.pred import gaussian, pred

if __name__ == "__main__":
  SAMPLES = 100

  X1 = np.random.normal(loc=-5, scale=1, size=SAMPLES)
  X2 = np.random.normal(loc=+5, scale=2, size=SAMPLES)

  X = np.random.uniform(low=-10, high=+10, size=SAMPLES)

  y = pred(X, X1, 0, X2, 1)

  plt.figure(figsize=(10, 6))

  # Plot the points
  plt.scatter(X1, np.zeros(X1.shape[0]), color='blue', alpha=0.6)
  plt.scatter(X2, np.zeros(X2.shape[0]), color='red', alpha=0.6)

  # Plot the PDFs
  Xset = np.concatenate((X1, X2))
  x_range = np.linspace(min(Xset) - 1, max(Xset) + 1, 1000)
  pdf1 = gaussian(x_range, np.mean(X1), np.std(X1))
  pdf2 = gaussian(x_range, np.mean(X2), np.std(X2))

  plt.plot(x_range, pdf1, color='blue', label='PDF Class 0')
  plt.plot(x_range, pdf2, color='red', label='PDF Class 1')

  # Find and plot the decision boundary
  decision_boundary = x_range[np.abs(pdf1 - pdf2).argmin()]
  plt.axvline(decision_boundary, color='green', linestyle='--', label='Decision Boundary')

  plt.title("Data Points, PDFs, and Decision Boundary")
  plt.xlabel("X")
  plt.ylabel("Density")
  plt.show()

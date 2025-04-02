import numpy as np
import matplotlib.pyplot as plt

from bayes.pred import gaussian, pred

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


if __name__ == "__main__":
  SAMPLES = 100

  unidimensional(SAMPLES)

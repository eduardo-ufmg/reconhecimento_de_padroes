import numpy as np
import matplotlib.pyplot as plt

from common.generate import generate_dataset, generate_grid
from perceptron.pred import pred
from perceptron.train import train
from common.plot import plot_decision_boundary

if __name__ == "__main__":
  X1, Y1, X2, Y2 = generate_dataset('blobs', 100, 2)

  # Combine the two classes into a single dataset
  X = np.vstack((X1, X2))
  Y = np.vstack((Y1, Y2)).flatten()

  # Train the perceptron
  w_guess = np.random.rand(X.shape[1])
  t_guess = np.random.rand(1)
  
  w, t, errors = train(X, Y, w_guess, t_guess, 0.1, 1000)

  # Generate a grid of points for plotting decision boundaries
  G = generate_grid(X, 1)
  P = np.zeros(G.shape[0])

  # Predict the class labels for the grid points
  for i in range(G.shape[0]):
    P[i] = pred(G[i], w, t)

  # Plot the dataset and decision boundary
  plot_decision_boundary(X, Y, G, P, "blobs")
  plt.show()



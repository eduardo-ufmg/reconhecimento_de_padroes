import numpy as np

from common.generate import generate_dataset, generate_grid
from kNN.mykNN import mykNN_batch
from common.plot import plot_decision_boundary

if __name__ == "__main__":
  SAMPLES = 500
  LINEAR_DATASET = 'blobs'
  NONLIN_DATASET = 'spirals'

  X1l, Y1l, X2l, Y2l = generate_dataset(LINEAR_DATASET, SAMPLES, noise=1)
  X = np.vstack((X1l, X2l))
  Y = np.hstack((Y1l, Y2l)).flatten()

  G = generate_grid(X, 1)

  P = mykNN_batch(G, X, Y, k=5, h=1)

  plot_decision_boundary(X, Y, G, P, LINEAR_DATASET)
  
  
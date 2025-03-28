import numpy as np

from common.generate import generate_dataset, generate_grid
from kNN.mykNN import mykNN_batch, to_characteristic_space
from common.plot import plot_decision_boundary, plot_characteristic_space

if __name__ == "__main__":
  SAMPLES = 500
  LINEAR_DATASET = 'blobs'
  NONLIN_DATASET = 'spirals'
  K = SAMPLES // 10

  X1l, Y1l, X2l, Y2l = generate_dataset(LINEAR_DATASET, SAMPLES, noise=2)
  X = np.vstack((X1l, X2l))
  Y = np.hstack((Y1l, Y2l)).flatten()

  G = generate_grid(X, 1)

  P = mykNN_batch(G, X, Y, k=K, h=1)

  plot_decision_boundary(X, Y, G, P, LINEAR_DATASET)
  
  QX1, QX2 = to_characteristic_space(X, X, Y, k=K, h=1)
  QG1, QG2 = to_characteristic_space(G, X, Y, k=K, h=1)

  QX = np.vstack((QX1, QX2))
  QG = np.vstack((QG1, QG2))

  plot_characteristic_space(QX, Y, QG, P, LINEAR_DATASET)
  
  
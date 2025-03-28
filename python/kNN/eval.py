import numpy as np

from common.generate import generate_dataset, generate_grid
from kNN.mykNN import mykNN_batch, to_characteristic_space
from common.plot import plot_decision_boundary, plot_characteristic_space

if __name__ == "__main__":
  SAMPLES = 500
  LINEAR_DATASET = 'blobs'
  NONLIN_DATASET = 'spirals'
  K = 5
  H = 1

  SETS = [LINEAR_DATASET, NONLIN_DATASET]
  NOIS = [1.5, 0.001]

  for dataset, noise in zip(SETS, NOIS):
    X1, Y1, X2, Y2 = generate_dataset(dataset, SAMPLES, noise=noise)
    X = np.vstack((X1, X2))
    Y = np.hstack((Y1, Y2)).flatten()

    G = generate_grid(X, 1)

    P = mykNN_batch(G, X, Y, k=K, h=H)

    plot_decision_boundary(X, Y, G, P, dataset)

    QX1, QX2 = to_characteristic_space(X, X, Y, k=K, h=H)
    QG1, QG2 = to_characteristic_space(G, X, Y, k=K, h=H)

    QX = np.vstack((QX1, QX2))
    QG = np.vstack((QG1, QG2))

    plot_characteristic_space(QX, Y, QG, P, dataset)
  
import numpy as np
from sklearn.model_selection import KFold

from kNN.mykNN import mykNN
from kNN.k_optimization.load_prepare_occupancy import load_prepare_occup
from kNN.k_optimization.prepare_set import prepare_set

X, y = load_prepare_occup()

kf = KFold(n_splits=10, shuffle=True, random_state=1)

for train_index, test_index in kf.split(X):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

  complete_set, X_test, y_pred = prepare_set(X_train, y_train, X_test, y_test)

  for i, x in enumerate(X_test):
    y_pred[i] = mykNN(x, complete_set, k=5, h=1.0)

  accuracy = np.mean(y_pred == y_test.values.ravel())
  print(f"mykNN Fold accuracy: {accuracy:.2f}")

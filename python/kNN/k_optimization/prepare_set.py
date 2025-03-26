import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_set(X_train, y_train, X_test, y_test):
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  y_pred = np.zeros(y_test.shape)
  complete_set = np.hstack((X_train_scaled, y_train.values.reshape(-1, 1)))

  return complete_set, X_test_scaled, y_pred
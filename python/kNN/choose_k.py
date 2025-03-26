import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import KFold

from kNN.mykNN import mykNN

# fetch dataset
occupancy_detection = fetch_ucirepo(id=357)

# Create explicit copies of the DataFrames to avoid views
X = occupancy_detection.data.features.copy()
y = occupancy_detection.data.targets.copy()

# Convert 'date' to numerical timestamp (float)
X['date'] = (
  pd.to_datetime(X['date'], errors='coerce')
  .astype('int64')        # Convert to nanoseconds (int64)
  .floordiv(10**9)        # Convert to seconds (integer division)
  .astype(float)          # Convert to float
)

# Convert other columns to float
for column in X.columns.drop('date'):
  X[column] = pd.to_numeric(X[column], errors='coerce')

# Transform y: 1 stays 1, 0 becomes -1
y = y.map(lambda val: 1 if val == 1 else -1)

# k-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=1)

for train_index, test_index in kf.split(X):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

  y_pred = np.zeros(y_test.shape)

  complete_set = np.hstack((X_train.values, y_train.values.reshape(-1, 1)))

  for i, x in enumerate(X_test.values):
    y_pred[i] = mykNN(x, complete_set, k=5, h=1.0)

  accuracy = np.mean(y_pred == y_test.values)
  print(f"Fold accuracy: {accuracy:.2f}")

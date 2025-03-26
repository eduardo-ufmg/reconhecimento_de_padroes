import pandas as pd

from ucimlrepo import fetch_ucirepo

def load_prepare_occup():
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

  # Drop rows with NaN values in X and y
  X = X.dropna()
  y = y.loc[X.index]

  return X, y

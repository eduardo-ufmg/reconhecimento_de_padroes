import pandas as pd

from ucimlrepo import fetch_ucirepo

def load_prepare_occup():
  # Fetch the occupancy detection dataset from the UCI repository
  occupancy_detection = fetch_ucirepo(id=357)
  
  # Extract features (X) and targets (y) from the dataset
  X = occupancy_detection.data.features.copy()
  y = occupancy_detection.data.targets.copy()
  
  # Convert the 'date' column to a UNIX timestamp (seconds since epoch)
  X['date'] = (
    pd.to_datetime(X['date'], errors='coerce')  # Convert to datetime, handling errors
    .astype('int64')                           # Convert datetime to int64 (nanoseconds)
    .floordiv(10**9)                           # Convert nanoseconds to seconds
    .astype(float)                             # Ensure the column is of float type
  )

  # Convert all other columns to numeric, coercing invalid values to NaN
  for column in X.columns.drop('date'):
    X[column] = pd.to_numeric(X[column], errors='coerce')

  # Map target values: 1 remains 1, others are converted to -1
  y = y.map(lambda val: 1 if val == 1 else -1)

  # Drop rows with NaN values in the features
  X = X.dropna()
  
  # Align the target values (y) with the cleaned feature indices
  y = y.loc[X.index]

  # Return the cleaned and prepared features (X) and targets (y)
  return X, y

import pandas as pd
import numpy as np

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
    .astype('int64')                            # Convert datetime to int64 (nanoseconds)
    .floordiv(10**9)                            # Convert nanoseconds to seconds
    .astype(float)                              # Ensure the column is of float type
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


def load_prepare_uciset(id, nan_strategy='drop', max_categories=10):

  try:
    dataset = fetch_ucirepo(id=id)
  except Exception as e:
    raise ValueError(f"Failed to fetch dataset {id}: {str(e)}") from e

  # Validate dataset structure
  if not hasattr(dataset, 'data') or not hasattr(dataset.data, 'features'):
    raise ValueError(f"Dataset {id} has invalid structure")
  
  # Initialize features and targets
  X = pd.DataFrame()
  y = pd.Series(dtype='float64')

  # Handle features
  if isinstance(dataset.data.features, pd.DataFrame):
    X = dataset.data.features.copy()
  else:
    try:
      X = pd.DataFrame(dataset.data.features)
    except Exception as e:
      raise ValueError(f"Failed to parse features for dataset {id}: {str(e)}") from e

  # Handle targets
  if hasattr(dataset.data, 'targets'):
    if isinstance(dataset.data.targets, (pd.DataFrame, pd.Series)):
      y = dataset.data.targets.copy().squeeze()
    else:
      try:
        y = pd.Series(dataset.data.targets).squeeze()
      except Exception as e:
        raise ValueError(f"Failed to parse targets for dataset {id}: {str(e)}") from e

  # Process datetime columns
  datetime_cols = []
  for col in X.select_dtypes(include=['object', 'datetime64']).columns:
    try:
      X[col] = pd.to_datetime(X[col], format='%Y-%m-%d %H:%M:%S', errors='raise')
      datetime_cols.append(col)
    except (TypeError, ValueError):
      pass

  for col in datetime_cols:
    X[col] = X[col].astype('int64') // 10**9

  # Process categorical columns
  categorical_cols = X.select_dtypes(include=['object', 'category']).columns
  for col in categorical_cols:
    if X[col].nunique() <= max_categories:
      dummies = pd.get_dummies(X[col], prefix=col)
      X = pd.concat([X, dummies], axis=1)
      X.drop(col, axis=1, inplace=True)
    else:
      X[col] = X[col].factorize()[0]

  # Ensure unique column names
  new_cols = []
  seen = {}
  for col in X.columns:
    if col in seen:
      seen[col] += 1
      new_col = f"{col}_{seen[col]}"
    else:
      seen[col] = 0
      new_col = col
    new_cols.append(new_col)
  X.columns = new_cols

  # Convert remaining columns to numeric
  for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

  # Handle missing values
  if nan_strategy == 'drop':
    X = X.dropna()
  elif nan_strategy == 'fill_mean':
    X = X.fillna(X.mean())
  elif nan_strategy == 'fill_median':
    X = X.fillna(X.median())
  else:
    raise ValueError(f"Invalid nan_strategy: {nan_strategy}")

  # Align targets with cleaned features
  y = y.loc[X.index]
  
  # Process targets
  unique_targets = y.unique()
  if len(unique_targets) == 2:
    y = y.map({unique_targets[0]: -1, unique_targets[1]: 1})
  elif len(unique_targets) > 2:
    pass
  
  y = pd.to_numeric(y, errors='coerce').dropna()
  X = X.loc[y.index]

  return X.astype(np.float64), y.astype(np.float64)

def window_bigset(X, y, window_size=1000):
  """Split a large dataset into smaller windows."""
  
  windows = []
  for i in range(0, len(X), window_size):
    window = slice(i, i + window_size)
    windows.append((X[window], y[window]))

  num_instances, num_features = X.shape
  return windows, num_instances, num_features

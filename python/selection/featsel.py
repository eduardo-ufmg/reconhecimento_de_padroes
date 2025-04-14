import numpy as np
import pandas as pd

def drop_highcorr(X, y, threshold=0.8):
  """
  Drop features that are highly correlated with each other.
  Args:
    X: DataFrame of features (n_samples, n_features)
    threshold: Correlation threshold for dropping features

  Returns:
    DataFrame with highly correlated features dropped
  """
  
  # Compute the absolute correlation matrix between features
  corr_matrix = X.corr().abs()

  # Extract the upper triangle to avoid duplicate pairs
  upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

  # Flatten the matrix and filter pairs above the threshold
  high_corr_pairs = corr_matrix.where(upper_tri).stack()
  high_corr_pairs = high_corr_pairs[high_corr_pairs > threshold].sort_values(ascending=False)

  # Compute each feature's absolute correlation with the target variable
  y_corr = X.apply(lambda col: col.corr(y)).abs()

  # Determine which features to drop
  dropped_features = set()

  for (feature_i, feature_j) in high_corr_pairs.index:
    if feature_i in dropped_features or feature_j in dropped_features:
      continue
    # Keep the feature with higher correlation to the target
    if y_corr[feature_i] > y_corr[feature_j]:
      dropped_features.add(feature_j)
    else:
      dropped_features.add(feature_i)

  # Drop the selected features from the dataset
  return X.drop(columns=dropped_features)

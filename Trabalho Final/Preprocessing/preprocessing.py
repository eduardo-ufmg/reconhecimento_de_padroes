import logging
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from typing_extensions import Self
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # Import PCA

class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Transformer to remove highly correlated features."""
    def __init__(self, threshold: np.float64 = np.float64(0.9)) -> None:
        self.threshold = threshold
        self.to_drop_: NDArray[np.int32] | None = None # Features to drop

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int32] | None = None) -> Self:
        X = check_array(X, dtype=np.float64)
        if X.shape[1] < 2:
            self.to_drop_ = np.array([], dtype=int)
            return self # Not enough features to compare correlations

        corr_matrix = np.corrcoef(X, rowvar=False)
        # Ensure corr_matrix is 2D, even if X has only one feature (np.corrcoef might return scalar)
        corr_matrix = np.atleast_2d(corr_matrix)

        upper_triangle_mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        highly_correlated_pairs = (np.abs(corr_matrix) > self.threshold) & upper_triangle_mask
        
        # Get indices of columns to drop (prefer dropping the second feature in a pair)
        self.to_drop_ = np.unique(np.where(highly_correlated_pairs)[1]).astype(np.int32)
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        check_is_fitted(self, 'to_drop_')
        X = check_array(X, dtype=np.float64)

        if self.to_drop_ is None or self.to_drop_.size == 0:
            return X.copy()
        
        # Ensure to_drop_ indices are valid for X's shape
        valid_to_drop = self.to_drop_[self.to_drop_ < X.shape[1]]
        if len(valid_to_drop) < len(self.to_drop_):
            logging.warning("CorrelationFilter: Some indices to drop were out of bounds for the input X.")

        return np.delete(X, valid_to_drop, axis=1)
    
class Preprocessor(BaseEstimator, TransformerMixin):
    """Pipeline for preprocessing data with variance threshold, correlation filter, scaling, and PCA."""
    
    def __init__(self) -> None:
        self.pipeline = Pipeline([
            ('variance_threshold', VarianceThreshold(threshold=0.05)),
            ('correlation_filter', CorrelationFilter(threshold=np.float64(0.95))),
            ('scaler', StandardScaler()),
            ('pca', PCA())
        ])

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int32] | None = None) -> Self:
        self.pipeline.fit(X, y)
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.pipeline.transform(X)


import os
import sys

import numpy as np

from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Kernel.kernel import kernel_matrix
from Dissimilarity.dissimilarity import objective_function_dissimilarity
from SpatialSpread.spatial_spread import objective_function_spatial_spread


class OptimizationResult:
    h_opt: float
    scaled_inv_cov: np.ndarray
    scaled_norm_factor: float
    K_matrix_opt: np.ndarray

    def __init__(
        self,
        h_opt: float,
        scaled_inv_cov: np.ndarray,
        scaled_norm_factor: float,
        K_matrix_opt: np.ndarray,
    ):
        self.h_opt = h_opt
        self.scaled_inv_cov = scaled_inv_cov
        self.scaled_norm_factor = scaled_norm_factor
        self.K_matrix_opt = K_matrix_opt


def optimize_h(
    X: np.ndarray,
    y: np.ndarray,
    inv_cov: np.ndarray,
    norm_factor: float,
    optimization_metric: str,
) -> OptimizationResult:
    raise NotImplementedError(
        "This function should be implemented to optimize the bandwidth parameter 'h' based on the provided optimization metric."
    )


class CustomSVC(BaseEstimator, ClassifierMixin):
    optimization_metric_: str
    inv_cov_: np.ndarray
    norm_factor_: float
    h_opt_: float
    classes_: np.ndarray
    svc_: SVC

    def __init__(self, optimization_metric: str | None = None):

        if optimization_metric is None or optimization_metric not in [
            "dissimilarity",
            "spatial_spread",
        ]:
            raise ValueError(
                "Invalid optimization metric. Choose 'dissimilarity' or 'spatial_spread'."
            )

        self.optimization_metric_ = optimization_metric

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomSVC":

        self.classes_ = np.unique(y)

        cov = np.cov(X, rowvar=False)
        cov_det = np.linalg.det(cov)
        n_features = X.shape[1]

        self.inv_cov_ = np.linalg.inv(cov)
        self.norm_factor_ = 1.0 / np.sqrt((2 * np.pi) ** n_features * cov_det)

        optimization_result = optimize_h(
            X, y, self.inv_cov_, self.norm_factor_, self.optimization_metric_
        )

        self.h_opt_ = optimization_result.h_opt
        self.inv_cov_ = optimization_result.scaled_inv_cov
        self.norm_factor_ = optimization_result.scaled_norm_factor
        K_matrix_opt = optimization_result.K_matrix_opt

        self.svc_ = SVC(kernel="precomputed").fit(K_matrix_opt, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.svc_.predict(
            kernel_matrix(X, self.h_opt_, self.inv_cov_, self.norm_factor_)
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(accuracy_score(y, self.predict(X)))

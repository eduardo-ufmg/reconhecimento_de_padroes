import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from Dissimilarity.dissimilarity import objective_function_dissimilarity
from Kernel.kernel import kernel_matrix
from SpatialSpread.spatial_spread import objective_function_spatial_spread


class OptimizationResult:
    h_opt: float
    H: np.ndarray
    K_matrix_opt: np.ndarray

    def __init__(
        self,
        h_opt: float,
        K_matrix_opt: np.ndarray,
    ):
        self.h_opt = h_opt
        self.K_matrix_opt = K_matrix_opt


def optimize_h(
    X: np.ndarray, y: np.ndarray, optimization_metric: str
) -> OptimizationResult:
    def objective(h_val):
        # This is the function that minimize_scalar will optimize
        K_matrix = kernel_matrix(X, h_val)
        if optimization_metric == "dissimilarity":
            Warning(
                "The 'dissimilarity' metric is not implemented yet. Using 'spatial_spread' instead."
            )
            return objective_function_spatial_spread(K_matrix, y)
        elif optimization_metric == "spatial_spread":
            return objective_function_spatial_spread(K_matrix, y)
        else:
            raise ValueError("Invalid optimization metric.")

    # Define bounds for h. This is crucial for 'bounded' method and good practice generally.
    # The range should be chosen based on the expected scale of your data.
    # For a bandwidth, it's typically positive, often in a logarithmic scale.
    bounds_h = (1e-1, 1e1)  # Example bounds: h can be between 0.1 and 10.0

    # Use minimize_scalar with the 'bounded' method
    result = minimize_scalar(objective, bounds=bounds_h, method="bounded")

    if not result.success:
        raise RuntimeError(f"Optimization for h failed: {result.message}")

    best_h = result.x

    best_K_matrix = kernel_matrix(X, best_h)

    return OptimizationResult(best_h, best_K_matrix)


class CustomSVC(BaseEstimator, ClassifierMixin):
    optimization_metric_: str
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

        optimization_result = optimize_h(X, y, self.optimization_metric_)

        self.h_opt_ = optimization_result.h_opt
        K_matrix_opt = optimization_result.K_matrix_opt

        self.svc_ = SVC(kernel="precomputed").fit(K_matrix_opt, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.svc_.predict(kernel_matrix(X, self.h_opt_))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(accuracy_score(y, self.predict(X)))

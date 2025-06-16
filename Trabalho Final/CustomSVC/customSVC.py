import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ConvexHull.convex_hull import objective_function_convex_hull_intersection_area
from Dissimilarity.dissimilarity import objective_function_dissimilarity
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
    """Optimizes the kernel bandwidth 'h' based on the chosen metric."""

    def objective(h_val: float) -> float:
        """Objective function for the optimizer."""
        if h_val <= 1e-6:  # Prevent h from being too close to zero
            return np.inf

        K_matrix = kernel_matrix(X, h=h_val)
        if optimization_metric == "dissimilarity":
            return -objective_function_dissimilarity(K_matrix, y)
        elif optimization_metric == "spatial_spread":
            return -objective_function_spatial_spread(K_matrix, y)
        elif optimization_metric == "convex_hull":
            return -objective_function_convex_hull_intersection_area(K_matrix, y)
        else:
            raise ValueError(f"Invalid optimization metric: {optimization_metric}")

    # --- Data-driven bounds for h ---
    # Heuristic: Set bounds based on percentiles of pairwise distances
    # to ensure the search range is relevant to the data's scale.
    if X.shape[0] > 1:
        pairwise_dists = pdist(X, "euclidean")
        # Use 10th and 90th percentiles as a robust range estimate
        lower_bound = np.percentile(pairwise_dists, 10)
        upper_bound = np.percentile(pairwise_dists, 90)
    else:
        # Fallback for single-sample data
        lower_bound, upper_bound = 0.1, 10.0

    # Ensure bounds are reasonable
    if lower_bound < 1e-5:
        lower_bound = 1e-5
    if upper_bound <= lower_bound:
        upper_bound = lower_bound * 100  # Ensure upper > lower

    result = minimize_scalar(
        objective, bounds=(lower_bound, upper_bound), method="bounded"
    )

    if not result.success:
        raise RuntimeError(f"Optimization for h failed: {result.message}")

    best_h = result.x
    best_K_matrix = kernel_matrix(X, h=best_h)

    return OptimizationResult(best_h, best_K_matrix)


class CSVC(BaseEstimator, ClassifierMixin):
    optimization_metric: str
    h_opt_: float
    classes_: np.ndarray
    svc_: SVC
    X_train_: np.ndarray

    def __init__(self, optimization_metric: str | None = None):

        valid_metrics = ["dissimilarity", "spatial_spread", "convex_hull"]
        if optimization_metric is None or optimization_metric not in valid_metrics:
            raise ValueError(
                f"Invalid optimization metric. Choose from {valid_metrics}"
            )

        self.optimization_metric = optimization_metric

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSVC":

        self.classes_ = np.unique(y)
        self.X_train_ = X

        optimization_result = optimize_h(X, y, self.optimization_metric)

        self.h_opt_ = optimization_result.h_opt
        K_matrix_opt = optimization_result.K_matrix_opt

        self.svc_ = SVC(kernel="precomputed").fit(K_matrix_opt, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        K_pred = kernel_matrix(X, self.X_train_, self.h_opt_)
        return self.svc_.predict(K_pred)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(accuracy_score(y, self.predict(X)))

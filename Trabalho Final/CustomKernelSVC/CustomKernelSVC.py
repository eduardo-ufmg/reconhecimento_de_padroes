import os
import sys
import numpy as np

from numpy.typing import NDArray
from typing import cast, Any
from typing_extensions import Self

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize_scalar

import logging
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Kernel.kernel import kernel, kernel_fit 
from AxisSpread.vector_spread import objective_function as vecspd_objfunc
from twoDSpreadDistance.twod_spread_distance import objective_function as twod_objfunc

# Default bounds for h optimization
DEFAULT_H_BOUNDS: tuple[np.float64, np.float64] = (np.float64(1e-1), np.float64(1e1))

class OptimizationResult:
    success: bool
    x: np.float64
    message: str | None = None

    def __init__(self, success: bool, x: np.float64, message: str | None = None):
        self.success = success
        self.x = x
        self.message = message


class CustomKernelSVC(BaseEstimator, ClassifierMixin):
    """
    Custom SVM using a precomputed kernel where the kernel parameter 'h'
    is optimized using scipy.optimize.minimize_scalar based on a custom metric.
    """
    def __init__(
        self,
        h_bounds: tuple[np.float64, np.float64] | None = None,
        kernel_fit_type: str = 'scale',
        objective_metric: str = 'spatial',
        svm_kwargs: dict[str, Any] | None = None
    ) -> None:
        """
        Initialize the SVM.

        Parameters:
        -----------
        h_bounds : tuple[np.float64, np.float64], optional
            The lower and upper bounds for the 'h' parameter optimization, e.g., (0.1, 10.0).
            If None, defaults to DEFAULT_H_BOUNDS.
        kernel_fit_type : str, default='scale'
            The type of kernel fitting to perform. Passed to `kernel_fit`.
        objective_metric : str, default='spatial'
            The metric to use for optimizing 'h'. Must be 'spatial' or 'axis'.
        svm_kwargs : dict[str, object], optional
            Additional keyword arguments to pass to the underlying `sklearn.svm.SVC`.
        """

        if objective_metric not in ['spatial', 'axis']:
            raise ValueError(
                f"Invalid objective_metric '{objective_metric}'. "
                "Expected 'spatial' or 'axis'."
            )

        self.h_bounds = tuple(h_bounds) if h_bounds is not None else DEFAULT_H_BOUNDS
        self.kernel_fit_type = kernel_fit_type
        self.objective_metric = objective_metric
        self.svm_kwargs = svm_kwargs if svm_kwargs is not None else {}

        self.h_opt_: np.float64 | None = None
        self.cov_inv_: NDArray[np.float64] | None = None
        self.norm_factor_: np.float64 | None = None
        self.X_train_: NDArray[np.float64] | None = None
        self.svm_: SVC | None = None
        self.classes_: NDArray[np.int32] | None = None # unique_labels returns array of int/str

    def _objective_for_h_optimization(self, 
                                      h_candidate: np.float64, 
                                      X_checked: NDArray[np.float64], 
                                      y_checked: NDArray[np.int32]) -> np.float64 | float:
        """
        Objective function to be minimized for h optimization.
        Calculates -score from `objective_function`.

        Parameters:
        -----------
        h_candidate : np.float64
            The candidate value for the kernel parameter 'h'.
        X_checked : NDArray[np.float64]
            The checked training input samples.
        y_checked : NDArray[np.int32]
            The checked training target values.

        Returns:
        --------
        np.float64
            The negative score from the `objective_function`. Returns np.inf if score is NaN
            or if issues arise (e.g., class imbalance making Q0/Q1 problematic).
        """
        # These attributes should be set in fit() before this method is called.
        if self.norm_factor_ is None or self.cov_inv_ is None or self.classes_ is None:
            raise RuntimeError("Kernel fit parameters or classes not initialized before h optimization.")

        K_h = kernel(X_checked, X_checked, self.norm_factor_, self.cov_inv_, h_candidate)
        
        if K_h.shape[0] != len(y_checked):
            logger.warning(f"Mismatch in K_h rows ({K_h.shape[0]}) and y_checked length ({len(y_checked)}) for h={h_candidate}. Returning inf.")
            return np.inf

        # Ensure there are at least two classes for the Q0, Q1 logic
        if len(self.classes_) < 2:
            logger.warning(f"Less than 2 classes found during h optimization callback ({self.classes_}). Objective score is ill-defined. Returning inf.")
            return np.inf

        # Assuming binary classification
        # If multiclass, this part needs to be adapted based on `objective_function`'s expectation
        Q0 = np.sum(K_h[:, y_checked == self.classes_[0]], axis=1)
        Q1 = np.sum(K_h[:, y_checked == self.classes_[1]], axis=1)

        if self.objective_metric == 'axis':
            # Use axis spread distance objective function
            current_obj_score = vecspd_objfunc(Q0, Q1, y_checked)
        elif self.objective_metric == 'spatial':
            # Use two-dimensional spread distance objective function
            current_obj_score = twod_objfunc(Q0, Q1, y_checked)
        else:
            raise ValueError(
                f"Invalid objective_metric '{self.objective_metric}'. "
                "Expected 'spatial' or 'axis'."
            )

        if np.isnan(np.float64(current_obj_score)):
            return np.inf  # Optimizer should avoid NaN values.
        
        # We want to maximize objective_function, so minimize its negative
        return -np.float64(current_obj_score)

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int32]) -> Self:
        """
        Fit the SVM model according to the given training data.

        Parameters:
        -----------
        X : NDArray[np.float64]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : NDArray[np.int32]
            Target values (class labels) as integers.

        Returns:
        --------
        Self
            Fitted estimator.
        """
        X_checked, y_checked = check_X_y(X, y, dtype=[np.float64, np.int32])
        self.classes_ = unique_labels(y_checked)
        self.X_train_ = X_checked

        # 1. Kernel Fit
        # These will be stored in self.norm_factor_ and self.cov_inv_
        self.norm_factor_, self.cov_inv_ = kernel_fit(X_checked, type=self.kernel_fit_type)

        # 2. Optimize h (h_opt_)
        # Ensure there are at least two classes for Q0, Q1 logic in objective function
        if len(self.classes_) < 2:
            raise ValueError(
                f"The number of classes found is {len(self.classes_)}. "
                "This SVM's 'h' optimization requires at least two classes for the current "
                "objective_function logic (Q0, Q1 calculation). "
                "Please ensure your training data has at least two distinct classes."
            )
            
        # Arguments to pass to the objective function (h_candidate is passed by optimizer)
        optimization_args = (X_checked, y_checked)

        result = minimize_scalar(
            self._objective_for_h_optimization, # Function to minimize
            args=optimization_args,             # Extra arguments to our function
            bounds=self.h_bounds,               # Bounds for h
            method='bounded'                    # Use bounded optimization method
        )

        result: OptimizationResult = cast(OptimizationResult, result)

        if result.success:
            self.h_opt_ = np.float64(result.x)
        else:
            logger.warning(f"'h' optimization failed. Optimizer message: {result.message}")
            val_at_lower_bound = self._objective_for_h_optimization(self.h_bounds[0], *optimization_args)
            val_at_upper_bound = self._objective_for_h_optimization(self.h_bounds[1], *optimization_args)

            best_bound_h: np.float64 | None = None
            min_objective_at_bound = np.inf

            if not np.isinf(val_at_lower_bound):
                min_objective_at_bound = val_at_lower_bound
                best_bound_h = self.h_bounds[0]
            
            if not np.isinf(val_at_upper_bound) and val_at_upper_bound < min_objective_at_bound:
                min_objective_at_bound = val_at_upper_bound
                best_bound_h = self.h_bounds[1]

            if best_bound_h is not None:
                self.h_opt_ = best_bound_h
                logger.info(f"Defaulting to boundary value h={self.h_opt_} with objective value {-min_objective_at_bound:.4f}.")
            else:
                raise ValueError(
                    "Optimization of 'h' failed, and evaluating at boundaries also resulted "
                    "in invalid objective values (inf). Cannot determine optimal 'h'."
                )
        
        if self.h_opt_ is None: # Should ideally be caught by the logic above
             raise ValueError("Could not determine optimal 'h' after optimization attempts.")

        # 3. Compute final training kernel with h_opt_
        if self.norm_factor_ is None or self.cov_inv_ is None: # Should be set by kernel_fit
            raise RuntimeError("Kernel fit parameters (norm_factor_ or cov_inv_) are not set after kernel_fit.")

        K_train_optimal = kernel(X_checked, X_checked, self.norm_factor_, self.cov_inv_, self.h_opt_)

        # 4. Fit SVM
        self.svm_ = SVC(kernel='precomputed', **self.svm_kwargs)
        self.svm_.fit(K_train_optimal, y_checked)
        
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int32]:
        """
        Perform classification on samples in X.

        Parameters:
        -----------
        X : NDArray[np.float64]
            For kernel='precomputed', the expected shape of X is (n_samples_test, n_samples_train).
            However, our kernel function computes this matrix. So X should be feature vectors.

        Returns:
        --------
        NDArray[np.int32]
            Class labels for samples in X.
        """
        check_is_fitted(self, ['svm_', 'X_train_', 'norm_factor_', 'cov_inv_', 'h_opt_'])
        # Ensure X is a 2D array of floats
        X_checked = check_array(X, dtype=np.float64) 
        
        # These attributes are guaranteed to be non-None by check_is_fitted and fit() logic
        K_test = kernel(X_checked, 
                        cast(NDArray[np.float64], self.X_train_), 
                        cast(np.float64, self.norm_factor_),
                        cast(NDArray[np.float64], self.cov_inv_), 
                        cast(np.float64, self.h_opt_))
        
        return cast(SVC, self.svm_).predict(K_test)

    def score(self, X: NDArray[np.float64], y: NDArray[np.int32]) -> np.float64:
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        -----------
        X : NDArray[np.float64]
            Test samples.
        y : NDArray[np.int32]
            True labels for X.

        Returns:
        --------
        np.float64
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return np.float64(accuracy_score(y, y_pred)) # Ensure return is np.float64

    def get_params(self, deep: bool = True) -> dict[str, float | str | tuple[np.float64, np.float64]]:
        """
        Get parameters for this estimator. Includes constructor parameters and h_opt_ if fitted.
        """
        params = super().get_params(deep)
        # Optionally include fitted parameters if desired for inspection,
        # but note that standard get_params usually only returns constructor params.
        if hasattr(self, 'h_opt_') and self.h_opt_ is not None:
            params['h_opt_'] = self.h_opt_ 
        return params

    def set_params(self, **params: float | str | tuple[np.float64, np.float64]) -> Self:
        """
        Set the parameters of this estimator.
        """
        if 'h_opt_' in params:
            raise ValueError("Cannot set 'h_opt_' directly. It is determined during fit().")
        
        super().set_params(**params)
        return self
import os
import sys
import numpy as np

from numpy.typing import NDArray # ArrayLike can be removed if not used elsewhere
from typing import cast, Any

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize_scalar

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Kernel.kernel import kernel, kernel_fit 
from AxisSpread.vector_spread import objective_function

# Default bounds for h optimization
DEFAULT_H_BOUNDS: tuple[float, float] = (5e-1, 5e0)

class SVM(BaseEstimator, ClassifierMixin):
    """
    Custom SVM using a precomputed kernel where the kernel parameter 'h'
    is optimized using scipy.optimize.minimize_scalar based on a custom metric.
    """
    def __init__(self, 
                 h_bounds: tuple[float, float] | None = None, 
                 kernel_fit_type: str = 'scale', 
                 svm_kwargs: dict[str, Any] | None = None):
        """
        Initialize the SVM.

        Parameters:
        -----------
        h_bounds : tuple[float, float], optional
            The lower and upper bounds for the 'h' parameter optimization, e.g., (0.1, 10.0).
            If None, defaults to DEFAULT_H_BOUNDS.
        kernel_fit_type : str, default='scale'
            The type of kernel fitting to perform. Passed to `kernel_fit`.
        svm_kwargs : dict, optional
            Additional keyword arguments to pass to the underlying `sklearn.svm.SVC`.
        """
        self.h_bounds = tuple(h_bounds) if h_bounds is not None else DEFAULT_H_BOUNDS
        self.kernel_fit_type = kernel_fit_type
        self.svm_kwargs = svm_kwargs if svm_kwargs is not None else {}

        self.h_opt_: float | None = None
        self.cov_inv_: NDArray[np.float64] | None = None
        self.norm_factor_: float | None = None
        self.X_train_: NDArray[np.float64] | None = None
        self.svm_: SVC | None = None
        self.classes_: NDArray[np.int_] | None = None # unique_labels returns array of int/str

    def _objective_for_h_optimization(self, 
                                      h_candidate: float, 
                                      X_checked: NDArray[np.float64], 
                                      y_checked: NDArray[np.int_]) -> float:
        """
        Objective function to be minimized for h optimization.
        Calculates -score from `objective_function`.

        Parameters:
        -----------
        h_candidate : float
            The candidate value for the kernel parameter 'h'.
        X_checked : NDArray[np.float64]
            The checked training input samples.
        y_checked : NDArray[np.int_]
            The checked training target values.

        Returns:
        --------
        float
            The negative score from the `objective_function`. Returns np.inf if score is NaN
            or if issues arise (e.g., class imbalance making Q0/Q1 problematic).
        """
        # These attributes should be set in fit() before this method is called.
        if self.norm_factor_ is None or self.cov_inv_ is None or self.classes_ is None:
            raise RuntimeError("Kernel fit parameters or classes not initialized before h optimization.")

        K_h = kernel(X_checked, X_checked, self.norm_factor_, self.cov_inv_, h_candidate)
        
        if K_h.shape[0] != len(y_checked):
            print(f"Warning: Mismatch in K_h rows ({K_h.shape[0]}) and y_checked length ({len(y_checked)}) for h={h_candidate}. Returning inf.")
            return np.inf

        # Ensure there are at least two classes for the Q0, Q1 logic from original code
        if len(self.classes_) < 2:
            # This scenario should be caught earlier in fit(), but as a safeguard:
            print(f"Warning: Less than 2 classes found during h optimization callback ({self.classes_}). Objective score is ill-defined. Returning inf.")
            return np.inf

        # Assuming binary classification based on original Q0, Q1 logic
        # If multiclass, this part needs to be adapted based on `objective_function`'s expectation
        Q0 = np.sum(K_h[:, y_checked == self.classes_[0]], axis=1)
        Q1 = np.sum(K_h[:, y_checked == self.classes_[1]], axis=1)

        current_obj_score = objective_function(Q0, Q1, y_checked)

        if np.isnan(current_obj_score):
            return np.inf  # Optimizer should avoid NaN values.
        
        # We want to maximize objective_function, so minimize its negative
        return -current_obj_score

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int_]) -> "SVM":
        """
        Fit the SVM model according to the given training data.

        Parameters:
        -----------
        X : NDArray[np.float64]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : NDArray[np.int_]
            Target values (class labels) as integers.

        Returns:
        --------
        Self
            Fitted estimator.
        """
        X_checked, y_checked = check_X_y(X, y, dtype=[np.float64, np.int_])
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

        if result.success:
            self.h_opt_ = float(result.x)
        else:
            # If optimization failed, try evaluating at the boundaries as a fallback
            print(f"Warning: 'h' optimization failed. Optimizer message: {result.message}")
            val_at_lower_bound = self._objective_for_h_optimization(self.h_bounds[0], *optimization_args)
            val_at_upper_bound = self._objective_for_h_optimization(self.h_bounds[1], *optimization_args)

            best_bound_h: float | None = None
            min_objective_at_bound = np.inf

            if not np.isinf(val_at_lower_bound):
                min_objective_at_bound = val_at_lower_bound
                best_bound_h = self.h_bounds[0]
            
            if not np.isinf(val_at_upper_bound) and val_at_upper_bound < min_objective_at_bound:
                min_objective_at_bound = val_at_upper_bound
                best_bound_h = self.h_bounds[1]

            if best_bound_h is not None:
                self.h_opt_ = best_bound_h
                # Original objective score is -min_objective_at_bound
                print(f"Defaulting to boundary value h={self.h_opt_} with objective value {-min_objective_at_bound:.4f}.")
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

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Perform classification on samples in X.

        Parameters:
        -----------
        X : NDArray[np.float64]
            For kernel='precomputed', the expected shape of X is (n_samples_test, n_samples_train).
            However, our kernel function computes this matrix. So X should be feature vectors.

        Returns:
        --------
        NDArray[np.int_]
            Class labels for samples in X.
        """
        check_is_fitted(self, ['svm_', 'X_train_', 'norm_factor_', 'cov_inv_', 'h_opt_'])
        # Ensure X is a 2D array of floats
        X_checked = check_array(X, dtype=np.float64) 
        
        # These attributes are guaranteed to be non-None by check_is_fitted and fit() logic
        K_test = kernel(X_checked, 
                        cast(NDArray[np.float64], self.X_train_), 
                        cast(float, self.norm_factor_),
                        cast(NDArray[np.float64], self.cov_inv_), 
                        cast(float, self.h_opt_))
        
        return cast(SVC, self.svm_).predict(K_test)

    def score(self, X: NDArray[np.float64], y: NDArray[np.int_]) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        -----------
        X : NDArray[np.float64]
            Test samples.
        y : NDArray[np.int_]
            True labels for X.

        Returns:
        --------
        float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return float(accuracy_score(y, y_pred)) # Ensure return is float

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Get parameters for this estimator. Includes constructor parameters and h_opt_ if fitted.
        """
        params = super().get_params(deep)
        # Optionally include fitted parameters if desired for inspection,
        # but note that standard get_params usually only returns constructor params.
        if hasattr(self, 'h_opt_') and self.h_opt_ is not None:
            params['h_opt_'] = self.h_opt_ 
        return params

    def set_params(self, **params: Any) -> "SVM":
        """
        Set the parameters of this estimator.
        """
        # Handle h_opt_ specifically if it's part of params,
        # though typically fitted parameters are not set this way.
        if 'h_opt_' in params:
            self.h_opt_ = params.pop('h_opt_') # Remove it so super().set_params doesn't complain
        
        super().set_params(**params)
        return self
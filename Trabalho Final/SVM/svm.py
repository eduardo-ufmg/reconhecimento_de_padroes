import os
import sys

import numpy as np

from numpy.typing import ArrayLike, NDArray
from typing import cast
from typing_extensions import Self

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Kernel.kernel import kernel, kernel_fit
from AxisSpread.vector_spread import objective_function

DEFAULT_HS_RANGE = np.linspace(5e-1, 5e0, 100)

class SVM(BaseEstimator, ClassifierMixin):
    """
    Custom SVM using a precomputed kernel where the kernel parameter 'h'
    is optimized based on a custom metric.
    Assumes input data (X) is already preprocessed.
    """
    def __init__(self, hs_range: ArrayLike | None = None, kernel_fit_type: str = 'scale', svm_kwargs: dict | None = None):
        self.hs_range = cast(list, hs_range if hs_range is not None else DEFAULT_HS_RANGE)
        self.kernel_fit_type = kernel_fit_type
        self.svm_kwargs = svm_kwargs if svm_kwargs is not None else {}

        self.h_opt_ = None
        self.cov_inv_ = None
        self.norm_factor_ = None
        self.X_train_ = None
        self.svm_ = None
        self.classes_ = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int32]) -> Self:

        X_checked, y_checked = check_X_y(X, y) # check_X_y also handles conversion if X is list etc.
        self.classes_ = unique_labels(y_checked)

        # Store the training data for kernel computation at predict time
        self.X_train_ = X_checked

        # 1. Kernel Fit
        self.norm_factor_, self.cov_inv_ = kernel_fit(X_checked, type=self.kernel_fit_type)

        # 2. Optimize h (h_opt)
        best_h = None
        max_obj_score = -np.inf

        for h_candidate in self.hs_range:
            K_h = kernel(X_checked, X_checked, self.norm_factor_, self.cov_inv_, h_candidate)
            
            if K_h.shape[0] != len(y_checked):
                 raise ValueError(f"Mismatch in K_h rows ({K_h.shape[0]}) and y_checked length ({len(y_checked)}) for h={h_candidate}")

            # Assuming binary classification, adjust if multiclass spread logic is different
            Q0 = np.sum(K_h[:, y_checked == self.classes_[0]], axis=1)
            Q1 = np.sum(K_h[:, y_checked == self.classes_[1]], axis=1)

            current_obj_score = objective_function(Q0, Q1, y_checked)

            if not np.isnan(current_obj_score) and current_obj_score > max_obj_score:
                max_obj_score = current_obj_score
                best_h = h_candidate
        
        if best_h is None and len(self.hs_range) > 0:
            best_h = self.hs_range[0]
            print(f"Warning: Could not find an optimal h, defaulting to first in hs_range: {best_h}")
        elif best_h is None:
             raise ValueError("Could not determine optimal h, hs_range might be empty or all scores were NaN.")
        self.h_opt_ = best_h

        # 3. Compute final training kernel with h_opt
        K_train_optimal = kernel(X_checked, X_checked, self.norm_factor_, self.cov_inv_, self.h_opt_)

        # 4. Fit SVM
        self.svm_ = SVC(kernel='precomputed', **self.svm_kwargs)
        self.svm_.fit(K_train_optimal, y_checked)
        
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int32]:
        check_is_fitted(self, ['svm_', 'X_train_', 'norm_factor_', 'cov_inv_', 'h_opt_'])
        X_checked = check_array(X) # Ensure X is a numpy array
        
        # Compute kernel between new X_checked and original training X_train_
        K_test = kernel(X_checked, cast(NDArray[np.float64], self.X_train_), cast(float, self.norm_factor_),
                        cast(NDArray[np.float64], self.cov_inv_), cast(float, self.h_opt_))
        
        return cast(SVC, self.svm_).predict(K_test)

    def score(self, X: NDArray[np.float64], y: NDArray[np.int32]) -> float:
        y_pred = self.predict(X)
        return cast(float, accuracy_score(y, y_pred))

    def get_params(self, deep=True):
        params = super().get_params(deep)
        if hasattr(self, 'h_opt_') and self.h_opt_ is not None:
            params['h_opt_'] = self.h_opt_ 
        return params

    def set_params(self, **params):
        super().set_params(**params)
        return self
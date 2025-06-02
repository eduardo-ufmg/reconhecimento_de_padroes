import numpy as np
from numpy.typing import NDArray

# Adjusted threshold for float64 determinant checks.
# np.finfo(np.float64).eps is approx 1.19e-7.
# This threshold checks if determinant is practically zero for float64.
COV_DET_THRESHOLD_F32 = np.float64(1e-7) 

def kernel(X: NDArray[np.float64], Y: NDArray[np.float64],
           norm_factor: np.float64, cov_inv: NDArray[np.float64],
           h: np.float64) -> NDArray[np.float64]:
    """
    Computes the kernel matrix (highly optimized for speed and memory with float64).
    mvpdf logic is inlined.
    """
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return np.empty((X.shape[0], Y.shape[0]), dtype=np.float64)

    # Input validation (can be removed if inputs are guaranteed correct by caller for extreme speed)
    if X.shape[1] != cov_inv.shape[0] or Y.shape[1] != cov_inv.shape[0]:
        raise ValueError("Feature dimensions of X or Y do not match cov_inv.")
    if cov_inv.shape[0] != cov_inv.shape[1]:
        raise ValueError("cov_inv is not a square matrix.")

    n_features: int = X.shape[1]

    if h == 0: # h is np.float64, so direct comparison is okay
        raise ValueError("Bandwidth parameter h cannot be zero.")
    
    h_squared = np.float64(h ** 2) # Ensure h_squared is float64
    if abs(h_squared) < np.finfo(np.float64).tiny:
        # This situation can lead to division by zero or extremely large scaled_cov_inv.
        # Depending on the desired behavior for such extreme h, this might need
        # more specific handling or indicate an issue with the h value.
        # Proceeding might lead to Inf/NaN if not careful.
        # For now, allow to see if normalization handles it, but it's risky.
        pass

    scaled_cov_inv: NDArray[np.float64] = cov_inv / h_squared

    h_pow_n_features = np.float64(h) ** n_features
    
    scaled_norm_factor: np.float64
    if norm_factor == np.float64(0.0):
        scaled_norm_factor = np.float64(0.0)
    elif abs(h_pow_n_features) < np.finfo(np.float64).tiny:
        scaled_norm_factor = np.float64(1.0) # Avoid Inf from norm_factor / tiny_h_pow_n
    else:
        scaled_norm_factor = norm_factor / h_pow_n_features

    # --- Inlined mvpdf logic ---    
    term_X_C_X = np.einsum('ni,ij,nj->n', X, scaled_cov_inv, X, optimize=True)
    term_Y_C_Y = np.einsum('ni,ij,nj->n', Y, scaled_cov_inv, Y, optimize=True)
    term_X_C_Y = np.einsum('ni,ij,mj->nm', X, scaled_cov_inv, Y, optimize=True)

    quadratic_form_values = (
        term_X_C_X[:, np.newaxis] - 2 * term_X_C_Y + term_Y_C_Y[np.newaxis, :]
    )
    
    exp_term = np.exp(np.float64(-0.5) * quadratic_form_values) # Ensure -0.5 is also float64
    pdf = scaled_norm_factor * exp_term
    # --- End of inlined mvpdf logic ---

    max_pdf_val = np.max(pdf)

    if max_pdf_val <= np.finfo(pdf.dtype).tiny:
        pdf_normalized = np.zeros_like(pdf, dtype=np.float64)
    else:
        pdf_normalized = pdf / max_pdf_val

    return pdf_normalized.astype(np.float64) # Final explicit cast


def kernel_fit(X: NDArray[np.float64], type: str='cov') -> tuple[np.float64, NDArray[np.float64]]:
    """
    Fits kernel parameters (optimized for speed and memory with float64).
    All ops performed in float64, accepting higher numerical stability risk.
    """
    if type not in ['cov', 'scale']:
        raise ValueError("Invalid type. Supported types are 'cov' and 'scale'.")
    
    n_samples, n_features = X.shape

    if n_features == 0:
        raise ValueError("Cannot fit kernel with 0 features.")

    norm_factor_val: np.float64
    cov_inv_val: NDArray[np.float64]
    
    if type == 'cov':
        if n_samples <= 1:
            raise ValueError(f"Cannot compute covariance for type='cov' with {n_samples} samples.")

        # Perform all covariance calculations in float64
        # np.cov might use float64 internally for accumulation, so cast result.
        cov_matrix_f32 = np.cov(X, rowvar=False).astype(np.float64)

        if n_features == 1 and cov_matrix_f32.ndim == 0:
            cov_matrix_f32 = np.array([[cov_matrix_f32.item()]], dtype=np.float64)
        elif cov_matrix_f32.shape != (n_features, n_features):
             raise ValueError(
                 f"Covariance matrix shape {cov_matrix_f32.shape} is not ({n_features}, {n_features})."
            )
        
        # Determinant calculation directly on float64 matrix. Result might be float64 scalar by numpy.
        cov_det_f32 = np.float64(np.linalg.det(cov_matrix_f32))

        if abs(cov_det_f32) <= COV_DET_THRESHOLD_F32 or cov_det_f32 < 0: # Check for negative also, det of PSD should be non-negative
            raise ValueError(
                f"Covariance matrix (float64) is singular or ill-conditioned (determinant: {cov_det_f32:.2e})."
            )

        sqrt_cov_det_f32 = np.sqrt(cov_det_f32) # sqrt on float64
        # Ensure intermediate terms in denominator are float64 if possible, or result casted
        pi_f32 = np.float64(np.pi)
        norm_factor_val = np.float64(1.0) / ( ( (np.float64(2.0) * pi_f32) ** (np.float64(n_features) / np.float64(2.0)) ) * sqrt_cov_det_f32 )
        
        # Inverse calculation directly on float64 matrix
        cov_inv_val = np.linalg.inv(cov_matrix_f32).astype(np.float64) # Ensure result is float64

    elif type == 'scale':
        cov_inv_val = np.eye(n_features, dtype=np.float64)
        pi_f32 = np.float64(np.pi)
        norm_factor_val = np.float64(1.0) / ( (np.float64(2.0) * pi_f32) ** (np.float64(n_features) / np.float64(2.0)) )
    
    return norm_factor_val, cov_inv_val
## Analysis of Kernel Computation and Fitting Procedures

This document details the mathematical operations underlying two functions: one for fitting initial parameters based on input data (`kernel_fit`), and another for computing a kernel matrix using these parameters along with a bandwidth hyperparameter (`kernel`). These functions are designed for use with `float64` precision.

---
## 1. Initial Kernel Parameter Estimation Procedure (`kernel_fit`)

This procedure estimates a normalization factor and an inverse covariance-like matrix from an input data matrix. It supports two modes of operation: 'cov' (covariance-based) and 'scale' (scale-based, using an identity matrix).

### Conceptual Definition
The estimation of initial kernel parameters involves calculating a scaling coefficient (normalization factor) and a matrix (inverse covariance or identity) that will be used in subsequent kernel computations. The 'cov' method derives these from the data's empirical covariance, while the 'scale' method uses a default identity structure.

### Mathematical Formulation

**Inputs:**
* Data matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$, where $N$ is the number of samples and $D$ is the number of features.
* Estimation type (a string): 'cov' or 'scale'.

**Outputs:**
* Normalization factor $\eta \in \mathbb{R}$.
* Inverse matrix $\mathbf{\Psi}^{-1} \in \mathbb{R}^{D \times D}$ (this is $\mathbf{\Sigma}^{-1}$ if type is 'cov', or $\mathbf{I}$ if type is 'scale').

**Procedure A: Covariance-based Estimation ('cov')**
This method is applicable if $N > 1$ and $D > 0$.
1.  **Empirical Covariance Matrix Calculation ($\mathbf{\Sigma}$):**
    The $D \times D$ empirical covariance matrix of the data $\mathbf{X}$ is computed. Assuming $\bm{x}_i$ are row vectors representing samples:
    $$\hat{\bm{\mu}} = \frac{1}{N} \sum_{i=1}^{N} \bm{x}_i$$
    $$\mathbf{\Sigma} = \frac{1}{N-1} \sum_{i=1}^{N} (\bm{x}_i - \hat{\bm{\mu}})^T (\bm{x}_i - \hat{\bm{\mu}})$$
    The implementation uses `np.cov(X, rowvar=False)`.

2.  **Covariance Matrix Validation:**
    * The determinant of $\mathbf{\Sigma}$, denoted $\det(\mathbf{\Sigma})$, is calculated.
    * If $|\det(\mathbf{\Sigma})| \le \epsilon_{det}$ (where $\epsilon_{det}$ is a small threshold, e.g., $10^{-7}$) or $\det(\mathbf{\Sigma}) < 0$, the matrix is considered singular or ill-conditioned, and an error is raised.

3.  **Normalization Factor Calculation ($\eta$):**
    The normalization factor is derived from the formula for a multivariate Gaussian probability density function's constant:
    $$\eta = \frac{1}{(2\pi)^{D/2} \sqrt{\det(\mathbf{\Sigma})}}$$
   

4.  **Inverse Covariance Matrix Calculation ($\mathbf{\Psi}^{-1} = \mathbf{\Sigma}^{-1}$):**
    The inverse of the covariance matrix $\mathbf{\Sigma}$ is computed:
    $$\mathbf{\Psi}^{-1} = \mathbf{\Sigma}^{-1}$$
   

**Procedure B: Scale-based Estimation ('scale')**
This method is applicable if $D > 0$.
1.  **Inverse Matrix Setup ($\mathbf{\Psi}^{-1} = \mathbf{I}$):**
    The inverse matrix is set to the $D \times D$ identity matrix $\mathbf{I}$.
    $$\mathbf{\Psi}^{-1} = \mathbf{I}$$

2.  **Normalization Factor Calculation ($\eta$):**
    The normalization factor is calculated as if for a multivariate Gaussian distribution with an identity covariance matrix (where $\det(\mathbf{I}) = 1$):
    $$\eta = \frac{1}{(2\pi)^{D/2}}$$
   

**Error Conditions:**
* Invalid estimation type specified.
* Zero features ($D=0$) in the input data.
* For 'cov' type:
    * Insufficient samples ($N \le 1$) to compute covariance.
    * Covariance matrix is singular, ill-conditioned, or its determinant is negative.

### Algorithmic Process: Estimation of Initial Kernel Parameters

```pseudocode
ALGORITHM EstimateInitialKernelParameters(Data_Matrix X, Estimation_Type type_str):
    INPUT:
        Data_Matrix X (N samples, D features).
        Estimation_Type type_str ('cov' or 'scale').
    OUTPUT:
        Tuple (NormalizationFactor eta, InverseMatrix Psi_inv).

    1. VALIDATE type_str.
    2. LET N, D be dimensions of X.
    3. IF D = 0 THEN RAISE Error.

    4. IF type_str = 'cov' THEN
    5.     IF N <= 1 THEN RAISE Error.
    6.     CovarianceMatrix Sigma = ComputeCovariance(X).
    7.     Determinant_Sigma = Determinant(Sigma).
    8.     IF AbsoluteValue(Determinant_Sigma) <= Threshold OR Determinant_Sigma < 0 THEN
    9.         RAISE Error (Singular/Ill-conditioned Covariance).
    10.    END IF
    11.    NormalizationFactor eta = 1.0 / ( (2 * PI)^(D / 2.0) * SquareRoot(Determinant_Sigma) ).
    12.    InverseMatrix Psi_inv = Inverse(Sigma).
    13. ELSE IF type_str = 'scale' THEN
    14.    InverseMatrix Psi_inv = IdentityMatrix(D).
    15.    NormalizationFactor eta = 1.0 / ( (2 * PI)^(D / 2.0) ).
    16. END IF

    17. RETURN (eta, Psi_inv).
```

---
## 2. Parameterized Kernel Function (`kernel`)

This function computes a kernel matrix between two sets of input samples, $X$ and $Y$. The computation involves a scaled inverse covariance-like matrix and a bandwidth parameter $h$, resembling a multivariate Gaussian probability density function (PDF) evaluation, followed by normalization.

### Conceptual Definition
The kernel computation evaluates a measure of similarity between pairs of samples, one from set $X$ and one from set $Y$. This similarity is based on a quadratic form (related to Mahalanobis distance) scaled by a bandwidth parameter $h$, transformed by an exponential function, and then scaled by a normalization factor also adjusted by $h$. The resulting matrix of these values is then normalized by its maximum value.

### Mathematical Formulation

**Inputs:**
* Data matrix $\mathbf{X} \in \mathbb{R}^{N_X \times D}$ ($N_X$ samples, $D$ features).
* Data matrix $\mathbf{Y} \in \mathbb{R}^{N_Y \times D}$ ($N_Y$ samples, $D$ features).
* Initial normalization factor $\eta_0 \in \mathbb{R}$ (obtained from `kernel_fit`).
* Initial inverse matrix $\mathbf{\Psi}^{-1}_0 \in \mathbb{R}^{D \times D}$ (obtained from `kernel_fit`).
* Bandwidth hyperparameter $h \in \mathbb{R}$, $h \neq 0$.

**Output:**
* Normalized kernel matrix $\mathbf{K} \in \mathbb{R}^{N_X \times N_Y}$.

**Calculation Steps:**
1.  **Input Validation & Pre-computation:**
    * Ensure feature dimensions of $\mathbf{X}$, $\mathbf{Y}$ match $\mathbf{\Psi}^{-1}_0$.
    * Ensure $\mathbf{\Psi}^{-1}_0$ is square.
    * If $h=0$, an error is raised.
    * Calculate $h^2$.
    * Scaled Inverse Matrix ($\mathbf{\Psi}^{-1}_h$):
        $$\mathbf{\Psi}^{-1}_h = \frac{\mathbf{\Psi}^{-1}_0}{h^2}$$
       
    * Scaled Normalization Factor ($\eta_h$):
        Let $h^D = h \text{ raised to the power of } D$.
        If $\eta_0 = 0$, then $\eta_h = 0$.
        If $|h^D|$ is very small (close to machine epsilon for `float64`), $\eta_h = 1.0$ (to avoid division by zero/overflow).
        Otherwise:
        $$\eta_h = \frac{\eta_0}{h^D}$$
       

2.  **Quadratic Form Calculation:**
    This step computes a value related to the squared Mahalanobis distance $d^2_{\mathbf{\Psi}^{-1}_h}(\bm{x}_i, \bm{y}_j) = (\bm{x}_i - \bm{y}_j)^T \mathbf{\Psi}^{-1}_h (\bm{x}_i - \bm{y}_j)$ for each pair of samples $(\bm{x}_i \in \mathbf{X}, \bm{y}_j \in \mathbf{Y})$. The implementation uses an optimized expansion:
    $d^2_{\mathbf{\Psi}^{-1}_h}(\bm{x}_i, \bm{y}_j) = \bm{x}_i^T \mathbf{\Psi}^{-1}_h \bm{x}_i - 2 \bm{x}_i^T \mathbf{\Psi}^{-1}_h \bm{y}_j + \bm{y}_j^T \mathbf{\Psi}^{-1}_h \bm{y}_j$.
    Let:
    * $\text{diag}(\mathbf{X} \mathbf{\Psi}^{-1}_h \mathbf{X}^T)_i = \bm{x}_i^T \mathbf{\Psi}^{-1}_h \bm{x}_i$
    * $\text{diag}(\mathbf{Y} \mathbf{\Psi}^{-1}_h \mathbf{Y}^T)_j = \bm{y}_j^T \mathbf{\Psi}^{-1}_h \bm{y}_j$
    * $(\mathbf{X} \mathbf{\Psi}^{-1}_h \mathbf{Y}^T)_{ij} = \bm{x}_i^T \mathbf{\Psi}^{-1}_h \bm{y}_j$
    The matrix of quadratic form values $\mathbf{Q} \in \mathbb{R}^{N_X \times N_Y}$ has elements:
    $$Q_{ij} = (\text{diag}(\mathbf{X} \mathbf{\Psi}^{-1}_h \mathbf{X}^T))_i - 2 (\mathbf{X} \mathbf{\Psi}^{-1}_h \mathbf{Y}^T)_{ij} + (\text{diag}(\mathbf{Y} \mathbf{\Psi}^{-1}_h \mathbf{Y}^T))_j$$
   

3.  **Unnormalized Kernel-like Values (PDF values):**
    A matrix $\mathbf{P} \in \mathbb{R}^{N_X \times N_Y}$ is computed:
    $$P_{ij} = \eta_h \exp\left(-\frac{1}{2} Q_{ij}\right)$$
   

4.  **Final Normalization:**
    * Let $P_{max} = \max_{i,j} (P_{ij})$.
    * If $P_{max}$ is very small (close to machine epsilon for the data type of $\mathbf{P}$), the final kernel matrix $\mathbf{K}$ is a zero matrix.
    * Otherwise, the final kernel matrix $\mathbf{K}$ is:
        $$K_{ij} = \frac{P_{ij}}{P_{max}}$$
       

**Error Conditions:**
* Empty input matrices $\mathbf{X}$ or $\mathbf{Y}$ result in an empty kernel matrix.
* Mismatch in feature dimensions between input matrices and the initial inverse matrix.
* Initial inverse matrix is not square.
* Bandwidth parameter $h=0$.

### Algorithmic Process: Computation of Parameterized Kernel

```pseudocode
ALGORITHM ComputeParameterizedKernel(Data_X, Data_Y, InitialNormFactor eta0, InitialInvMatrix Psi0_inv, Bandwidth_h):
    INPUT:
        Data_X (NX samples, D features).
        Data_Y (NY samples, D features).
        InitialNormFactor eta0.
        InitialInvMatrix Psi0_inv (D x D).
        Bandwidth_h (scalar, non-zero).
    OUTPUT:
        Kernel_Matrix K (NX x NY).

    1. IF Data_X or Data_Y is empty THEN RETURN Empty Matrix(NX, NY).
    2. VALIDATE feature dimensions of Data_X, Data_Y against Psi0_inv.
    3. VALIDATE Psi0_inv is square.
    4. IF Bandwidth_h = 0 THEN RAISE Error.

    5. h_squared = Bandwidth_h * Bandwidth_h.
    6. Scaled_Psi_inv = Psi0_inv / h_squared.
    7. h_pow_D = Bandwidth_h ^ D.

    8. IF eta0 = 0.0 THEN
    9.     Scaled_eta = 0.0.
    10. ELSE IF AbsoluteValue(h_pow_D) < TinyValue THEN
    11.    Scaled_eta = 1.0.
    12. ELSE
    13.    Scaled_eta = eta0 / h_pow_D.
    14. END IF

    15. Term_X_Psi_X = DiagonalElements(Data_X * Scaled_Psi_inv * Transpose(Data_X)). // Vector of length NX
    16. Term_Y_Psi_Y = DiagonalElements(Data_Y * Scaled_Psi_inv * Transpose(Data_Y)). // Vector of length NY
    17. Term_X_Psi_Y = Data_X * Scaled_Psi_inv * Transpose(Data_Y). // Matrix of size NX x NY

    18. INITIALIZE QuadraticFormMatrix Q (NX x NY).
    19. FOR i FROM 0 TO NX-1:
    20.     FOR j FROM 0 TO NY-1:
    21.         Q[i,j] = Term_X_Psi_X[i] - 2 * Term_X_Psi_Y[i,j] + Term_Y_Psi_Y[j].
    22.     END FOR
    23. END FOR
        // Alternative using broadcasting: Q = Term_X_Psi_X[:,newaxis] - 2*Term_X_Psi_Y + Term_Y_Psi_Y[newaxis,:]

    24. Unnormalized_Kernel_P = Scaled_eta * Exponential(-0.5 * Q).

    25. Max_P_Value = MaximumElement(Unnormalized_Kernel_P).
    26. IF Max_P_Value <= TinyValue THEN
    27.    Kernel_Matrix K = ZeroMatrix(NX, NY).
    28. ELSE
    29.    Kernel_Matrix K = Unnormalized_Kernel_P / Max_P_Value.
    30. END IF

    31. RETURN K.
```

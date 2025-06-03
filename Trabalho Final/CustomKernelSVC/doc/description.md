## Custom Kernel Support Vector Classifier (`CustomKernelSVC`)

The `CustomKernelSVC` is a classification algorithm that extends the Support Vector Machine (SVM) framework by incorporating a custom kernel. A key feature is the optimization of a kernel hyperparameter, denoted as $h$, to maximize a configurable objective metric related to data separability or spread.

### 1. Initialization

**Conceptual Definition:**
The initialization phase configures the classifier with parameters controlling the optimization process and the underlying SVM.

**Parameters:**
* **Hyperparameter Bounds ($h_{bounds}$):** A tuple $(h_{min}, h_{max})$ defining the search interval for the optimal kernel hyperparameter $h_{opt}$. Default is $(10^{-1}, 10^1)$.
* **Initial Kernel Parameter Estimation Type (`kernel_fit_type`):** A string identifier specifying the method for an external procedure (`kernel_fit`) to estimate initial kernel parameters.
* **Optimization Objective Metric Type (`objective_metric`):** A string ('spatial' or 'axis') determining which custom objective function is used to evaluate and optimize $h$.
    * 'spatial': Corresponds to a previously analyzed "Two-Dimensional Spread Distance Objective Function" ($S_{spatial}$).
    * 'axis': Corresponds to a previously analyzed "Vector Spread Objective Function" ($S_{axis}$).
* **Underlying SVM Arguments (`svm_kwargs`):** A dictionary of arguments passed to the standard `sklearn.svm.SVC` instance that uses the precomputed custom kernel.

Internal state variables for storing the optimal $h$ ($h_{opt}$), initial kernel parameters ($\eta, \Sigma^{-1}$), training data ($X_{train}$), class labels ($\mathcal{C}$), and the fitted SVM instance are also prepared.

### 2. Hyperparameter Optimization Objective Criterion (`_objective_for_h_optimization`)

**Conceptual Definition:**
This criterion quantifies the quality of a candidate kernel hyperparameter $h_{candidate}$ by computing a score based on the chosen `objective_metric`. This function is designed to be minimized by an optimization routine (hence it typically returns the negative of the desired score).

**Mathematical Formulation:**

**Inputs:**
* $h_{candidate}$: A candidate value for the kernel hyperparameter $h$.
* $X \in \mathbb{R}^{N \times D}$: The training data matrix, with $N$ samples and $D$ features.
* $y \in \mathcal{Y}^N$: The vector of $N$ training class labels. Let $\mathcal{C} = \{C_0, C_1, \dots, C_{K-1}\}$ be the set of $K$ unique class labels. The implementation primarily focuses on $K \ge 2$, and for constructing $Q$-vectors, it specifically uses $C_0$ and $C_1$ (the first two unique classes).

**Pre-requisites (from the `fit` method):**
* $\eta \in \mathbb{R}$ (normalization factor) and $\Sigma^{-1}$ (covariance-like matrix): Initial parameters derived from an external "Initial Kernel Parameter Estimation Procedure" (`kernel_fit`) applied to $X$.
* $\mathcal{C}$: The set of unique class labels identified from $y$.

**Calculation Steps:**
1.  **Parameterized Kernel Matrix Computation:** Given $h_{candidate}$, $\eta$, and $\Sigma^{-1}$, compute the $N \times N$ kernel matrix $K_h$ where $(K_h)_{uv} = \text{kernel}(x_u, x_v; \eta, \Sigma^{-1}, h_{candidate})$. The function `kernel` is an external procedure.

2.  **Data Projection Vector Computation:** For each training sample $x_i$ ( $i=1, \dots, N$), construct two projection values based on its kernel similarities to samples in class $C_0 = \mathcal{C}[0]$ and $C_1 = \mathcal{C}[1]$:
    * $(Q_{C_0}(h))_{i} = \sum_{j \text{ s.t. } y_j = C_0} (K_h)_{i,j}$
    * $(Q_{C_1}(h))_{i} = \sum_{j \text{ s.t. } y_j = C_1} (K_h)_{i,j}$
    This yields two vectors $Q_{C_0}(h), Q_{C_1}(h) \in \mathbb{R}^N$. Each element $(Q_{C_k}(h))_i$ represents the aggregated kernel similarity of sample $x_i$ to all training samples belonging to class $C_k$.

3.  **Objective Score Calculation ($S(h)$):** Depending on the `objective_metric` setting:
    * If 'spatial': $S(h) = S_{spatial}(Q_{C_0}(h), Q_{C_1}(h), y)$. $S_{spatial}$ is the previously analyzed "Two-Dimensional Spread Distance Objective Function".
    * If 'axis': $S(h) = S_{axis}(Q_{C_0}(h), Q_{C_1}(h), y)$. $S_{axis}$ is the previously analyzed "Vector Spread Objective Function".

4.  **Return Value for Minimization:** The function returns $-S(h)$. If $S(h)$ is undefined (e.g., NaN), or if $K < 2$, $\infty$ is returned to guide the optimizer away from such $h$ values.

**Output:**
A scalar value representing the negative of the custom objective score for the given $h_{candidate}$.

### 3. Model Training Procedure (`fit`)

**Conceptual Definition:**
The model training procedure fits the custom SVM to the training data. This involves initial kernel parameter estimation, optimization of the hyperparameter $h$, computation of the final kernel matrix using $h_{opt}$, and training the underlying SVM with this precomputed kernel.

**Mathematical Formulation & Algorithmic Steps:**

**Inputs:**
* $X \in \mathbb{R}^{N \times D}$: Training data.
* $y \in \mathcal{Y}^N$: Training labels.

**Procedure:**
1.  **Input Validation and Preparation:**
    * Validate $X$ and $y$.
    * Store $X$ as $X_{train}$.
    * Determine and store the set of unique class labels $\mathcal{C}$ from $y$. If fewer than two classes are present, the process may be aborted as the current $Q$-vector logic and objective functions typically require at least two classes.

2.  **Initial Kernel Parameter Estimation:**
    * The external "Initial Kernel Parameter Estimation Procedure" (`kernel_fit`) is invoked:
        $(\eta, \Sigma^{-1}) = \text{kernel\_fit}(X_{train}, \text{type}=\text{kernel\_fit\_type})$.
    * Store $\eta$ and $\Sigma^{-1}$.

3.  **Kernel Hyperparameter $h$ Optimization:**
    * The optimal hyperparameter $h_{opt}$ is found by maximizing the selected score $S(h)$ (or minimizing $-S(h)$):
        $$h_{opt} = \arg\max_{h \in [h_{min}, h_{max}]} S(h) = \arg\min_{h \in [h_{min}, h_{max}]} \left( -\text{Criterion}(h, X_{train}, y, \eta, \Sigma^{-1}) \right)$$
        where "Criterion" refers to the "Hyperparameter Optimization Objective Criterion" detailed in Section 2.
    * This optimization is performed using a scalar minimization routine (e.g., `scipy.optimize.minimize_scalar` with a bounded method).
    * If optimization fails, $h_{opt}$ may be set to one of the boundary values ($h_{min}$ or $h_{max}$) that yields a better (less negative or non-infinite) objective criterion value. If no valid $h$ is found, an error is raised.
    * Store $h_{opt}$.

4.  **Final Training Kernel Matrix Computation:**
    * Compute the $N \times N$ optimal training kernel matrix $K_{train\_opt}$ using the determined $h_{opt}$:
        $$K_{train\_opt} = \text{kernel}(X_{train}, X_{train}, \eta, \Sigma^{-1}, h_{opt})$$

5.  **Underlying SVM Training:**
    * An `sklearn.svm.SVC` instance is initialized with `kernel='precomputed'` and any user-provided `svm_kwargs`.
    * The SVM is trained using the optimal kernel matrix and training labels:
        $\text{SVC.fit}(K_{train\_opt}, y)$.
    * Store the fitted SVM instance.

**Output:**
The fitted `CustomKernelSVC` instance.

### 4. Prediction Procedure (`predict`)

**Conceptual Definition:**
The prediction procedure assigns class labels to new, unseen samples using the trained custom kernel SVM.

**Mathematical Formulation & Algorithmic Steps:**

**Inputs:**
* $X_{new} \in \mathbb{R}^{M \times D}$: New samples for prediction, with $M$ samples.

**Pre-requisites (from a fitted model):**
* The fitted underlying SVM.
* Stored $X_{train}, \eta, \Sigma^{-1}, h_{opt}$.

**Procedure:**
1.  **Input Validation:** Check $X_{new}$.
2.  **Test Kernel Matrix Computation:** Compute the $M \times N$ kernel matrix $K_{test}$ between the new samples $X_{new}$ and the original training samples $X_{train}$, using the stored optimal parameters:
    $$K_{test} = \text{kernel}(X_{new}, X_{train}, \eta, \Sigma^{-1}, h_{opt})$$
    Each entry $(K_{test})_{ij}$ represents the kernel similarity between the $i$-th new sample and the $j$-th training sample.
3.  **Prediction by Underlying SVM:** Use the fitted SVM to predict labels for $X_{new}$ based on $K_{test}$:
    $$\hat{y}_{new} = \text{SVC.predict}(K_{test})$$

**Output:**
$\hat{y}_{new} \in \mathcal{Y}^M$: Predicted class labels for the new samples.

### Pseudocode Summaries

**Pseudocode for Hyperparameter Optimization Objective Criterion (`_objective_for_h_optimization_logic`)**
```pseudocode
FUNCTION CalculateObjectiveCriterion(h_candidate, X_data, y_labels, eta_param, sigma_inv_param, unique_classes, metric_type):
    INPUT:
        h_candidate: Current value of h.
        X_data: Training features.
        y_labels: Training labels.
        eta_param, sigma_inv_param: Initial kernel parameters.
        unique_classes: List of unique class labels [C_0, C_1, ...].
        metric_type: 'spatial' or 'axis'.
    OUTPUT:
        Negative score, or Infinity if score is invalid.

    1. IF number of unique_classes < 2 THEN RETURN Infinity.
    2. K_h = ExternalKernelFunction(X_data, X_data, eta_param, sigma_inv_param, h_candidate).
    3. Q_C0_vector = Vector of sums of K_h rows, for columns where y_labels match unique_classes[0].
       // For each row i in K_h: Q_C0_vector[i] = sum(K_h[i,j] for all j where y_labels[j] == unique_classes[0])
    4. Q_C1_vector = Vector of sums of K_h rows, for columns where y_labels match unique_classes[1].
       // For each row i in K_h: Q_C1_vector[i] = sum(K_h[i,j] for all j where y_labels[j] == unique_classes[1])

    5. IF metric_type is 'axis' THEN
    6.     Score = ExternalAxisSpreadObjective(Q_C0_vector, Q_C1_vector, y_labels).
    7. ELSE IF metric_type is 'spatial' THEN
    8.     Score = ExternalSpatialSpreadObjective(Q_C0_vector, Q_C1_vector, y_labels).
    9. ELSE
    10.    RAISE Error (Invalid metric_type).
    11. END IF

    12. IF Score is NaN THEN RETURN Infinity.
    13. RETURN -Score.
```

**Pseudocode for Model Training Procedure (`fit_logic`)**
```pseudocode
FUNCTION TrainCustomSVC(X_train_data, y_train_labels, h_bounds_config, kernel_fit_config, objective_metric_config, svm_args_config):
    INPUT:
        X_train_data, y_train_labels: Training data.
        h_bounds_config: Bounds for h optimization.
        kernel_fit_config: Type for initial kernel fitting.
        objective_metric_config: Metric for h optimization.
        svm_args_config: Arguments for underlying SVC.
    OUTPUT:
        Fitted CustomKernelSVC model.

    1. Validate X_train_data, y_train_labels.
    2. UniqueClasses = GetUniqueLabels(y_train_labels).
    3. IF number of UniqueClasses < 2 THEN RAISE Error.
    4. Stored_X_train = X_train_data.

    5. (Eta_val, SigmaInv_val) = ExternalKernelFitProcedure(Stored_X_train, type=kernel_fit_config).

    6. OptimizationResult = MinimizeScalarRoutine(
            function=CalculateObjectiveCriterion, // using Eta_val, SigmaInv_val, UniqueClasses, objective_metric_config
            bounds=h_bounds_config,
            args=(Stored_X_train, y_train_labels, Eta_val, SigmaInv_val, UniqueClasses, objective_metric_config)
       ).
    7. IF OptimizationResult is successful THEN
    8.     h_optimal = OptimizationResult.x.
    9. ELSE
    10.    // Attempt to use boundary values for h_optimal based on criterion evaluation
    11.    IF no valid h_optimal found THEN RAISE Error.
    12. END IF

    13. K_optimal_train_matrix = ExternalKernelFunction(Stored_X_train, Stored_X_train, Eta_val, SigmaInv_val, h_optimal).

    14. SVM_classifier = Initialize_SVC(kernel='precomputed', **svm_args_config).
    15. SVM_classifier.fit(K_optimal_train_matrix, y_train_labels).

    16. Store h_optimal, Eta_val, SigmaInv_val, Stored_X_train, UniqueClasses, SVM_classifier in model.
    17. RETURN model.
```

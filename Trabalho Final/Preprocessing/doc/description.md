## 1. High-Correlation Feature Filter

This transformer identifies and removes features from a dataset that exhibit a high degree of linear correlation with other features.

### Conceptual Definition
The High-Correlation Feature Filter assesses the pairwise Pearson correlation between all features in a dataset. If the absolute correlation between a pair of features exceeds a specified upper limit, one of the features in the pair is marked for removal. This process aims to reduce multicollinearity and dimensionality.

### Mathematical Formulation

**Initialization:**
The filter is initialized with a "correlation magnitude upper limit", denoted $\tau_{corr} \in [0, 1]$. (Default value is $0.9$).

**Fitting Process:**
The fitting process identifies the set of feature indices to be removed.
**Inputs:**
* Data matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$, where $N$ is the number of samples and $D$ is the number of features.
* Optional target vector $\bm{y}$ (not used in this filter's logic).

**Procedure:**
1.  **Input Validation:** The input $\mathbf{X}$ is validated. If $D < 2$, no comparisons are possible, and the set of features to remove is empty.
2.  **Correlation Matrix Computation:** The $D \times D$ Pearson correlation matrix $\mathbf{C}$ is computed from $\mathbf{X}$. Each element $C_{ij}$ is the correlation coefficient between feature $i$ (column $\bm{x}_i$) and feature $j$ (column $\bm{x}_j$):
    $$C_{ij} = \frac{\sum_{k=1}^{N} (X_{ki} - \bar{X}_i)(X_{kj} - \bar{X}_j)}{\sqrt{\sum_{k=1}^{N} (X_{ki} - \bar{X}_i)^2} \sqrt{\sum_{k=1}^{N} (X_{kj} - \bar{X}_j)^2}}$$
    where $X_{ki}$ is the $k$-th sample of the $i$-th feature, and $\bar{X}_i$ is the mean of the $i$-th feature.
3.  **Identification of Highly Correlated Pairs:** Pairs of distinct features $(i, j)$ are identified such that their absolute correlation $|C_{ij}|$ exceeds the upper limit $\tau_{corr}$. To avoid redundant checks and self-correlation, only pairs where $i < j$ are considered (upper triangle of the correlation matrix).
4.  **Determination of Features for Removal:** A "set of feature indices identified for removal", denoted $\mathcal{I}_{drop}$, is formed. For each identified pair $(i, j)$ where $i < j$ and $|C_{ij}| > \tau_{corr}$, the index $j$ (the higher index in the pair) is added to $\mathcal{I}_{drop}$. Duplicate indices are removed to form the final set.
    $$\mathcal{I}_{drop} = \{ j \mid \exists i < j \text{ such that } |C_{ij}| > \tau_{corr} \}$$

**Transformation Process:**
The transformation process removes the identified features from a given data matrix.
**Inputs:**
* Data matrix $\mathbf{X}' \in \mathbb{R}^{N' \times D'}$. (Note: $D'$ might be different from $D$ during fitting if applied to different data, but should correspond to the original feature space for the stored $\mathcal{I}_{drop}$ to be meaningful).

**Procedure:**
1.  **Input Validation:** The input $\mathbf{X}'$ is validated. It's checked that the filter has been fitted.
2.  **Feature Removal:** If $\mathcal{I}_{drop}$ is empty, a copy of $\mathbf{X}'$ is returned. Otherwise, columns of $\mathbf{X}'$ whose indices are in $\mathcal{I}_{drop}$ (and are valid for $\mathbf{X}'$) are removed. The resulting matrix $\mathbf{X}'' \in \mathbb{R}^{N' \times (D' - |\mathcal{I}_{valid\_drop}|)}$ is returned.

### Algorithmic Process

**Algorithm: Fitting the High-Correlation Feature Filter**
```pseudocode
ALGORITHM FitCorrelationFilter(Input_Data X_fit, Correlation_Limit tau_c):
    INPUT:
        X_fit: Data matrix (N samples, D features).
        tau_c: Correlation magnitude upper limit.
    OUTPUT:
        Set_of_Indices_to_Drop I_drop.

    1. VALIDATE X_fit.
    2. LET D be the number of features in X_fit.
    3. IF D < 2 THEN
    4.     I_drop = EmptySet.
    5.     RETURN I_drop.
    6. END IF

    7. Correlation_Matrix C = ComputePearsonCorrelationMatrix(X_fit, features_are_columns=true).
    8. I_drop = EmptySet.
    9. FOR i FROM 0 TO D-2:
    10.    FOR j FROM i+1 TO D-1:
    11.        IF AbsoluteValue(C[i,j]) > tau_c THEN
    12.            ADD j to I_drop. // Add the index of the second feature in the pair
    13.        END IF
    14.    END FOR
    15. END FOR
    16. I_drop = UniqueElements(I_drop).
    17. RETURN I_drop.
```

**Algorithm: Transforming Data with High-Correlation Feature Filter**
```pseudocode
ALGORITHM TransformWithCorrelationFilter(Input_Data X_transform, Fitted_Indices_to_Drop I_drop):
    INPUT:
        X_transform: Data matrix (N_prime samples, D_prime features).
        I_drop: Set of feature indices identified for removal from fitting.
    OUTPUT:
        Transformed_Data X_transformed.

    1. VALIDATE X_transform.
    2. ENSURE filter has been fitted (I_drop is available).
    3. IF I_drop is empty THEN
    4.     RETURN Copy(X_transform).
    5. END IF

    6. Valid_I_drop = Filter I_drop to contain only indices < D_prime.
       // Log warning if some indices in I_drop were out of bounds for X_transform.
    7. X_transformed = Remove columns from X_transform specified by Valid_I_drop.
    8. RETURN X_transformed.
```

---
## 2. Sequential Preprocessing Transformer

This transformer applies a fixed sequence of preprocessing steps to the data. The sequence includes variance-based feature selection, filtering of highly correlated features, data scaling, and Principal Component Analysis (PCA).

### Conceptual Definition
The Sequential Preprocessing Transformer streamlines data preparation by chaining several common preprocessing techniques into a single, reusable component. Each step transforms the data, and the output of one step becomes the input to the next.

### Description of Pipeline Stages

The "sequence of transformation stages" comprises the following, applied in order:

1.  **Low-Variance Feature Removal:**
    * **Objective:** To remove features with very low variance, as they may not contribute significantly to the modeling process.
    * **Mechanism:** A feature $\bm{x}_j$ (column $j$) is removed if its sample variance $s_j^2$ is below a specified threshold $\tau_{var}$.
        $$s_j^2 = \frac{1}{N-1}\sum_{k=1}^{N} (X_{kj} - \bar{X}_j)^2$$
        If $s_j^2 < \tau_{var}$, feature $j$ is discarded. (The pipeline uses $\tau_{var} = 0.05$).
    * **Output:** Data matrix with potentially fewer features.

2.  **High-Correlation Feature Filtering:**
    * **Objective:** To reduce multicollinearity by removing one feature from any pair of highly correlated features.
    * **Mechanism:** The "High-Correlation Feature Filter" (described in Section 1) is applied. (The pipeline uses a correlation magnitude upper limit $\tau_{corr} = 0.95$).
    * **Output:** Data matrix with potentially fewer features.

3.  **Data Scaling (Standardization):**
    * **Objective:** To transform features to have zero mean and unit standard deviation. This is often beneficial for algorithms sensitive to feature magnitudes.
    * **Mechanism:** For each feature $\bm{x}_j$ with mean $\bar{X}_j$ and standard deviation $s_j$:
        Each value $X_{kj}$ in the feature is transformed to $X'_{kj}$:
        $$X'_{kj} = \frac{X_{kj} - \bar{X}_j}{s_j}$$
        If $s_j = 0$, $X'_{kj}$ is set to $0$.
    * **Output:** Data matrix with scaled features.

4.  **Principal Component Analysis (PCA):**
    * **Objective:** To reduce dimensionality by transforming the data into a new set of uncorrelated variables (principal components) that capture the most variance in the data.
    * **Mechanism:**
        1.  The input data (which is already mean-centered by the previous scaling step) is considered: $\mathbf{X}_{scaled}$.
        2.  The covariance matrix of $\mathbf{X}_{scaled}$ is computed: $\mathbf{S} = \frac{1}{N-1} \mathbf{X}_{scaled}^T \mathbf{X}_{scaled}$.
        3.  Eigenvalue decomposition of $\mathbf{S}$ is performed to find eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D$ and corresponding eigenvectors $\bm{w}_1, \dots, \bm{w}_D$.
        4.  A subset of $D_{pca}$ eigenvectors (principal components) is chosen, typically corresponding to the largest eigenvalues. (The pipeline uses default PCA settings, which typically means $D_{pca} = \min(N, D_{\text{input\_to\_PCA}})$).
        5.  A projection matrix $\mathbf{W} = [\bm{w}_1, \dots, \bm{w}_{D_{pca}}]$ is formed.
        6.  The data is transformed: $\mathbf{X}_{pca} = \mathbf{X}_{scaled} \mathbf{W}$.
    * **Output:** Data matrix transformed into the principal component space, with $D_{pca}$ features.

### Overall Process

**Fitting Process:**
The fitting process involves fitting each stage of the pipeline sequentially on the training data. The input data for each stage is the output of the previous stage.
**Inputs:**
* Data matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$.
* Optional target vector $\bm{y}$.

**Procedure:**
1.  Fit the Low-Variance Feature Removal stage on $\mathbf{X}$, then transform $\mathbf{X}$ to get $\mathbf{X}^{(1)}$.
2.  Fit the High-Correlation Feature Filtering stage on $\mathbf{X}^{(1)}$, then transform $\mathbf{X}^{(1)}$ to get $\mathbf{X}^{(2)}$.
3.  Fit the Data Scaling stage on $\mathbf{X}^{(2)}$, then transform $\mathbf{X}^{(2)}$ to get $\mathbf{X}^{(3)}$.
4.  Fit the PCA stage on $\mathbf{X}^{(3)}$.

**Transformation Process:**
The transformation process applies all fitted stages of the pipeline sequentially to new data.
**Inputs:**
* Data matrix $\mathbf{X}' \in \mathbb{R}^{N' \times D}$. (Must have the same original number of features as the data used for fitting).

**Procedure:**
1.  Transform $\mathbf{X}'$ using the fitted Low-Variance Feature Removal stage to get $\mathbf{X}'^{(1)}$.
2.  Transform $\mathbf{X}'^{(1)}$ using the fitted High-Correlation Feature Filtering stage to get $\mathbf{X}'^{(2)}$.
3.  Transform $\mathbf{X}'^{(2)}$ using the fitted Data Scaling stage to get $\mathbf{X}'^{(3)}$.
4.  Transform $\mathbf{X}'^{(3)}$ using the fitted PCA stage to get $\mathbf{X}'^{(4)}$.
The final transformed data $\mathbf{X}'^{(4)}$ is returned.

### Algorithmic Process

**Algorithm: Fitting the Sequential Preprocessing Transformer**
```pseudocode
ALGORITHM FitSequentialPreprocessor(Input_Data X_fit, Optional_Labels y_fit):
    INPUT:
        X_fit: Training data matrix.
        y_fit: Optional training labels.
    OUTPUT:
        Fitted_Pipeline.

    1. // Stage 1: Low-Variance Feature Removal (threshold tau_var = 0.05)
    2. Fit Variance_Filter on X_fit.
    3. X_intermediate_1 = Transform X_fit using Variance_Filter.

    4. // Stage 2: High-Correlation Feature Filtering (correlation limit tau_corr = 0.95)
    5. Fit Correlation_Filter on X_intermediate_1 (as per Section 1 algorithm).
    6. X_intermediate_2 = Transform X_intermediate_1 using Correlation_Filter.

    7. // Stage 3: Data Scaling (Standardization)
    8. Fit Scaler on X_intermediate_2.
    9. X_intermediate_3 = Transform X_intermediate_2 using Scaler.

    10. // Stage 4: Principal Component Analysis
    11. Fit PCA_Transformer on X_intermediate_3.

    12. Store Fitted Variance_Filter, Correlation_Filter, Scaler, PCA_Transformer as Fitted_Pipeline.
    13. RETURN Fitted_Pipeline.
```

**Algorithm: Transforming Data with Sequential Preprocessing Transformer**
```pseudocode
ALGORITHM TransformWithSequentialPreprocessor(Input_Data X_transform, Fitted_Pipeline):
    INPUT:
        X_transform: Data matrix to transform.
        Fitted_Pipeline (containing fitted stages: Variance_Filter, Correlation_Filter, Scaler, PCA_Transformer).
    OUTPUT:
        Transformed_Data X_final.

    1. X_intermediate_1 = Transform X_transform using Fitted_Pipeline.Variance_Filter.
    2. X_intermediate_2 = Transform X_intermediate_1 using Fitted_Pipeline.Correlation_Filter.
    3. X_intermediate_3 = Transform X_intermediate_2 using Fitted_Pipeline.Scaler.
    4. X_final = Transform X_intermediate_3 using Fitted_Pipeline.PCA_Transformer.
    5. RETURN X_final.
```

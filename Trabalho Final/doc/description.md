## Model Performance Comparison Framework

The core of the script is a framework designed to systematically evaluate and compare different classification model configurations. This involves standardized data handling, model training and evaluation using cross-validation, performance metric recording (accuracy and computation time), and statistical comparison of model performances.

### 1. Experimental Setup and Configuration

**Key Parameters:**
* **Random Seed:** A fixed seed for random number generators to ensure reproducibility of results, particularly for data shuffling and model initializations.
* **Number of Cross-Validation Splits ($K_{cv}$):** The number of folds to use in the stratified K-fold cross-validation process (e.g., $K_{cv}=10$).
* **Significance Level for Equivalence Heuristic ($\alpha_{equiv}$):** A threshold for p-values used to heuristically determine if the performance of two models is statistically "equivalent" (e.g., $\alpha_{equiv}=0.05$).
* **Data Source:** A designated directory containing datasets, typically in `.npz` format, each with features $\mathbf{X}$ and labels $\bm{y}$.
* **Results Storage:** A JSON file for persistent storage of comparison outcomes.

**Model Configurations:**
The framework defines a set of model pipelines for comparison. Each pipeline typically includes a preprocessing stage followed by a classifier. The script specifically configures three pipelines:
1.  **Model $M_{2D}$ (Optimized Spatial):** Consists of a "Sequential Preprocessing Transformer" (analyzed previously, including variance thresholding, correlation filtering, scaling, and PCA) followed by a "Custom Kernel Support Vector Classifier" (analyzed previously). The custom classifier is configured to use a 'spatial' objective metric for its internal hyperparameter optimization and a 'scale' type for its initial kernel parameter estimation.
2.  **Model $M_{Axis}$ (Optimized Axis):** Similar to $M_{2D}$, but the "Custom Kernel Support Vector Classifier" is configured to use an 'axis' objective metric.
3.  **Model $M_{Ref}$ (Reference):** Consists of the same "Sequential Preprocessing Transformer" followed by a standard Support Vector Classifier (SVC), typically with its default configuration (e.g., RBF kernel).

### 2. Performance Evaluation Methodology

**Stratified K-Fold Cross-Validation Procedure:**
For each dataset $\mathcal{D} = \{(\bm{x}_i, y_i)\}_{i=1}^N$ and for each configured model $M$:
1.  The dataset $\mathcal{D}$ is partitioned into $K_{cv}$ disjoint folds, $\mathcal{D}_1, \mathcal{D}_2, \dots, \mathcal{D}_{K_{cv}}$. This partitioning is stratified, meaning it attempts to preserve the percentage of samples for each class in all folds.
2.  For each fold $k \in \{1, \dots, K_{cv}\}$:
    * The model $M$ is trained on the training set $\mathcal{T}_k = \mathcal{D} \setminus \mathcal{D}_k$. Let the trained model instance be $M_k$.
    * The performance of $M_k$ is evaluated on the validation set $\mathcal{V}_k = \mathcal{D}_k$.
3.  **Classification Accuracy Metric:** The primary performance metric is classification accuracy. For each fold $k$, the accuracy $A_k(M)$ is:
    $$A_k(M) = \frac{1}{|\mathcal{V}_k|} \sum_{(\bm{x},y) \in \mathcal{V}_k} \mathbb{I}(M_k(\bm{x}) = y)$$
    where $\mathbb{I}(\cdot)$ is the indicator function (1 if the condition is true, 0 otherwise), and $M_k(\bm{x})$ is the prediction of the model trained on $\mathcal{T}_k$ for sample $\bm{x}$.
    This process yields a vector of $K_{cv}$ accuracy scores for model $M$ on dataset $\mathcal{D}$: $\bm{A}(M) = (A_1(M), A_2(M), \dots, A_{K_{cv}}(M))$.

**Aggregated Performance Metrics (per dataset, per model):**
From the vector of cross-validation accuracy scores $\bm{A}(M)$:
* **Mean Accuracy ($\bar{A}(M)$):**
    $$\bar{A}(M) = \frac{1}{K_{cv}} \sum_{k=1}^{K_{cv}} A_k(M)$$
* **Standard Deviation of Accuracy ($s_A(M)$):**
    $$s_A(M) = \sqrt{\frac{1}{K_{cv}-1} \sum_{k=1}^{K_{cv}} (A_k(M) - \bar{A}(M))^2}$$ (Sample standard deviation).
    The reported accuracy is typically in the format "$\bar{A}(M) \pm s_A(M)$".

**Computation Time Recording:**
The total wall-clock time taken to complete the $K_{cv}$-fold cross-validation for each model $M$ on each dataset $\mathcal{D}$ is recorded.

### 3. Statistical Comparison of Models

**Paired Samples t-test Procedure:**
To compare the performance of two models, $M_1$ and $M_2$, on a given dataset, a paired samples t-test is applied to their respective vectors of cross-validation accuracy scores, $\bm{A}(M_1)$ and $\bm{A}(M_2)$. The pairing arises from the use of identical cross-validation folds for both models.
1.  **Difference Scores:** Compute the element-wise differences: $d_k = A_k(M_1) - A_k(M_2)$ for $k=1, \dots, K_{cv}$.
2.  **Null Hypothesis ($H_0$):** The true mean of these differences is zero ($\mu_d = 0$). This implies that, on average, the models have the same accuracy on this dataset.
3.  **Alternative Hypothesis ($H_1$):** The true mean of these differences is not zero ($\mu_d \neq 0$).
4.  **Test Statistic ($t$):**
    $$t = \frac{\bar{d}}{s_d / \sqrt{K_{cv}}}$$
    where $\bar{d} = \frac{1}{K_{cv}}\sum d_k$ is the sample mean of the differences, and $s_d = \sqrt{\frac{1}{K_{cv}-1}\sum (d_k - \bar{d})^2}$ is the sample standard deviation of the differences.
5.  **P-value Calculation:** The p-value is derived from the t-statistic using the t-distribution with $K_{cv}-1$ degrees of freedom.
    * If the observed scores for $M_1$ and $M_2$ are identical across all folds (i.e., all $d_k=0$), $s_d=0$, leading to an undefined t-statistic or a NaN p-value. The script handles this by interpreting such cases as a p-value of $1.0$.

**Heuristic for Performance Equivalence:**
Based on the calculated p-value and the predefined "significance level for equivalence heuristic" $\alpha_{equiv}$:
* The performances of models $M_1$ and $M_2$ are deemed "statistically equivalent" if p-value $ > \alpha_{equiv}$.
* Otherwise, they are considered to have statistically different performances.
It's important to note that failing to reject $H_0$ (i.e., p-value $ > \alpha_{equiv}$) does not formally prove that the models are equivalent, but rather that there is insufficient evidence to conclude they are different at the chosen significance level.

### 4. Comparative Analysis Execution and Reporting

The overall execution flow involves iterating through all specified datasets. For each dataset:
1.  Data is loaded and subjected to validity checks (e.g., sufficient samples, number of classes). Datasets failing checks are skipped.
2.  If results for the current dataset are already present in the persistent storage, it may be skipped.
3.  Each configured model pipeline ($M_{2D}, M_{Axis}, M_{Ref}$) undergoes the Stratified K-Fold Cross-Validation Procedure, yielding accuracy scores and computation times.
4.  Mean accuracy, standard deviation of accuracy, and computation time are recorded for each model.
5.  Specified pairs of models (e.g., ($M_{2D}$, $M_{Ref}$), ($M_{Axis}$, $M_{Ref}$), ($M_{2D}$, $M_{Axis}$)) are compared using the Paired Samples t-test Procedure. The resulting p-value and the outcome of the "equivalence" heuristic are recorded.
6.  All results for the current dataset (accuracies, times, p-values, equivalence flags) are saved to the persistent JSON storage file.
This process is repeated for all datasets.

### Algorithmic Process: Top-Level Comparison Logic

```pseudocode
ALGORITHM ExecuteComparativeAnalysis:
    INPUT:
        List_of_Datasets D_paths.
        Set_of_Models M_configs.
        CV_Splits K_cv_val.
        P_Value_Threshold alpha_eq_val.
        Results_Storage_File R_file.
    OUTPUT:
        Updated Results_Storage_File R_file.

    1. Initialize Overall_Results = LoadResultsFromFile(R_file).
    2. Initialize CV_Strategy = StratifiedKFold(splits=K_cv_val, shuffle=true, random_state=SEED).

    3. FOR EACH DatasetPath dp IN D_paths:
    4.     DatasetName = GetName(dp).
    5.     IF DatasetName is in Overall_Results THEN
    6.         Log "Skipping {DatasetName} (already processed)".
    7.         CONTINUE FOR.
    8.     END IF

    9.     TRY Load Data (X_data, y_labels) from dp.
    10.        VALIDATE X_data, y_labels (size, classes, suitability for K_cv_val).
    11.    CATCH Error e:
    12.        Log "Error loading/validating {DatasetName}: {e}".
    13.        CONTINUE FOR.
    14.    END TRY

    15.    Initialize Dataset_Model_Scores = empty map.
    16.    Initialize Dataset_Model_Times = empty map.

    17.    FOR EACH ModelName mn, ModelPipeline mp IN M_configs:
    18.        Log "Processing {mn} on {DatasetName}".
    19.        StartTime = CurrentTime().
    20.        TRY
    21.            Fold_Accuracies = CrossValidate(mp, X_data, y_labels, cv=CV_Strategy, scoring='accuracy').
    22.            Duration = CurrentTime() - StartTime.
    23.            Dataset_Model_Scores[mn] = Fold_Accuracies.
    24.            Dataset_Model_Times[mn] = Duration.
    25.        CATCH Error e:
    26.            Duration = CurrentTime() - StartTime.
    27.            Log "Error CV for {mn} on {DatasetName}: {e}. Time: {Duration}".
    28.            Dataset_Model_Scores[mn] = Null_or_Error_Indicator.
    29.            Dataset_Model_Times[mn] = Duration. // Time until error
    30.        END TRY
    31.    END FOR

    32.    Initialize Current_Dataset_Results = BuildResultStructure().
    33.    FOR EACH ModelName mn, Scores sc IN Dataset_Model_Scores:
    34.        IF sc is valid THEN
    35.            Current_Dataset_Results.Accuracy[mn] = Format(Mean(sc), StdDev(sc)).
    36.        ELSE
    37.            Current_Dataset_Results.Accuracy[mn] = "Error".
    38.        END IF
    39.    END FOR
    40.    FOR EACH ModelName mn, Time_val tm IN Dataset_Model_Times:
    41.         Current_Dataset_Results.Time[mn] = Format(tm).
    42.    END FOR

    43.    FOR EACH Pair (Model1_Name m1n, Model2_Name m2n) in Predefined_Comparisons:
    44.        Scores_m1 = Dataset_Model_Scores.Get(m1n).
    45.        Scores_m2 = Dataset_Model_Scores.Get(m2n).
    46.        IF Scores_m1 and Scores_m2 are valid THEN
    47.            TRY
    48.                PVal = Paired_t_Test(Scores_m1, Scores_m2).PValue.
    49.                IF PVal is NaN THEN PVal = 1.0. // Handle identical score arrays
    50.                Current_Dataset_Results.PValue[m1n vs m2n] = PVal.
    51.                Current_Dataset_Results.Equivalent[m1n vs m2n] = (PVal > alpha_eq_val).
    52.            CATCH Error e:
    53.                Log "Error t-test for {m1n} vs {m2n} on {DatasetName}: {e}".
    54.                Current_Dataset_Results.PValue[m1n vs m2n] = "Error".
    55.                Current_Dataset_Results.Equivalent[m1n vs m2n] = "Error".
    56.            END TRY
    57.        ELSE
    58.            Current_Dataset_Results.PValue[m1n vs m2n] = "Scores unavailable".
    59.            Current_Dataset_Results.Equivalent[m1n vs m2n] = "Scores unavailable".
    60.        END IF
    61.    END FOR

    62.    Overall_Results[DatasetName] = Current_Dataset_Results.
    63.    SaveResultsToFile(R_file, Overall_Results).
    64. END FOR
    65. Log "All datasets processed".
```

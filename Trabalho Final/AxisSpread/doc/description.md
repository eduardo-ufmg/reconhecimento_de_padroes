## 1. Measure of Even Spacing for a Numerical Sequence (Vector Evenness Score)

This metric evaluates how uniformly distributed the values are within a one-dimensional numerical sequence. A score approaching 1.0 signifies more even spacing, while a score closer to 0.0 indicates more clustered or uneven spacing.

### Conceptual Definition
The Vector Evenness Score is derived from the statistical properties of the differences between consecutive elements in the sorted sequence. It combines a factor related to the standard deviation of these differences (penalizing variability in spacing) with a factor related to the normalized mean of these differences (favoring larger average spacing relative to the maximum spacing).

### Mathematical Formulation

**Inputs:**
A numerical sequence (vector) $V = (v_1, v_2, \dots, v_n)$ containing $n$ real numbers.

**Preprocessing and Special Cases:**
1.  The input sequence $V$ is first converted to an array of floating-point numbers. If this conversion is not possible or if any element $v_i$ is Not-a-Number (NaN), the score is undefined.
2.  If $n=0$ (empty sequence), the score is undefined.
3.  If $n=1$ (single element sequence), the score $S_V$ is defined as $1.0$.

**Core Calculation (for $n \ge 2$):**
1.  **Sorting:** The sequence $V$ is sorted to produce $V'_{sorted} = (v'_{(1)}, v'_{(2)}, \dots, v'_{(n)})$, where $v'_{(j)} \le v'_{(j+1)}$.
2.  **Difference Sequence ($\Delta$):** A sequence of $n-1$ differences is computed from $V'_{sorted}$:
    $\Delta = (\delta_1, \delta_2, \dots, \delta_{n-1})$, where $\delta_j = v'_{(j+1)} - v'_{(j)}$ for $j=1, \dots, n-1$.
    Since $V'_{sorted}$ is sorted, all $\delta_j \ge 0$.

3.  **Statistical Measures of $\Delta$ (for $n \ge 2$):**
    * **Mean of Differences ($\mu_{\Delta}$):**
        $$\mu_{\Delta} = \frac{1}{n-1} \sum_{j=1}^{n-1} \delta_j$$
    * **Population Standard Deviation of Differences ($\sigma_{\Delta}$):**
        $$\sigma_{\Delta} = \sqrt{\frac{1}{n-1} \sum_{j=1}^{n-1} (\delta_j - \mu_{\Delta})^2}$$
        If $n=2$, $\Delta$ has one element, say $\delta_1$. Then $\mu_{\Delta} = \delta_1$ and $\sigma_{\Delta} = 0$.
    * **Maximum Difference ($\delta_{max}$):**
        $$\delta_{max} = \max(\delta_1, \delta_2, \dots, \delta_{n-1})$$
        If all $\delta_j = 0$ (i.e., all $v_i$ in $V$ are identical), then $\delta_{max} = 0$.

4.  **Score Components (for $n \ge 2$):**
    * **Standard Deviation Score ($S_{\sigma}$):** This component favors low variability in spacing.
        $$S_{\sigma} = \frac{1}{1 + \sigma_{\Delta}}$$
        If $\sigma_{\Delta}$ is NaN or infinite (e.g., due to infinite values in the input $V$), $S_{\sigma}$ and thus the final score are undefined.
    * **Mean Spacing Score ($S_{\mu}$):** This component reflects the magnitude of average spacing relative to the largest spacing.
        $$S_{\mu} = \begin{cases} \frac{\mu_{\Delta}}{\delta_{max}} & \text{if } \delta_{max} \neq 0 \\ 0 & \text{if } \delta_{max} = 0 \end{cases}$$
        Note: If $\delta_{max}=0$, then all $\delta_j=0$, which implies $\mu_{\Delta}=0$.

5.  **Final Vector Evenness Score ($S_V$ for $n \ge 2$):**
    $$S_V = S_{\sigma} \times S_{\mu}$$

**Output:**
A scalar score $S_V$, typically between $0.0$ and $1.0$. A score of $1.0$ indicates perfectly uniform, non-zero spacing (for $n \ge 2$) or a single/two-element sequence. A score of $0.0$ can occur if all elements are identical (for $n \ge 2$), leading to $\delta_{max}=0$.

**Conditions for Undefined Result:**
The score is undefined (e.g., returns `None` or `NaN`) if:
* The input sequence cannot be converted to a numeric array of floats.
* The input sequence contains NaN values.
* The input sequence is empty ($n=0$).
* The calculated $\sigma_{\Delta}$ is NaN or infinite.

### Algorithmic Process: Computation of Vector Evenness Score

```pseudocode
ALGORITHM CalculateVectorEvennessScore(InputSequence V):
    INPUT:
        V: A sequence of numbers (v_1, ..., v_n).
    OUTPUT:
        Score_Value: Scalar (approx. 0.0 to 1.0), or an indicator of an undefined result.

    1. ATTEMPT conversion of V to NumericArray of floats. IF failed, RETURN Undefined.
    2. IF NumericArray contains any NaN, RETURN Undefined.
    3. LET n = length of NumericArray.
    4. IF n = 0, RETURN Undefined.
    5. IF n = 1, RETURN 1.0.

    6. Sorted_V = Sort(NumericArray).
    7. IF n >= 2 THEN
    8.     Differences_Delta = sequence of (Sorted_V[j+1] - Sorted_V[j]) for j=0 to n-2.
    9. ELSE /* This case (n<2) is handled by steps 4 & 5, but for completeness */
    10.    Differences_Delta = empty sequence. /* Or handle as per n=1 directly */
    11. END IF

    12. PopStdDev_Delta = PopulationStandardDeviation(Differences_Delta).
        /* If n=2, Differences_Delta has 1 element, PopStdDev_Delta is 0. */
    13. IF PopStdDev_Delta is NaN or Infinite, RETURN Undefined.

    14. Mean_Delta = Mean(Differences_Delta).
    15. Max_Delta_Difference = Maximum(Differences_Delta). (If Differences_Delta is empty, this needs careful definition; however, n>=2 ensures it's not empty).

    16. Score_Sigma = 1.0 / (1.0 + PopStdDev_Delta).

    17. IF Max_Delta_Difference = 0 THEN
    18.    Score_Mu = 0.0.
    19. ELSE
    20.    Score_Mu = Mean_Delta / Max_Delta_Difference.
    21. END IF

    22. Final_Score = Score_Sigma * Score_Mu.
    23. RETURN Final_Score.
```

---

## 2. Comparative Spread Objective for Labeled Feature Sets

This objective function computes a score that reflects the evenness of data distribution within two distinct feature sets, based on associated class labels. It favors scenarios where both relevant sub-sequences exhibit high evenness scores and these scores are similar.

### Conceptual Definition
The objective value is derived by:
1.  Identifying two sub-sequences based on class labels: one from a primary feature set for class '0', and another from a secondary feature set for class '1'.
2.  Calculating the Vector Evenness Score (described in Section 1) for each of these two sub-sequences.
3.  Combining these two scores. The combination formula is the average of the two scores minus the absolute difference between them, effectively rewarding high average spread and penalizing disparity in spread scores.

### Mathematical Formulation

**Inputs:**
* $X = (x_1, x_2, \dots, x_M)$: A primary numerical sequence of $M$ feature values.
* $W = ({W_1}, w_2, \dots, w_M)$: A secondary numerical sequence of $M$ feature values.
* $L = (l_1, l_2, \dots, l_M)$: A sequence of $M$ binary class labels, where each $l_i \in \{0, 1\}$.

**Sub-sequence Extraction:**
* Let $V_{X_0} = \{x_i \mid l_i = 0\}$ be the sub-sequence of elements from $X$ corresponding to label '0'.
* Let $V_{W_1} = \{w_i \mid l_i = 1\}$ be the sub-sequence of elements from $W$ corresponding to label '1'.

**Spread Calculations:**
* Calculate the Vector Evenness Score for $V_{X_0}$, denoted $S_{X_0} = \text{CalculateVectorEvennessScore}(V_{X_0})$.
* Calculate the Vector Evenness Score for $V_{W_1}$, denoted $S_{W_1} = \text{CalculateVectorEvennessScore}(V_{W_1})$.

**Objective Function Value ($J_O$):**
If either $S_{X_0}$ or $S_{W_1}$ is undefined (e.g., due to empty sub-sequences or other issues in their calculation), then $J_O$ is undefined (e.g., NaN).
Otherwise, $J_O$ is calculated as:
$$J_O = \frac{S_{X_0} + S_{W_1}}{2} - |S_{X_0} - S_{W_1}|$$
This value rewards similarity between $S_{X_0}$ and $S_{W_1}$ and their magnitudes. The maximum is achieved when $S_{X_0} = S_{W_1}$, in which case $J_O = S_{X_0} (=S_{W_1})$. If the scores differ significantly, the penalty $|S_{X_0} - S_{W_1}|$ reduces $J_O$.

**Output:**
A scalar value $J_O \in \mathbb{R}$, or an indicator of an undefined result (NaN).

**Conditions for Undefined Result:**
The objective value is undefined if the evenness score for either $V_{X_0}$ or $V_{W_1}$ cannot be computed (e.g., if a sub-sequence is empty and `CalculateVectorEvennessScore` returns Undefined for empty inputs, or if it contains NaNs).

### Algorithmic Process: Computation of Comparative Spread Objective

```pseudocode
ALGORITHM CalculateComparativeSpreadObjective(Sequence_X, Sequence_W, Labels_L):
    INPUT:
        Sequence_X: A numerical sequence.
        Sequence_W: A numerical sequence (same length as X).
        Labels_L: A sequence of labels {0,1} (same length as X).
    OUTPUT:
        ObjectiveValue: Scalar, or an indicator of an undefined result.

    1. SubSequence_X0 = Filter Sequence_X for elements where corresponding Label in L is 0.
    2. SubSequence_W1 = Filter Sequence_W for elements where corresponding Label in L is 1.

    3. Score_X0 = CalculateVectorEvennessScore(SubSequence_X0).
    4. Score_W1 = CalculateVectorEvennessScore(SubSequence_W1).

    5. IF Score_X0 is Undefined OR Score_W1 is Undefined THEN
    6.     RETURN Undefined.
    7. END IF

    8. Mean_Of_Scores = (Score_X0 + Score_W1) / 2.0.
    9. Absolute_Difference_Of_Scores = AbsoluteValue(Score_X0 - Score_W1).

    10. ObjectiveValue = Mean_Of_Scores - Absolute_Difference_Of_Scores.
    11. RETURN ObjectiveValue.
```
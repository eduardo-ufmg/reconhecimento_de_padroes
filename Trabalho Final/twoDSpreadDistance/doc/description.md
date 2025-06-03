## 1. Mean Distance Within a Single Point Group

This metric quantifies the average internal spread or dispersion of points within a single designated group (or class).

### Conceptual Definition
The mean distance within a single point group is the arithmetic average of all unique pairwise Euclidean distances between points belonging to that group.

### Mathematical Formulation

**Inputs:**
A single group of $N$ points, denoted as $G = \{P_1, P_2, \dots, P_N\}$. Each point $P_i$ is defined by its coordinates $(x_i, y_i)$ in a 2D Cartesian space.

**Point Representation:**
Each point $P_i \in G$ is represented as $P_i = (x_i, y_i)$.

**Distance Metric:**
The Euclidean distance $d(P_i, P_j)$ between two points $P_i = (x_i, y_i)$ and $P_j = (x_j, y_j)$ is given by:
$d(P_i, P_j) = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$

**Calculation:**
The set of all unique pairwise distances within the group $G$ is $D_G = \{d(P_i, P_j) \mid 1 \le i < j \le N\}$.
The total number of such unique pairs is $\binom{N}{2} = \frac{N(N-1)}{2}$.

The mean distance within the point group, $\mu_G$, is calculated as:
$$\mu_G = \frac{\sum_{1 \le i < j \le N} d(P_i, P_j)}{\binom{N}{2}}$$
For this calculation to be meaningful, the number of points $N$ must be at least 2.

**Output:**
A non-negative scalar value $\mu_G \in \mathbb{R}_{\ge 0}$ representing the mean distance.

**Conditions for Undefined Result:**
The metric is undefined if the number of points $N$ in the group is less than 2, as no pairs can be formed.

### Algorithmic Process: Computation of Mean Internal Separation

```pseudocode
ALGORITHM AlgorithmForMeanInternalSeparation(Points_Group):
    INPUT:
        Points_Group: A collection of N points {P_1, ..., P_N}, where P_i = (x_i, y_i).
    OUTPUT:
        Mean_Distance_Value: Scalar, or an indicator of an undefined result.

    1. LET N be the number of points in Points_Group.
    2. IF N < 2 THEN
    3.     RETURN Undefined (e.g., NaN or error indication).
    4. END IF

    5. INITIALIZE an empty list All_Pairwise_Distances.
    6. FOR i FROM 1 TO N-1:
    7.     FOR j FROM i+1 TO N:
    8.         LET P_i be the i-th point in Points_Group.
    9.         LET P_j be the j-th point in Points_Group.
    10.        CALCULATE Euclidean_Distance d(P_i, P_j).
    11.        ADD Euclidean_Distance to All_Pairwise_Distances.
    12.    END FOR
    13. END FOR

    14. CALCULATE Mean_Distance_Value = Sum(All_Pairwise_Distances) / Count(All_Pairwise_Distances).
    15. RETURN Mean_Distance_Value.
```

---

## 2. Mean Distance Between Two Distinct Point Groups

This metric measures the average separation between points belonging to two different, specified groups.

### Conceptual Definition
The mean distance between two distinct point groups is the arithmetic average of all Euclidean distances between pairs of points, where one point is drawn from the first group and the other from the second group.

### Mathematical Formulation

**Inputs:**
A collection of $M$ points in total, $P_{total} = \{P_1, \dots, P_M\}$, where each point $P_k = (x_k, y_k)$.
A set of categorical labels $L = \{l_1, \dots, l_M\}$, where each $l_k \in \{0, 1\}$ assigns point $P_k$ to one of two groups, $G_0$ or $G_1$.

Let $G_0 = \{P_k \in P_{total} \mid l_k = 0\}$ be the set of $N_0$ points in group 0.
Let $G_1 = \{P_k \in P_{total} \mid l_k = 1\}$ be the set of $N_1$ points in group 1.

**Distance Metric:**
The Euclidean distance $d(A, B)$ between a point $A \in G_0$ and a point $B \in G_1$ is used.

**Calculation:**
The set of all distances between points in $G_0$ and points in $G_1$ is $D_{G_0,G_1} = \{d(A, B) \mid A \in G_0, B \in G_1\}$.
The total number of such pairs is $N_0 \times N_1$.

The mean distance between groups $G_0$ and $G_1$, denoted $\mu_{G_0,G_1}$, is:
$$\mu_{G_0,G_1} = \frac{\sum_{A \in G_0} \sum_{B \in G_1} d(A, B)}{N_0 N_1}$$
This calculation requires both $N_0 > 0$ and $N_1 > 0$.

**Output:**
A non-negative scalar value $\mu_{G_0,G_1} \in \mathbb{R}_{\ge 0}$ representing the mean distance between the two groups.

**Conditions for Undefined Result:**
The metric is undefined if either group $G_0$ or $G_1$ is empty (i.e., $N_0 = 0$ or $N_1 = 0$).

### Algorithmic Process: Computation of Mean Between-Group Separation

```pseudocode
ALGORITHM AlgorithmForMeanBetweenGroupSeparation(All_Points, All_Labels):
    INPUT:
        All_Points: A collection of M points {P_1, ..., P_M}.
        All_Labels: A list of M labels {l_1, ..., l_M}, where l_k is 0 or 1.
    OUTPUT:
        Mean_Inter_Group_Distance: Scalar, or an indicator of an undefined result.

    1. PARTITION All_Points into Group_0 and Group_1 based on All_Labels.
    2. LET N0 be the number of points in Group_0.
    3. LET N1 be the number of points in Group_1.

    4. IF N0 = 0 OR N1 = 0 THEN
    5.     RETURN Undefined (e.g., NaN or error indication).
    6. END IF

    7. INITIALIZE an empty list All_Inter_Group_Distances.
    8. FOR EACH point A_i IN Group_0:
    9.     FOR EACH point B_j IN Group_1:
    10.        CALCULATE Euclidean_Distance d(A_i, B_j).
    11.        ADD Euclidean_Distance to All_Inter_Group_Distances.
    12.    END FOR
    13. END FOR

    14. CALCULATE Mean_Inter_Group_Distance = Sum(All_Inter_Group_Distances) / Count(All_Inter_Group_Distances).
    15. RETURN Mean_Inter_Group_Distance.
```

---

## 3. Composite Separability and Cohesion Measure

This objective function provides a single value that balances the separation between two distinct groups of points against the average internal spread within these groups, adjusted by the variability of their internal spreads.

### Conceptual Definition
The measure is defined as the sum of the mean distance between the two groups and the average of their individual mean internal distances, penalized by the standard deviation of these two mean internal distances. A higher value suggests better separation between groups and/or maintained internal dispersion.

### Mathematical Formulation

**Inputs:**
Same as for the Mean Distance Between Two Distinct Point Groups:
A collection of $M$ points $P_{total} = \{P_1, \dots, P_M\}$ with coordinates $(x_k, y_k)$.
A set of categorical labels $L = \{l_1, \dots, l_M\}$ assigning points to group $G_0$ or $G_1$.

**Component Calculations:**

1.  **Mean Inter-Group Distance ($M_{inter}$):**
    Calculated as $\mu_{G_0,G_1}$ described in Section 2.
    $M_{inter} = \mu_{G_0,G_1}$.
    If $N_0=0$ or $N_1=0$, this component (and thus the objective function) is undefined.

2.  **Mean Intra-Group Distance for Group 0 ($M_{intra,0}$):**
    Calculated as $\mu_{G_0}$ (Section 1) using only points in $G_0$.
    Requires $N_0 \ge 2$. If not, this component is undefined.

3.  **Mean Intra-Group Distance for Group 1 ($M_{intra,1}$):**
    Calculated as $\mu_{G_1}$ (Section 1) using only points in $G_1$.
    Requires $N_1 \ge 2$. If not, this component is undefined.

4.  **Average of Mean Intra-Group Distances ($\bar{M}_{intra}$):**
    $\bar{M}_{intra} = \frac{M_{intra,0} + M_{intra,1}}{2}$

5.  **Standard Deviation of Mean Intra-Group Distances ($\sigma_{M_{intra}}$):**
    This is the population standard deviation of the two values $M_{intra,0}$ and $M_{intra,1}$.
    Given two values $a$ and $b$, their population standard deviation is $\frac{|a-b|}{2}$.
    So, $\sigma_{M_{intra}} = \frac{|M_{intra,0} - M_{intra,1}|}{2}$.

**Objective Function Value ($J$):**
The objective function $J$ is defined as:
$$J = M_{inter} + \bar{M}_{intra} - \sigma_{M_{intra}}$$
Substituting the expressions for $\bar{M}_{intra}$ and $\sigma_{M_{intra}}$:
$$J = M_{inter} + \frac{M_{intra,0} + M_{intra,1}}{2} - \frac{|M_{intra,0} - M_{intra,1}|}{2}$$
This expression simplifies to:
$$J = M_{inter} + \min(M_{intra,0}, M_{intra,1})$$

**Output:**
A scalar value $J \in \mathbb{R}$ representing the composite measure.

**Conditions for Undefined Result:**
The objective function value is undefined (e.g., results in NaN) if:
* Either group $G_0$ or $G_1$ is empty (affecting $M_{inter}$).
* Either group $G_0$ or $G_1$ contains fewer than two points (affecting $M_{intra,0}$ or $M_{intra,1}$ respectively).

### Algorithmic Process: Computation of Composite Separability Measure

```pseudocode
ALGORITHM AlgorithmForCompositeSeparabilityMeasure(All_Points, All_Labels):
    INPUT:
        All_Points: A collection of M points.
        All_Labels: A list of M labels (0 or 1).
    OUTPUT:
        Objective_Value: Scalar, or an indicator of an undefined result.

    1. TRY:
    2.     CALCULATE Inter_Group_Mean_Dist = AlgorithmForMeanBetweenGroupSeparation(All_Points, All_Labels).
    3. CATCH error (e.g., a group is empty):
    4.     RETURN Undefined.
    5. END TRY

    6. PARTITION All_Points into Points_Group_0 and Points_Group_1 based on All_Labels.

    7. TRY:
    8.     CALCULATE Intra_Group_0_Mean_Dist = AlgorithmForMeanInternalSeparation(Points_Group_0).
    9. CATCH error (e.g., Points_Group_0 has < 2 points):
    10.    RETURN Undefined.
    11. END TRY

    12. TRY:
    13.    CALCULATE Intra_Group_1_Mean_Dist = AlgorithmForMeanInternalSeparation(Points_Group_1).
    14. CATCH error (e.g., Points_Group_1 has < 2 points):
    15.    RETURN Undefined.
    16. END TRY

    17. CALCULATE Avg_Intra_Group_Mean_Dist = (Intra_Group_0_Mean_Dist + Intra_Group_1_Mean_Dist) / 2.0.

    18. // Population standard deviation for two values a, b is |a-b|/2
    19. CALCULATE StdDev_Intra_Group_Mean_Dists = Abs(Intra_Group_0_Mean_Dist - Intra_Group_1_Mean_Dist) / 2.0.

    20. CALCULATE Objective_Value = Inter_Group_Mean_Dist + Avg_Intra_Group_Mean_Dist - StdDev_Intra_Group_Mean_Dists.
        // This is equivalent to: Inter_Group_Mean_Dist + Min(Intra_Group_0_Mean_Dist, Intra_Group_1_Mean_Dist).

    21. RETURN Objective_Value.
```

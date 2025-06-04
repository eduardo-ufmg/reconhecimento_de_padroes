import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Generate synthetic data with redundant features
X, y = make_classification(
    n_samples=1000, n_features=3, n_informative=1, n_redundant=2, n_clusters_per_class=1
)

# Artificially create high correlation between first two features
X[:, 1] = X[:, 0] + np.random.normal(0, 0.1, X.shape[0])  # X1 is redundant copy of X0
X[:, 2] = np.random.normal(0, 1, X.shape[0])  # X2 is pure noise

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Print feature descriptions
print(
    """
Feature Definitions:
- X0: Informative feature (true signal)
- X1: Redundant copy of X0 (r = {:.3f})
- X2: Pure noise feature (r = {:.3f} with X0)
""".format(
        np.corrcoef(X[:, 0], X[:, 1])[0, 1], np.corrcoef(X[:, 0], X[:, 2])[0, 1]
    )
)

# Univariate feature selection
selector = SelectKBest(score_func=f_classif, k=2)
selector.fit(X_train, y_train)

# Explain selection results
print("SelectKBest Results:")
print("-" * 50)
for i in range(X.shape[1]):
    score = selector.scores_[i]
    pval = selector.pvalues_[i]
    status = "SELECTED" if selector.get_support()[i] else "REJECTED"
    print(f"X{i}: F-score = {score:.1f}, p-value = {pval:.3e} → {status}")

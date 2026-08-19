import numpy as np


def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # 0. convert to numpy array
    y_left = np.asarray(y_left, dtype=int)
    y_right = np.asarray(y_right, dtype=int)

    # 1. validate the inputs
    if y_left.ndim != 1 or y_right.ndim != 1:
        raise ValueError("inputs can't be more than 1-d")

    # 2. calculate number of samples
    n_l = len(y_left)
    n_r = len(y_right)
    n = n_l + n_r

    if n == 0:
        return 0.0

    # 3. calculate gini for each node
    def gini_node(y):
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)

        return 1.0 - np.sum(probs ** 2)

    gini_l = gini_node(y_left)
    gini_r = gini_node(y_right)

    # 4. calculate weighted gini
    gini = (n_l / n) * gini_l + (n_r / n) * gini_r

    return gini
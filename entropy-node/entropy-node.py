import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # 0. Convert to numpy array
    y = np.asarray(y)

    # 1. validate inputs
    if y.ndim != 1:
        raise ValueError("y has to be 1-dimensional")

    # 2. calculate class proportions
    _, counts = np.unique(y, return_counts=True)

    # 3. calculate entropy
    p_i = counts / np.sum(counts)
    entropy = - np.sum(p_i * np.log2(p_i))

    return entropy
import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
        
    # 0. convert to array
    x = np.asarray(x)
    y = np.asarray(y)

    # 1. perform checks
    if x.size == 0 or y.size == 0:
        raise ValueError("Vectors cannot be empty")

    if x.shape != y.shape:
        raise ValueError("Vectors must have the same shape")

    return float(np.sum(np.abs(x - y)))
    
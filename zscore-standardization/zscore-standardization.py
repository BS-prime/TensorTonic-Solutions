import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    # 0. convert to numpy array
    x = np.asarray(X, dtype=np.float64)

    # 1. perform checks
    if x.size == 0:
        raise ValueError("X can't be empty")

    # 2. calculate z-score
    mean = np.mean(x, axis = axis, keepdims = True)
    std = np.std(x, axis = axis, keepdims = True)

    return (x - mean)/(std + eps)
    
import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    y_pred = np.asarray(y_pred, dtype = np.float64)
    y_true = np.asarray(y_true, dtype = np.float64)

    # 0. validate the input
    if y_pred.size == 0 or y_true.size == 0:
        raise ValueError("inputs could not be empty")
    if y_pred.shape != y_true.shape:
        raise ValueError("inputs has be of same shape")
    if y_pred.ndim > 1 or y_true.ndim > 1:
        raise ValueError("dimension could not be more than 1")

    # 1. handle edge cases
    if np.array_equal(y_pred, y_true):
        return 0.0

    # 2. calculate sum of squared error
    return np.mean((y_true - y_pred)**2)
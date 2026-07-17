import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    # 0. convert to NumPy array
    y = np.asarray(y_true, dtype=np.float64)
    yhat = np.asarray(y_pred, dtype=np.float64)

    # 1. Checking cases
    if y.ndim != 1 or yhat.ndim != 1:
        raise ValueError("array dimension must be 1")
        
    if y.size == 0 or yhat.size == 0:
        raise ValueError("array can't be empty")
        
    if y.shape != yhat.shape:
        raise ValueError("both the arrays are not equal")

    # 2. calculate r2_score
    mean = np.mean(y)

    ss_res = np.sum((y - yhat)**2)
    ss_total = np.sum((y - mean)**2)

    # 3. handle edge cases
    if ss_total == 0:
        return 1.0 if np.array_equal(y, yhat) else 0.0

    return 1 - (ss_res/ss_total)
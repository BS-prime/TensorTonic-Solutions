import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    
    # 0. validate inputs
    if y_pred.size == 0 or y_true.size==0:
        raise ValueError("inputs can't be empty")
    if y_pred.shape != y_true.shape:
        raise ValueError("both input are not the same shape")
    if y_pred.ndim > 1 or y_true.ndim > 1:
        raise ValueError("inputs could not be more than 1")

    # 1. calculate error
    error = np.abs(y_true - y_pred)

    # 2. small error
    se =  0.5*error**2

    # 3. large error
    le = delta * error - 0.5*delta**2

    # implement huber loss
    return np.mean(np.where(error <= delta, se, le))
        
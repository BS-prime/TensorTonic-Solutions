import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # 0. convert to array
    x = np.asarray(x)

    # 1. perform checks
    if x.size == 0:
        raise ValueError("x is empty")

    # 2. leaky relu
    return np.where(x >= 0, x, alpha*x)
import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.asarray(x, dtype=np.float64)
    return 1 / (1 + np.exp(-x))
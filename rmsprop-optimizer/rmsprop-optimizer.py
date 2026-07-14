import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    w = np.asarray(w, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)

    # 1. running average of gradient
    s = beta * s + (1-beta)*g**2

    # 2. update the parameters
    w = w - (lr /np.sqrt(s + eps)) * g

    return w , s
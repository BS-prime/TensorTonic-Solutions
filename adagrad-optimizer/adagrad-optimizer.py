import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    # 0. convert to numpy array
    w = np.asarray(w, dtype = np.float64)
    g = np.asarray(g, dtype = np.float64)
    G = np.asarray(G, dtype = np.float64)

    # 1. accumulate squared gradient
    G_t = G + (g**2)

    # 2. update the parameter
    w_t = w - (lr / (np.sqrt(G_t + eps))) * g

    return w_t, G_t
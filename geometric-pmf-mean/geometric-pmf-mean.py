import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """

    k = np.asarray(k)
    
    # 0. validate param
    if (k < 1).any():
        raise ValueError("Trails could not less zero")

    if not 0 < p <= 1:
        raise ValueError("p should be greater than 0 and less than or equal to 1")

    # 1. calculate pmf
    pmf = (1-p)**(k-1)*p

    # 2. calculate mean
    mean = 1 / p * 1.0

    return pmf, mean
import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    # 1. convert to array
    x = np.asarray(x, dtype=np.int64)

    if not np.all((x == 0) | (x == 1)):
        raise ValueError("x must contain only 0 and 1.")

    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1.")
        
    # 2. calculate pmf
    pmf = np.where(x==1, p, 1-p)

    # 3. mean
    mean = p

    # 4. variance 
    var = mean * (1 - mean)

    return pmf, mean, var
    
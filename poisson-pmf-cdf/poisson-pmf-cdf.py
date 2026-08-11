import numpy as np
from scipy.special import factorial

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """

    # 0. validate params
    if lam < 0:
        raise ValueError("lamda must be non-negative")

    if k < 0 or int(k) != k:
        raise ValueError("k must be non-negative interger")
        
    k = int(k)
    
    # 1. calculate pmf
    pmf = (lam**k * np.exp(-lam)) / factorial(k)

    # 2. calculate pmf
    cdf = 0.0
    
    for i in range(k+1):
        cdf += (lam**i * np.exp(-lam)) / factorial(i)

    return pmf, cdf
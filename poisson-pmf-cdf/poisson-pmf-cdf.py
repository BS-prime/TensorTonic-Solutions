import numpy as np
from scipy.special import factorial

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # 1. calculate pmf
    pmf = (lam**k * np.exp(-lam)) / factorial(k)

    # 2. calculate pmf
    cdf = 0
    for i in range(k+1):
        cdf += (lam**i * np.exp(-lam)) / factorial(i)

    return pmf, cdf
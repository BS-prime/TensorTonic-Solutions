import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """

    # 0. parameter validation
    if not 0 <= p <= 1:
        raise ValueError("Probability between 0 and 1")
    if not 0 <= k <= n:
        raise ValueError("Number of successes could not be more than total number trials")
        
    # 1. calculate pmf
    pmf = comb(n, k) * (p**k) * ((1-p)**(n-k))

    # 2. calculate cdf
    cdf = 0
    for i in range(k+1):
        cdf += comb(n, i) * (p**i) * ((1-p)**(n-i))

    return pmf, cdf
    
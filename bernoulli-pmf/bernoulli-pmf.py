import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    # 0. convert to numpy array
    x = np.asarray(x, dtype=np.int64)

    # 1. input validation
    if not 0 <= p <= 1:
        raise ValueError("probability should be between 0 and 1")
    if x.size == 0:
        raise ValueError("inputs can't be empty")

    if x.ndim > 1:
        raise ValueError("Both parameter should of same dimension")

    # 2. calculate PMF
    pmf = np.where(
        x == 1,
        p,
        1-p
    )

    # 3. calculate mean and varience
    mean = p
    var = p * (1-p)

    return pmf, mean, var
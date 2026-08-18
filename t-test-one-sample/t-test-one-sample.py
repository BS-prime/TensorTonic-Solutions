import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    # 0. convert to numpy array
    x = np.asarray(x, dtype=np.float64)

    # 1. validate parameters
    if x.size < 2 or mu0 is None:
        raise ValueError("x must contain at least 2 observations and mu0 cannot be None")

    if x.ndim != 1:
        raise ValueError("x could not be more than 1 dimensional")

    # 2. calculate sample mean and sample size
    s_mean = np.mean(x)
    s_n = len(x)

    # 3. calculate sample standard deviation using Bessel's correction
    s_std = np.sqrt(np.sum((x - s_mean) ** 2) / (s_n - 1))

    # 4. calculate one-sample t-statistic
    return (s_mean - mu0) / (s_std / np.sqrt(s_n))
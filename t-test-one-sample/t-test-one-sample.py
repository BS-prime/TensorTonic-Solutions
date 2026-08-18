import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    # 0. convert to numpy array
    x = np.asarray(x, dtype=np.float64)

    # 1. validate the params
    if x.size < 1 or mu0 is None:
        raise ValueError("parameters can't be empty")

    if x.ndim != 1:
        raise ValueError("x could not be more than 1 dimensional")

    # 2. calculate mean, and total length of the sample
    s_mean = np.mean(x)
    s_n = len(x)

    # 3. calculate sample mean Bessel correction
    s_std = np.sqrt(1 / (s_n - 1) * (np.sum((x - s_mean) ** 2)))

    # 4. calculate t_test_one_sample
    return (s_mean - mu0) / (s_std / np.sqrt(s_n))
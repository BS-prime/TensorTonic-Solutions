import numpy as np


def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # 0. convert to array
    x = np.asarray(x)

    # 1. validate params
    if x.ndim != 1:
        raise ValueError("x must be 1-dimensional")

    if x.size < 2:
        raise ValueError("x should have at least 2 elements")

    # 2. calculate the mean
    mean = np.mean(x)

    # 3. calculate sample variance using Bessel's correction
    var = np.sum((x - mean) ** 2) / (x.size - 1)

    # 4. calculate standard deviation
    std = np.sqrt(var)

    return var, std
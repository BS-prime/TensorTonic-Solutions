import numpy as np


def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # 0. convert to array
    x = np.asarray(x)

    # 1. validate params
    if x.size == 0:
        raise ValueError("x can't be empty")

    # 2. calculate the mean
    mean = np.mean(x)

    # 3. substract each data point from from the mean
    var = np.sum((x - mean)**2) / (len(x) - 1)

    # 4. calculate standard deviation
    std = np.sqrt(var)

    return var, std
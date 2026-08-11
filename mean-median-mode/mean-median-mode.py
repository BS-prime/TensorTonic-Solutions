import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    x = np.asarray(x)
    # 0. input validation
    if x.size == 0:
        raise ValueError("x can't be empty")

    if x.ndim > 1:
        raise ValueError("x can't more than 1 dimension")

    # 1. calculate mean and median
    mean = np.mean(x)
    median = np.median(x)

    # 2. calculate mode
    mode = Counter(x).most_common(1)[0][0]

    return float(mean), float(median), float(mode)
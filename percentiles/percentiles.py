import numpy as np


def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    # 0. convert to array 
    x = np.asarray(x)
    q = np.asarray(q)

    # 1. params validation
    if not np.all((q >= 0) & (q <= 100)):
        raise ValueError("q should be between 0 and 100")

    if x.size == 0 or q.size == 0:
        raise ValueError("params could not be emptpy")

    # 2. calculate percentiles
    pctile = np.percentile(x, q, method = "linear")

    return pctile

if __name__ == "__main__":
    print(percentiles(x=[1,2,3,4], q=[25, 50, 60]))
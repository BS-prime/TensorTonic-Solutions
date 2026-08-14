import numpy as np


def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """

    # 0. convert to numpy array
    C = np.asarray(C, dtype=float)

    # 1. validate parameters
    if C.size == 0:
        raise ValueError("C could not be empty")

    if C.ndim != 2:
        raise ValueError("C is not a 2D matrix")

    if np.any(C < 0):
        raise ValueError("Frequencies cannot be negative")

    total = C.sum()

    if total <= 0:
        raise ValueError("Total frequency must be greater than 0")

    # 2. calculate expected frequencies
    row_totals = C.sum(axis=1, keepdims=True)
    col_totals = C.sum(axis=0, keepdims=True)

    exp_freq = row_totals * col_totals / total

    # 3. calculate chi-square test statistic
    chi2_stat = np.sum((C - exp_freq) ** 2 / exp_freq)

    return chi2_stat, exp_freq 
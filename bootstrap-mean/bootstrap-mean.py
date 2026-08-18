import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    # 0. convert to numpy array
    x = np.asarray(x, dtype=np.float64)

    # 1. validate inputs
    if x.ndim != 1:
        raise ValueError("x should be 1-D")

    if x.size < 1:
        raise ValueError("x has to have at least two elements")

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")

    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1")

    # 2. initialize RNG
    n = len(x)

    if rng is None:
        rng = np.random.default_rng()

    # 3. generate bootstrap means
    boot_means = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        boot_means[i] = x[indices].mean()

    # 4. calculate percentile confidence interval
    alpha = (1 - ci) / 2

    lower = np.quantile(boot_means, alpha)
    upper = np.quantile(boot_means, 1 - alpha)

    return boot_means, lower, upper
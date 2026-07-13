import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    if len(x) != len(p):
        raise ValueError("x and y must have the same length")
    if np.any(p<0) or np.any(p>1):
        raise ValueError("Invalid probabilities")
    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("Invalid probabilities")
    return float(np.dot(x,p))
    

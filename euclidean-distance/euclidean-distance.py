import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # 0. convert to array
    x = np.asarray(x)
    y = np.asarray(y)

    # 1. perform validation
    if x.size == 0 or y.size == 0:
        raise ValueError("Empty arrays found")
    if x.shape != y.shape:
        raise ValueError("Both arrays are not same length")

    # 2. calculate the distance
    return np.sqrt(np.sum((x - y)**2))
    
    
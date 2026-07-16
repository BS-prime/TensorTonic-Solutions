import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # 0. Convert to NumPy array
    a = np.asarray(a)
    b = np.asarray(b)

    # 1. Perform checks
    if a.shape != b.shape:
        raise ValueError("Both arrays have to be the same length")
    if a.size == 0 and b.size == 0:
        raise ValueError("arrays can't be empty")
    if a.any() == 0 and b.any():
        return 0.0

    # 2. Perform dot product
    dp = np.dot(a, b)

    # 3. Perform Euclidean norm
    norm = np.linalg.norm(a) * np.linalg.norm(b)

    # 4. Calculate cosine similarity
    return float(dp / norm)
    
    
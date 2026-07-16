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

    # 2. Perform dot product
    dp = np.dot(a, b)

    # 3. Perform Euclidean norm
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # 4. Handle zero vectors
    if norm_a == 0 or norm_b == 0:
        return 0.0  

    # 5. Calculate cosine similarity
    return float(dp / (norm_a * norm_b))
    
    
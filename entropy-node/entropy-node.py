import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # handle empty array
    if len(y) == 0:
        return 0.0
        
    # 0. convert to array
    y = np.asarray(y)
        
    # 1. get class frequency    
    _, freq = np.unique(y, return_counts = True)

    # 2. calculate probability of each class
    p_i = freq/np.sum(freq)

    # 3. calculate entropy
    log_p = np.where(p_i > 0, np.log2(p_i), 0.0) # simply because of the docstring by tensortonic; otherwise, np.unique doesn't let zero-count classes pass through.
    return - np.sum(p_i * log_p) 
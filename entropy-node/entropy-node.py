import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # 0. convert to array
    y = np.asarray(y, dtype = np.int64)

    # 1. get class frequency
    if len(y) == 0:
        return 0.0
        
    classes, freq = np.unique(y, return_counts = True)

    # 2. handle single class
    if len(classes) == 1:
        return 0.0

    # 3. calculate entropy
    entropy = 0.0
    for i in range(len(classes)):
        p_i = freq[i] / len(y)
        entropy += float(p_i * - (np.log2(p_i)))

    return entropy
        
    
    
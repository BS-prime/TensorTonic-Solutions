import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # 1. store word with index
    vocab_idx = {word: i for i, word in enumerate(vocab)}

    # 2. create a zero vector of vocab length
    vec = np.zeros(len(vocab), dtype=int)

    # 3. replace
    for token in tokens:
        if token in vocab_idx:
            vec[vocab_idx[token]] += 1
    return vec
    
    
    
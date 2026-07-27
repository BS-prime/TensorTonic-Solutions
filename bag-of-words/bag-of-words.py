import numpy as np


def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # 1. mapping each word in vocab to it index
    vocab_lookup = {word: idx for idx, word in enumerate(vocab)}

    # 2. fetch the indices of each word from dictionary
    indices = [vocab_lookup[w] for w in tokens if w in vocab_lookup]

    # 3. count the frequency
    vector = np.bincount(indices, minlength=len(vocab))

    return vector.astype(int)
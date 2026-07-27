import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # 0. convert to numpy array
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    # 1. input validation
    if y_train.size == 0 or X_test.size == 0:
        raise ValueError("Empty inputs are not allowed")

    if y_train.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional.")

    # 2. calculate frequencies
    values, count = np.unique(y_train, return_counts=True)

    # 3. class with highest count
    majority_class =  values[np.argmax(count)]

    # 4. final output
    return np.full(len(X_test), majority_class)
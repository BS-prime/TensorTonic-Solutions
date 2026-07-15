import  numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    if len(y_pred) != len(y_true):
        raise ValueError("y_true and y_pred are not of same length")
    
    # 0. convert to array
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    # 1. element-wise matching
    classes = np.union1d(y_true, y_pred)

    # 2. calculate fp, fn, tp
    tp = fp = fn = 0

    for c in classes:
        tp += np.sum((y_true == c) & (y_pred == c))
        fp += np.sum((y_true != c) & (y_pred == c))
        fn += np.sum((y_true == c) & (y_pred != c))

    # 3. calculate f1-micro
    f1_micro = 2 * tp /(2 * tp + fp + fn)
    
    return f1_micro
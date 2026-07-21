import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    
    w = np.asarray(w)
    v = np.asarray(v)
    grad = np.asarray(grad)
    
    # 0. perform checks
    if w.size == 0 or v.size == 0 or grad.size == 0:
        raise ValueError("arrays can't be empty")

    if w.ndim > 1 or v.ndim > 1 or grad.ndim > 1:
        raise ValueError("dimension of the arrays could not be more than 1")

    if w.shape != v.shape and v.shape != grad.shape:
        raise ValueError("both arrays are not same shape")
        
    # 1. update the velocity
    v_t = momentum*v + lr*grad

    # 2. update the parameter
    w_t = w - v_t

    return w_t, v_t
    
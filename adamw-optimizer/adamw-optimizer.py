import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    """
    # 0. convert parameters to numpy array
    w = np.asarray(w)
    m = np.asarray(m)
    v = np.asarray(v)
    grad = np.asarray(grad)

    # 1. validate parameters
    if w.size == 0 or m.size == 0 or v.size == 0 or grad.size == 0:
        raise ValueError("parameters can't be empty")

    if w.ndim > 1 or m.ndim > 1 or v.ndim > 1 or grad.ndim > 1:
        raise ValueError("parmeters can't be more than 1d")

    if not (w.shape == m.shape == v.shape == grad.shape):
        raise ValueError("All inputs must have the same shape.")
    
    # 2. momentum update
    m_t = beta1*m + (1 - beta1)*grad

    # 3. velocity update
    v_t = beta2*v + (1 - beta2)*(grad**2)

    # # 4. implementing bias correction
    # m_bc = m_t / (1 - beta1**t)
    # v_bc = v_t / (1 - beta2**t)
    
    # 5. final AdamW implementation
    w_t = (
        w 
        - lr * weight_decay * w 
        - lr * (m_t / (np.sqrt(v_t) + eps))
    )

    return w_t, m_t, v_t
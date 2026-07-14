import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    m = np.asarray(m, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    param = np.asarray(param, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    
    # 1. calculate the momentum
    m_new = beta1 * m + (1 - beta1) * grad

    # 2. adaptive learning rate
    v_new = beta2 * v + (1 - beta2) * grad ** 2

    # 3. calculate bias 
    m_hat = m_new / (1 - beta1**t)
    v_hat = v_new / (1 - beta2**t)

    # 4. update the parameters
    param_new = param - lr * (m_hat / (np.sqrt(v_hat) + eps))

    return param_new, m_new, v_new
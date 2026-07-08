import numpy as np

def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    
    for _ in range(steps):
        gradient = 2 * a * x0 + b
        x0 = x0 - lr * gradient
    return x0
    
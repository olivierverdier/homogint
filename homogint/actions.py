import numpy as np

def left_multiplication(g: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Multiplication action of a group and a vector.
    """
    return np.dot(g, x)

def trans_adjoint(g: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.dot(np.dot(g,x),g.T)

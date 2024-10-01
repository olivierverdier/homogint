from typing import TypeAlias
from numpy.typing import NDArray
import numpy as np

Vector : TypeAlias = NDArray[np.float64]

def left_multiplication(g: Vector, x: Vector) -> Vector:
    """
    Multiplication action of a group and a vector.
    """
    return np.asarray(np.dot(g, x))

def trans_adjoint(g: Vector, x: Vector) -> Vector:
    return np.asarray(np.dot(np.dot(g,x),g.T))

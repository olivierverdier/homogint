from typing_extensions import TypeAlias
import numpy as np
from numpy.typing import NDArray

from padexp import Exponential  # type: ignore

Vector: TypeAlias = NDArray[np.float64]

Exp = Exponential(order=16)

def exponential(xi: Vector) -> Vector:
    """
    Maps a Lie algebra vector to a group element.
    """
    return np.asarray(Exp(xi)[0])

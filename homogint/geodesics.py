from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Callable, TypeAlias

import numpy as np
import scipy.linalg  # type: ignore

from .actions import left_multiplication

Vector: TypeAlias = NDArray[np.float64]


@dataclass
class Geodesic:
    action: Callable[[Vector, Vector], Vector] = left_multiplication
    movement: Callable[[Vector], Vector] = scipy.linalg.expm

    def __call__(self, x: Vector, ξ: Vector) -> Vector:
        return self.action(self.movement(ξ), x)

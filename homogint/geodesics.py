from dataclasses import dataclass
from typing import Callable

import scipy.linalg

from .actions import left_multiplication


@dataclass
class Geodesic:
    action: Callable = left_multiplication
    movement: Callable = scipy.linalg.expm

    def __call__(self, x, ξ):
        return self.action(self.movement(ξ), x)

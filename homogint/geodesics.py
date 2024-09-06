from dataclasses import dataclass
from typing import Callable

from .actions import left_multiplication
from .movements import exponential


@dataclass
class Geodesic:
    action: Callable = left_multiplication
    movement: Callable = exponential

    def __call__(self, x, ξ):
        return self.action(self.movement(ξ), x)

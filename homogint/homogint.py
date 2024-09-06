import numpy as np
from typing import Callable
from numpy.typing import NDArray

from .skeletons import Skeleton
from .geodesics import Geodesic


from dataclasses import dataclass, field

@dataclass
class Integrator:
    method: Skeleton
    geodesic: Geodesic = field(default_factory=Geodesic)

    def __post_init__(self):
        self.nb_stages = len(self.method.edges) + 1

    def compute_vectors(self, movement_field: Callable, stages: list) -> np.ndarray:
        """
        Compute the Lie algebra elements for the stages.
        """
        return np.array([movement_field(stage) for stage in stages])

    def get_iterate(self, movement_field: Callable) -> Callable:
        def evol(stages):
            new_stages = stages.copy()
            for (i,j, transition) in self.method.edges:
                # inefficient as a) only some vectors are needed b) recomputed for each edge
                vects = self.compute_vectors(movement_field, new_stages)
                # the order of the edges matters; the goal is that explicit method need only one iteration
                new_stages[i] = self.geodesic(new_stages[j], transition(vects))
            return new_stages
        return evol

    @classmethod
    def fix(self, iterate: Callable[[np.ndarray], np.ndarray], z: NDArray) -> tuple[NDArray, int]:
        """
        Find a fixed point to the iterating function `iterate`.
        """
        for i in range(30):
            new_z = iterate(z)
            if np.allclose(z, new_z, atol=1e-10, rtol=1e-16):
                break
            z = new_z
        else:
            raise np.exceptions.TooHardError("No convergence after {} steps".format(i))
        return z, i

    def step(self, movement_field: Callable, x0: np.ndarray) -> np.ndarray:
        iterate = self.get_iterate(movement_field)
        z0 = np.array([x0]*self.nb_stages)  # initial guess
        z, i = self.fix(iterate, z0)
        return z[-1]

RungeKutta = Integrator

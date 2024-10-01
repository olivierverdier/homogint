import numpy as np
from typing import Callable
from typing_extensions import TypeAlias
from numpy.typing import NDArray

from .skeletons import Skeleton
from .geodesics import Geodesic


from dataclasses import dataclass, field

Vector: TypeAlias = NDArray[np.float64]
Matrix: TypeAlias = NDArray[np.float64]


@dataclass
class Integrator:
    method: Skeleton
    geodesic: Geodesic = field(default_factory=Geodesic)

    def __post_init__(self) -> None:
        self.nb_stages = len(self.method.edges) + 1

    def compute_vectors(self, movement_field: Callable[[Vector], Vector], stages: Matrix) -> Matrix:
        """
        Compute the Lie algebra elements for the stages.
        """
        return np.array([movement_field(stage) for stage in stages])

    def get_iterate(self, movement_field: Callable[[Vector], Vector]) -> Callable[[Vector], Vector]:
        def evol(stages: Vector) -> Vector:
            new_stages = stages.copy()
            for (i,j, transition) in self.method.edges:
                # inefficient as a) only some vectors are needed b) recomputed for each edge
                vects = self.compute_vectors(movement_field, new_stages)
                # the order of the edges matters; the goal is that explicit method need only one iteration
                new_stages[i] = self.geodesic(new_stages[j], transition(vects))
            return new_stages
        return evol

    @classmethod
    def fix(self, iterate: Callable[[Vector], Vector], z: Vector) -> tuple[Vector, int]:
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

    def step(self, movement_field: Callable[[Vector], Vector], x0: Vector) -> Vector:
        iterate = self.get_iterate(movement_field)
        z0 = np.array([x0]*self.nb_stages)  # initial guess
        z, i = self.fix(iterate, z0)
        return np.asarray(z[-1])

RungeKutta = Integrator

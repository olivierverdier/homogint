from typing import Callable, TypeAlias
from numpy.typing import NDArray
from abc import ABC, abstractmethod

import numpy as np

Vector: TypeAlias = NDArray[np.float64]
Matrix: TypeAlias = NDArray[np.float64]
Transition: TypeAlias = Callable[[Matrix], Vector]
Edges: TypeAlias = list[tuple[int, int, Transition]]



def commutator(x1: Vector, x2: Vector) -> Vector:
    return np.asarray(np.dot(x1, x2) - np.dot(x2, x1))

class Skeleton(ABC):
    @property
    @abstractmethod
    def edges(self) -> Edges:  # pragma: no cover
        pass

class ForwardEuler(Skeleton):
    edges = [(1,0, lambda vecs:vecs[0])]

class BackwardEuler(Skeleton):
    edges = [(1,0, lambda vecs:vecs[1])]

class MidPoint(Skeleton):
    edges = [(1,0, lambda vecs:vecs[1]/2),
             (2,1, lambda vecs: vecs[1]/2)]

class Trapezoidal(Skeleton):
    edges = [(1,0, lambda vecs:(vecs[0]+vecs[1])/2.)]

class CommutatorFree4(Skeleton):
    def t10(self, F: Matrix) -> Vector:
        return np.asarray(F[0]/2)
    def t20(self, F: Matrix) -> Vector:
        return np.asarray(F[1]/2)
    def t31(self, F: Matrix) -> Vector:
        return np.asarray(F[2] - F[0]/2)
    def t40(self, F: Matrix) -> Vector:
        return np.asarray((3*F[0] + 2*(F[1]+F[2]) -F[3])/12)
    def t54(self, F: Matrix) -> Vector:
        return np.asarray((-F[0] + 2*(F[1]+F[2]) +3*F[3])/12)

    @property
    def edges(self) -> Edges:
        return [
            (1,0, self.t10),
            (2,0, self.t20),
            (3,1, self.t31),
            (4,0, self.t40),
            (5,4, self.t54),
        ]


class RKMK4(Skeleton):
    """
    Taken from http://www.math.ntnu.no/num/expint/talks/owren04innsbruck.pdf
    or http://arxiv.org/pdf/1207.0069.pdf
    """
    def t10(self,F: Matrix) -> Vector:
        return np.asarray(F[0]/2)
    def t20(self, F: Matrix) -> Vector:
        return np.asarray(F[1]/2 - 1./8*commutator(F[0],F[1]))
    def t30(self, F: Matrix) -> Vector:
        return np.asarray(F[2])
    def t40(self, F: Matrix) -> Vector:
        return np.asarray((F[0] + 2*(F[1]+F[2]) + F[3])/6. - 1./12*commutator(F[0],F[3]))

    @property
    def edges(self) -> Edges:
        return [
            (1,0, self.t10),
            (2,0, self.t20),
            (3,0, self.t30),
            (4,0, self.t40),
        ]

class RKMK3(Skeleton):
    """
    From McLachlan, Quispel, Integrating ODEs
    """
    def t10(self, F: Matrix) -> Vector:
        return np.asarray(F[0]/2)
    def t20(self, F: Matrix) -> Vector:
        return np.asarray(-F[0] + 2*F[1])
    def t30(self, F: Matrix) -> Vector:
        tmp = (F[0] + 4*F[1] + F[2])/6
        return np.asarray(tmp + commutator(tmp,F[0])/6)
    # equivalent to (F[0] + 4*F[1] + F[2])/6 + commutator(4*F[1]+F[2],F[0])/36
    @property
    def edges(self) -> Edges:
        return [
            (1,0, self.t10),
            (2,0, self.t20),
            (3,0, self.t30),
        ]

class CrouchGrossman3(Skeleton):
    """
    From Hairer, Lubich, Wanner 2006.
    """
    def t10(self,F: Matrix) -> Vector:
        return np.asarray(3/4*F[0])
    def t20(self,F: Matrix) -> Vector:
        return np.asarray(119/216*F[0])
    def t32(self,F: Matrix) -> Vector:
        return np.asarray(17/108*F[1])
    def t40(self,F: Matrix) -> Vector:
        return np.asarray(13/51*F[0])
    def t54(self,F: Matrix) -> Vector:
        return np.asarray(-2/3*F[1])
    def t65(self,F: Matrix) -> Vector:
        return np.asarray(24/17*F[3])

    @property
    def edges(self) -> Edges:
        return [
            (1,0, self.t10),
            (2,0, self.t20),
            (3,2, self.t32),
            (4,0, self.t40),
            (5,4, self.t54),
            (6,5, self.t65),
        ]

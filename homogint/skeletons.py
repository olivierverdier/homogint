#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np

from padexp import Exponential

Exp = Exponential(order=16)

def exponential(xi):
    return Exp(xi)[0]

def commutator(x1,x2):
    return np.dot(x1,x2) - np.dot(x2,x1)

class ExponentialSkeleton(object):
    def movement(self, xi):
        return exponential(xi)

class ForwardEuler(ExponentialSkeleton):
    edges = [(1,0, lambda vecs:vecs[0])]

class BackwardEuler(ExponentialSkeleton):
    edges = [(1,0, lambda vecs:vecs[1])]

class MidPoint(ExponentialSkeleton):
    edges = [(1,0, lambda vecs:vecs[1]/2),
             (2,1, lambda vecs: vecs[1]/2)]

class Trapezoidal(ExponentialSkeleton):
    edges = [(1,0, lambda vecs:(vecs[0]+vecs[1])/2.)]

class CommutatorFree4(ExponentialSkeleton):
    def t10(self, F):
        return F[0]/2
    def t20(self, F):
        return F[1]/2
    def t31(self, F):
        return F[2] - F[0]/2
    def t40(self, F):
        return (3*F[0] + 2*(F[1]+F[2]) -F[3])/12
    def t54(self, F):
        return (-F[0] + 2*(F[1]+F[2]) +3*F[3])/12

    @property
    def edges(self):
        return [
            (1,0, self.t10),
            (2,0, self.t20),
            (3,1, self.t31),
            (4,0, self.t40),
            (5,4, self.t54),
        ]


class RKMK4(ExponentialSkeleton):
    """
    Taken from http://www.math.ntnu.no/num/expint/talks/owren04innsbruck.pdf
    or http://arxiv.org/pdf/1207.0069.pdf
    """
    def t10(self,F):
        return F[0]/2
    def t20(self, F):
        return F[1]/2 - 1./8*commutator(F[0],F[1])
    def t30(self, F):
        return F[2]
    def t40(self, F):
        return (F[0] + 2*(F[1]+F[2]) + F[3])/6. - 1./12*commutator(F[0],F[3])

    @property
    def edges(self):
        return [
            (1,0, self.t10),
            (2,0, self.t20),
            (3,0, self.t30),
            (4,0, self.t40),
        ]

class RKMK3(ExponentialSkeleton):
    """
    From McLachlan, Quispel, Integrating ODEs
    """
    def t10(self, F):
        return F[0]/2
    def t20(self, F):
        return -F[0] + 2*F[1]
    def t30(self, F):
        tmp = (F[0] + 4*F[1] + F[2])/6
        return tmp + commutator(tmp,F[0])/6
    # equivalent to (F[0] + 4*F[1] + F[2])/6 + commutator(4*F[1]+F[2],F[0])/36
    @property
    def edges(self):
        return [
            (1,0, self.t10),
            (2,0, self.t20),
            (3,0, self.t30),
        ]

class CrouchGrossman3(ExponentialSkeleton):
    """
    From Hairer, Lubich, Wanner 2006.
    """
    def t10(self,F):
        return 3/4*F[0]
    def t20(self,F):
        return 119/216*F[0]
    def t32(self,F):
        return 17/108*F[1]
    def t40(self,F):
        return 13/51*F[0]
    def t54(self,F):
        return -2/3*F[1]
    def t65(self,F):
        return 24/17*F[3]

    @property
    def edges(self):
        return [
            (1,0, self.t10),
            (2,0, self.t20),
            (3,2, self.t32),
            (4,0, self.t40),
            (5,4, self.t54),
            (6,5, self.t65),
        ]

#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np


def left_multiplication(g, x):
    """
    Multiplication action of a group and a vector.
    """
    return np.dot(g, x)

def trans_adjoint(g, x):
    return np.dot(np.dot(g,x),g.T)

class RungeKutta(object):

    def __init__(self, method):
        self.method = method
        self.movement = self.method.movement
        self.nb_stages = len(self.method.edges) + 1

    def compute_vectors(self, movement_field, stages):
        """
        Compute the Lie algebra elements for the stages.
        """
        return np.array([movement_field(stage) for stage in stages])

    def get_iterate(self, movement_field, action):
        def evol(stages):
            new_stages = stages.copy()
            for (i,j, transition) in self.method.edges:
                # inefficient as a) only some vectors are needed b) recomputed for each edge
                vects = self.compute_vectors(movement_field, new_stages)
                # the order of the edges matters; the goal is that explicit method need only one iteration
                new_stages[i] = action(self.movement(transition(vects)), new_stages[j])
            return new_stages
        return evol

    @classmethod
    def fix(self, iterate, z):
        """
        Find a fixed point to the iterating function `iterate`.
        """
        for i in range(30):
            new_z = iterate(z)
            if np.allclose(z, new_z, atol=1e-10, rtol=1e-16):
                break
            z = new_z
        else:
            raise Exception("No convergence after {} steps".format(i))
        return z, i

    def step(self, movement_field, x0, action=None):
        if action is None:
            action = left_multiplication
        iterate = self.get_iterate(movement_field, action)
        z0 = np.array([x0]*self.nb_stages) # initial guess
        z, i = self.fix(iterate, z0)
        return z[-1]


import unittest
import numpy.testing as npt

import numpy as np

from homogint import Integrator, time_step
import homogint.skeletons as sk
from homogint.actions import trans_adjoint
from homogint.geodesics import Geodesic

import numpy.linalg as nl

rng = np.random.default_rng(10)

def rotation_field(x):
    """
    Circular motion around NS axis.
    """
    J = np.zeros([3,3])
    J[0,1] = -1.
    J[1,0] = 1.
    return .1*J


inertia = np.array([1.,2.,3.])

def body_field(x):
    ix = inertia*x
    xi = np.zeros([3,3])
    xi[0,1] = ix[2]
    xi[0,2] = -ix[1]
    xi[1,2] = ix[0]
    xi -= xi.T
    return xi

def iso_field(P):
    sk = np.tril(P) - np.triu(P)  # skew symmetric
    return sk

def solve(vf,xs,stopping,action=None, maxit=10000,solver=None):
    "Simple solver with stopping condition. The list xs is modified **in place**."
    if solver is None:
        solver = Integrator(sk.RKMK4(), Geodesic(action))
    for i in range(maxit):
        if stopping(i,xs[-1]):
            break
        xs.append(solver.step(vf, xs[-1]))

class TestSphere(unittest.TestCase):
    def test_main(self):
        rk = Integrator(sk.RKMK4())
        x0 = rng.standard_normal(3).reshape(-1,1)
        x = x0.copy()
        for i in range(10):
            x = rk.step(rotation_field, x)
        npt.assert_allclose(x[-1], x0[-1], err_msg="rotation around NS axis")
        npt.assert_allclose(np.sum(np.square(x)), np.sum(np.square(x0)), err_msg="stay on the sphere")

    def test_Toda(self, size=5):
        """
        Numerical flow is isospectral.
        """
        rmat = rng.standard_normal((size, size))
        init = rmat + rmat.T
        Ps = [init]
        dt = .25
        solve(time_step(dt)(iso_field), Ps, lambda i,x: i>10/dt, action=trans_adjoint)
        eigenvalues = [np.sort(nl.eigvals(P)) for P in Ps]
        aeigenvalues = np.array(eigenvalues)
        deig = aeigenvalues - aeigenvalues[0]
        npt.assert_allclose(deig, 0, atol=1e-12, err_msg="numerical flow is isospectral")

    def test_no_convergence(self):
        """
        Convergence failure is caught with an exception.
        """
        rk = Integrator(sk.BackwardEuler(), Geodesic(trans_adjoint))
        x0 = np.array([1.,1.,1])/np.sqrt(3)
        with self.assertRaises(np.exceptions.TooHardError):
            rk.step(time_step(10.)(body_field), x0)


class HarnessOrder(object):
    scaling = 1
    def test_order(self):
        rk = Integrator(self.method)
        ks = [0,1,7]
        x0 = np.array([1.,1.,1])/np.sqrt(3)
        sols = []
        for k in ks:
            x =x0.copy()
            N = pow(2,k)
            for j in range(N):
                x = rk.step(time_step(self.scaling*1./N)(body_field), x)
            sols.append(x)
        exact = sols[-1]
        errors = np.log2(np.max(np.abs(sols[:-1]-exact), axis=1))
        computed_order = errors[0]-errors[1]
        self.assertGreater(computed_order, self.order)

class TestRKMK4(HarnessOrder, unittest.TestCase):
    method = sk.RKMK4()
    order = 4

class TestCG3(HarnessOrder, unittest.TestCase):
    method = sk.CrouchGrossman3()
    order = 3
    scaling = .1

class TestCF4(HarnessOrder, unittest.TestCase):
    method = sk.CommutatorFree4()
    order = 4

class TestRKMK3(HarnessOrder, unittest.TestCase):
    method = sk.RKMK3()
    order = 3
    scaling=.5

class TestForwardEuler(HarnessOrder, unittest.TestCase):
    method = sk.ForwardEuler()
    order = 1
    scaling = .01

class TestBackwardEuler(HarnessOrder, unittest.TestCase):
    method = sk.BackwardEuler()
    order = 1
    scaling = .01

class TestMidPoint(HarnessOrder, unittest.TestCase):
    method = sk.MidPoint()
    order = 2
    scaling = .01

class TestTrapezoidal(HarnessOrder, unittest.TestCase):
    method = sk.Trapezoidal()
    order = 2
    scaling = .1

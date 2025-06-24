# Optimization for Engineers - Dr.Johannes Hild
# Nonlinear test function
# Do not change this file

# Required files:
# < none >

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 0
    return matrnr


class noisyObjective:
    # Nonlinear function R**8 -> R with parameter p and noise
    # 8-dimensional nonlinear function mapping x ->  0.5*x.T @ A @ x  - b.T@x + p/(0.5*x.T @ A @ x + 1) + small noise;
    # with A being spd and b some vector. Is coercive. GMP should be [[1.02614],[0],[0],[0],[0],[0],[0],[0]].

    # Class parameter:
    # p: scalar in R (parameter space)

    # Input Definition:
    # x: vector in R**8 (domain space)

    # Output Definition:
    # objective: real number, evaluation of nonlinearObjective at x
    # setParameters(): sets p

    # Test cases:

    def __init__(self, p=1):
        self.p = p
        self.A = np.array(
            [[10, 3, 1, 0, 0, 0, 0, 0], [3, 10, 3, 1, 0, 0, 0, 0], [1, 3, 10, 3, 1, 0, 0, 0],
             [0, 1, 3, 10, 3, 1, 0, 0],
             [0, 0, 1, 3, 10, 3, 1, 0], [0, 0, 0, 1, 3, 10, 3, 1], [0, 0, 0, 0, 1, 3, 10, 3],
             [0, 0, 0, 0, 0, 1, 3, 10]])
        self.b = np.array([[10], [3], [1], [0], [0], [0], [0], [0]])

    def objective(self, x: np.array):
        noise = 0.001 * np.sin(6.28318 * np.random.random())
        tau = 0.5 * x.T @ self.A @ x + 1
        value = 0.5 * x.T @ self.A @ x - self.b.T @ x + self.p / tau + noise
        return value

    def setParameters(self, p):
        self.p = p



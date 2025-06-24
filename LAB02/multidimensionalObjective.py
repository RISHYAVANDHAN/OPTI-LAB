# Optimization for Engineers - Dr.Johannes Hild
# Nonlinear test function
# Do not change this file

# Required files:
# < none >

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23356687
    return matrnr


class multidimensionalObjective:
    # Nonlinear function R**8 -> R with parameter p
    # 8-dimensional nonlinear function mapping x ->  0.5*x.T @ A @ x  - b.T@x + p/(0.5*x.T @ A2 @ x + 1);
    # with A being spd and b some vector. Is coercive but probably not convex. LMP at [[1.02614],[0],[0],[0],[0],[0],[0],[0]].

    # Class parameter:
    # p: vector in R (parameter space)

    # Input Definition:
    # x: vector in R**8 (domain space)

    # Output Definition:
    # objective: real number, evaluation of nonlinearObjective at x
    # gradient: real column vector in R**8, evaluation of the gradient with respect to x at x
    # hessian: real 8x8 matrix, evaluation of the hessian with respect to x at x
    # setParameters(): sets p
    # parameterGradient(): vector in R**1, evaluation of gradient wrt p

    # Test cases:

    def __init__(self, p=1):
        self.p = p # parameter of the function
        self.A = np.array(
            [[10, 3, 1, 0, 0, 0, 0, 0], [3, 10, 3, 1, 0, 0, 0, 0], [1, 3, 10, 3, 1, 0, 0, 0],
             [0, 1, 3, 10, 3, 1, 0, 0],
             [0, 0, 1, 3, 10, 3, 1, 0], [0, 0, 0, 1, 3, 10, 3, 1], [0, 0, 0, 0, 1, 3, 10, 3],
             [0, 0, 0, 0, 0, 1, 3, 10]]) # matrix of quadratic part
        self.b = np.array([[10], [3], [1], [0], [0], [0], [0], [0]]) # linear part 

    def objective(self, x: np.array):
        tau = 0.5 * x.T @ self.A @ x + 1 # store denominator
        value = 0.5 * x.T @ self.A @ x - self.b.T @ x + self.p / tau # compose function from parts
        return value

    def gradient(self, x: np.array):
        tau = 0.5 * x.T @ self.A @ x + 1 # store denominator
        g = self.A @ x - self.b - self.p / (tau ** 2) * (self.A @ x) # gradient via chain rule
        return g

    def hessian(self, x: np.array):
        tau = 0.5 * x.T @ self.A @ x + 1 # store denominator
        h = self.A - self.p / (tau ** 2) * self.A + (2*self.p) / (tau ** 3) * (self.A @ x) @ (self.A @ x).T # hessian via chain rule
        return h

    def setParameters(self, p):
        self.p = p # set parameter

    def parameterGradient(self, x: np.array):
        tau = 0.5 * x.T @ self.A @ x + 1 # store denominator
        value = 1 / tau # store inverse denominator
        myGradP = np.array([[value]], dtype=float) # store parameter gradient as 1x1 vector

        return myGradP

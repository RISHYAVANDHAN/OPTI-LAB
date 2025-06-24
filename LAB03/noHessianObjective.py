# Optimization for Engineers - Dr.Johannes Hild
# Nonlinear test function without Hessian
# Do not change this file

# Required files:
# < none >


import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23356687
    return matrnr


class noHessianObjective:
    # Nonlinear function R**2 -> R
    # 2-dimensional nonlinear function mapping x -> -0.03/((x(0) +0.25)**2 +(x(1) -0.2)**2 +0.03) -0.1/((x(0) -0.25)**2 +(x(1)+ 0.2)**2 +0.04) + 0.1/(x(0)**2 +x(1)**2 +0.05) +1 +x(0)**2 +x(1)**2 + 1;
    # Has a local maximizing point at approx [[-0.0158], [0.0126]], and a local minimizing point at approx [-0.265;0.212] and a global minimizing point at approx [[0.261], [-0.209]]

    # Input Definition:
    # x: vector in R**2 (domain space)

    # Output Definition:
    # objective: real number, evaluation of nonlinearObjective at x
    # gradient: real column vector in R**2, evaluation of the gradient with respect to x at x

    # Test cases:

    # myObjective = noHessianObjective.objective(np.array([[-0.015793], [0.012647]], dtype=float))
    # should return
    # myObjective = 3.0925

    # myGradient = noHessianObjective.gradient(np.array([[-0.015793], [0.012647]], dtype=float))
    # should return
    # myGradient close to [[0],[0]]

    @staticmethod
    def objective(x: np.array):
        x1 = x[0,0] # first argument
        x2 = x[1,0] # second argument
        value = -0.03 / ((x1 + 0.25) ** 2 + (x2 - 0.2) ** 2 + 0.03) - 0.1 / ((x1 - 0.25) ** 2 + (x2 + 0.2) ** 2 + 0.04) + 0.1 / (x1 ** 2 + x2 ** 2 + 0.05) + 1 + x1 ** 2 + x2 ** 2 + 1 # formula for objective
        return value

    @staticmethod
    def gradient(x: np.array):
        x1 = x[0,0] # first argument
        x2 = x[1,0] # second argument
        dx1 = 2 * (x1 + 0.25) * 0.03 / ((x1 + 0.25) ** 2 + (x2 - 0.2) ** 2 + 0.03) ** 2 + 2 * (x1 - 0.25) * 0.1 / ((x1 - 0.25) ** 2 + (x2 + 0.2) ** 2 + 0.04) ** 2 - 2 * x1 * 0.1 / (x1 ** 2 + x2 ** 2 + 0.05) ** 2 + 2 * x1 # formula for first gradient component
        dx2 = 2 * (x2 - 0.2) * 0.03 / ((x1 + 0.25) ** 2 + (x2 - 0.2) ** 2 + 0.03) ** 2 + 2 * (x2 + 0.2) * 0.1 / ((x1 - 0.25) ** 2 + (x2 + 0.2) ** 2 + 0.04) ** 2 - 2 * x2 * 0.1 / (x1 ** 2 + x2 ** 2 + 0.05) ** 2 + 2 * x2 # formula for second gradient component
        g = np.array([[dx1], [dx2]]) # compose result
        return g

    @staticmethod
    def hessian(x: np.array):
        print('This function has no Hessian!') # warn that calling the hessian of this function is not valid
        h = np.array([[0, 0], [0, 0]]) # still return zero matrix
        return h

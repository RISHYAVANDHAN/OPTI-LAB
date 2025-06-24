# Optimization for Engineers - Dr.Johannes Hild
# Banana Valley test function
# Do not change this file

# Required files:
# < none >

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23356687
    return matrnr


class bananaValleyObjective:
    # Nonlinear function R**2 -> R
    # test function mapping
    # x -> 100*(x[1]]-x[0]**2)**2+(1-x[0])**2+2

    # Input Definition:
    # x: vector in R**2 (domain space)

    # Output Definition:
    # objective: real number, evaluation at x
    # gradient: vector in R**2, evaluation of gradient wrt x
    # hessian: matrix in R**2x2, evaluation of hessian wrt x

    # Test cases:
    # myObjective = bananaValleyObjective.objective(np.array([[1],[1]], dtype=float))
    # should return
    # myObjective = 2

    # myGradient = bananaValleyObjective.gradient(np.array([[1],[1]], dtype=float))
    # should return
    # myGradient = [[0],[0]]

    # myHessian = bananaValleyObjective.hessian(np.array([[1],[1]], dtype=float))
    # should return
    # myHessian = [[802, -400],[-400, 200]]

    @staticmethod
    def objective(x: np.array):
        y = 100*(x[1,0]-x[0,0]**2)**2+(1-x[0,0])**2+2 # objective definition
        return y

    @staticmethod
    def gradient(x: np.array):
        f_dx1 = -400 * (x[1,0] - x[0,0] ** 2) * x[0,0] - 2 * (1 - x[0,0]) # first gradient component
        f_dx2 = 200 * (x[1,0] - x[0,0] ** 2) # second gradient component
        return np.array([[f_dx1], [f_dx2]], dtype=float) # gradient

    @staticmethod
    def hessian(x: np.array):
        f_dx11 = -400 * x[1,0] + 1200 * x[0,0] ** 2 + 2 # Hessian entry for x1x1
        f_dx12 = -400 * x[0,0] # Hessian entry for x1x2 = for x2x1
        f_dx22 = 200 # Hessian entry for x2x2
        return np.array([[f_dx11, f_dx12], [f_dx12, f_dx22]], dtype=float) # hessian

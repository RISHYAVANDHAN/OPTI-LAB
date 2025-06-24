# Optimization for Engineers - Dr.Johannes Hild
# flat objective
# Do not change this file

# 1-dimensional function mapping x -> x**4 -1000*x

# Class parameters:
# < none >

# Input Definition:
# x: vector in R (domain space)

# Output Definition:
# objective(): real number, evaluation at x
# gradient(): vector in R, evaluation of gradient wrt x
# hessian(): matrix in R, evaluation of hessian wrt x

# Required files:
# < none >

# Test cases:
# < none >

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23356687
    return matrnr


class flatObjective:

    def objective(self, x: np.array):
        f = x**4-1000*x # function definition
        return f

    def gradient(self, x: np.array):
        g = 4*x**3-1000 # gradient definition
        return g

    def hessian(self, x: np.array):
        h = 12*x**2 # hessian definition
        return h

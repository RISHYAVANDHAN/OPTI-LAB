# Optimization for Engineers - Dr.Johannes Hild
# directional Hessian Approximation

# Purpose: Approximates Hessian times direction with central differences

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x: column vector in R ** n(domain point)
# d: column vector in R ** n(search direction)
# delta: tolerance for termination. Default value: 1.0e-6
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# dH: Hessian times direction, column vector in R ** n

# Required files:
# < none >

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.01], [1]])
# d = np.array([[1], [1]])

# dH = directionalHessApprox(myObjective, x, d)
# should return dH = [[1.55491],[0]]

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23356687
    return matrnr


def directionalHessApprox(f, x: np.array, d: np.array, delta=1.0e-6, verbose=0):

    if verbose: # print information
        print('Start directionalHessApprox...') # print start

    norm_d = np.linalg.norm(d) # store norm of direction
    dH = 0.5*norm_d/delta*(f.gradient(x+delta/norm_d*d)-f.gradient(x-delta/norm_d*d)) # compute directional Hessian via formula

    if verbose: # print information
        print('directionalHessApprox terminated with dH=', dH) # print termination

    return dH

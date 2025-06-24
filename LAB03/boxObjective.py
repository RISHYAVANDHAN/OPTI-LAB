# Optimization for Engineers - Dr.Johannes Hild
# quadratic box objective that is not defined outside its box
# Do not change this file

# n-dimensional quadratic function mapping x -> 0.5*x'*A*x + b'*x +c

# Class parameters:
# A: real valued matrix nxn
# b: column vector in R^n
# c: real number
# aa: lower bounds of the box, column vector in R^n
# bb: upper bounds of the box, column vector in R^n

# Input Definition:
# x: vector in R**n (domain space)

# Output Definition:
# objective(): real number, evaluation at x
# gradient(): vector in R**n, evaluation of gradient wrt x
# hessian(): matrix in R**nxn, evaluation of hessian wrt x

# Required files:
# < none >

# Test cases:
# A = np.eye(2)
# b = np.ones((2,1))
# c = 1
# aa = -np.ones((2,1))
# bb =  np.ones((2,1))
# myObjective = quadraticObjective(A,b,c,aa,bb)
# y = myObjective.objective(b)
# should return y = 4

# grad = myObjective.gradient(b)
# should return grad = [[2],[2]]

# hess = myObjective.hessian(b)
# should return hess = [[1, 0],[0, 1]]

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23356687
    return matrnr


class boxObjective:

    def __init__(self, A: np.array, b: np.array, c: float, aa: np.array, bb: np.array):
        self.A = A # matrix of quadratic function
        self.b = b # linear part
        self.c = c # constant part
        self.aa = aa # lower bounds
        self.bb = bb # upper bounds

    def isFeasible(self, x: np.array):
        n = x.shape[0] # get dimension
        for i in range(n): # loop over dimension
            if x[i, 0] < self.aa[i, 0]: # if below lower bound
                return False

            if x[i, 0] > self.bb[i, 0]: # if above upper bound
                return False
        return True


    def objective(self, x: np.array):
        if self.isFeasible(x): # check feasibility first
            f = 0.5 * (x.T @ (self.A @ x)) + self.b.T @ x + self.c # evaluate function
            return f
        else:
            raise TypeError('boxObjective is not defined outside the box!')


    def gradient(self, x: np.array):
        if self.isFeasible(x): # check feasibility first
            g = self.A @ x + self.b # evaluate gradient
            return g
        else:
            raise TypeError('boxObjective is not defined outside the box!')


    def hessian(self, x: np.array):
        if self.isFeasible(x): # check feasibility first
            h = self.A # return hessian
            return h
        else:
            raise TypeError('boxObjective is not defined outside the box!')
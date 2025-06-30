# Optimization for Engineers - Dr.Johannes Hild
# scaled unit central simplex gradient

# Purpose: Approximates gradient of f on a scaled unit central simplex

# Input Definition:
# f: objective class with methods .objective()
# x: column vector in R ** n(domain point)
# h: simplex edge length
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# grad_f_h: simplex gradient
# stenFail: 0 by default, but 1 if stencil failure shows up

# Required files:
# < none >

# Test cases:
# myObjective = multidimensionalObjective()
# x = np.array([[1.02614],[0],[0],[0],[0],[0],[0],[0]], dtype=float)
# h = 1.0e-6
# myGradient = SUCSGradient(myObjective, x, h)
# should return
# myGradient close to [[0],[0],[0],[0],[0],[0],[0],[0]]


import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23356687
    return matrnr


def SUCSGradient(f, x: np.array, h: float, verbose=0):

    if verbose: # print information
        print('Start SUCSGradient...') # print start

    grad_f_h = x.copy() # initialize simplex gradient of f

    # INCOMPLETE CODE STARTS
    n = x.shape[0]                                          # get dimension of the vector x
    grad_f_h = np.zeros_like(x)                             # initialize gradient vector as zero vector
    
    for j in range(n):                                      # iterate through all dimensions
        e_j = np.zeros_like(x)                              # create unit vector e_j
        e_j[j, 0] = 1.0                                     # set j-th component to 1
        
        f_forward = f.objective(x + h * e_j)                # evaluate f at forward simplex point x + h*e_j  
        f_backward = f.objective(x - h * e_j)               # evaluate f at reflected simplex point x - h*e_j
        
        grad_f_h[j, 0] = (f_forward - f_backward) / (2.0 * h)  # compute finite difference gradient component
    # INCOMPLETE CODE ENDS

    if verbose: # print information
        print('SUCSGradient terminated with gradient =', grad_f_h) # print termination

    return grad_f_h


def SUCSStencilFailure(f, x: np.array, h: float, verbose=0):

    if verbose: # print information
        print('Check for SUCSStencilFailure...') # print start of check

    stenFail = 1 # initialize stencil failure with true

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    
    n = x.shape[0]                                          # get dimension of the vector x
    f_center = f.objective(x)                               # evaluate function at center point x
    stenFail = 1                                            # initialize stencil failure as true (assume failure)
    
    for j in range(n):                                      # iterate through all dimensions
        e_j = np.zeros_like(x)                              # create unit vector e_j
        e_j[j, 0] = 1.0                                     # set j-th component to 1
        
        f_forward = f.objective(x + h * e_j)                # evaluate f at forward simplex point
        f_backward = f.objective(x - h * e_j)               # evaluate f at backward simplex point
        
        if f_center > f_forward or f_center > f_backward:   # check if center point is not minimal
            stenFail = 0                                    # no stencil failure found
            break                                           # exit loop early since we found a point with lower function value
    # INCOMPLETE CODE ENDS
    
    if verbose: # print information
        print('SUCSStencilFailure check returns ', stenFail) # print termination

    return stenFail

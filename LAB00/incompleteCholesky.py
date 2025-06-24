# Optimization for Engineers - Dr.Johannes Hild
# Incomplete Cholesky decomposition

# Purpose: incompleteCholesky finds lower triangle matrix L such that A - L * L ^ T is small, but
# eigenvalues are positive and sparsity is preserved

# Input Definition:
# A: real valued symmetric matrix nxn
# alpha: non-negative scalar, lower bound for eigenvalues of L * L ^ T.Default value: 1.0e-3.
# delta: scalar, if positive it is tolerance for recognizing non-sparse entry.
# If negative, do complete cholesky.Default value: 1.0e-6.
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# A: real valued lower triangle matrix nxn

# Required files:
# < none >

# Test cases:
# alpha = 0
# delta = -1
# verbose = true
# L = incompleteCholesky(np.array([[5, 4, 3, 2, 1],[4, 5, 2, 1, 0],\
# [3, 2, 5, 0, 0],[2, 1, 0, 5, 0],[1, 0, 0, 0, 5]]), alpha, delta, verbose)
# executes complete Cholesky decomposition with norm of residual approx. 8.89e-16
# the warning is okay.

# alpha = 1.0e-3
# delta = 1.0e-6
# verbose = true
# L = incompleteCholesky(np.array([4, 1, 0],[1, 4, 0],[0, 0, 4]]), alpha, delta, verbose)
# should return approximately
# L = [[2 0 0],[0.5 1.94 0], [ 0 0 2]]

# alpha = 4
# delta = 1.0e-6
# verbose = true
# L = incompleteCholesky(np.array([4, 1, 0], [1, 4, 0], [0, 0, 4]]), alpha, delta, verbose)
# should return approximately
# L = [[2 0 0],[0.5 2 0], [ 0 0 2]]

# alpha = 1.0e-3
# delta = 1
# verbose = true
# L = incompleteCholesky(np.array([[4, 1, 0], [1, 4, 0], [0, 0, 4]]), alpha, delta, verbose)
# should return approximately
# L = [[2 0 0],[0 2 0], [ 0 0 2]]

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 0
    return matrnr


def incompleteCholesky(A: np.array, alpha=1.0e-3, delta=1.0e-6, verbose=0):
    L = np.copy(A) # initialize L as copy of A
    dim = np.shape(L) # get matrix dimensions
    n = dim[0] # matrix dimension
    if n != dim[1]: # check for quadratic matrix
        raise ValueError('A has wrong dimension.')

    if np.max(np.abs(A-A.T) > 1.0e-6): # check for symmetry
        raise ValueError('A is not symmetric.')

    if alpha < 0: # check for nonnegative alpha
        raise ValueError('range of alpha is wrong!')

    if delta < 0: # check for nonnegative delta
        print('Warning: negative delta detected, sparsity is not preserved.')

    if verbose: # print information
        print('Start incompleteCholesky...') # print start

    sqrt_alpha = np.sqrt(alpha) # store sqrt of alpha
    for k in range(n): # loop over matrix dimension
        if L[k, k] > alpha: # if diagonal element is positive
            L[k, k] = np.sqrt(L[k, k]) # set to its root
        else:
            L[k, k] = sqrt_alpha # set to root of alpha

        for i in range(k+1, n): # loop over current index up to dimension
            if np.abs(L[i, k]) > delta: # if element is big enough
                L[i, k] = L[i, k] / L[k, k] # scale it accordingly
            else:
                L[i, k] = 0 # round it down to zero

        for j in range(k+1, n): # loop over current index up to dimension
            for i in range(j, n): # loop over current subindex up to dimension
                if np.abs(L[i, j]) > delta: # if element is big enough
                    L[i, j] = L[i, j] - L[i, k] * L[j, k] # update according to formula
            L[k, j] = 0 # set remaining entries to zero

    if verbose: # print information
        residualmatrix = A - L @ L.T # residual matrix error
        residual = np.max(np.abs(residualmatrix)) # residual value
        print('IncompleteCholesky terminated with norm of residual: ', residual) # print termination with residual error

    return L

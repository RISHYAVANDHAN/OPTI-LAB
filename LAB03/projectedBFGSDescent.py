# Optimization for Engineers - Dr.Johannes Hild
# projected BFGS descent

# Purpose: Find xmin to satisfy norm(xmin - P(xmin - gradf(xmin)))<=eps
# Iteration: x_k = P(x_k + t_k * d_k)
# d_k is the reduced BFGS direction. If a descent direction check fails, d_k is set to steepest descent and the BFGS matrix is reset.
# t_k results from projected backtracking

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# d = PrecCGSolver(A,b) from PrecCGSolver.py
# t = projectedBacktrackingSearch(f, P, x, d) from projectedBacktrackingSearch.py

# Test cases:
# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

import numpy as np
import projectedBacktrackingSearch as PB
import PrecCGSolver as PCG


def matrnr():
    # set your matriculation number here
    matrnr = 23356687
    return matrnr

def projectedBFGSDescent(f, P, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0:                                                # check for positive eps
        raise TypeError('range of eps is wrong!')

    if verbose:                                                 # print information
        print('Start projectedBFGSDescent...')                  # print start

    countIter = 0                                               # counter for number of loop iterations
    xk = P.project(x0)                                          # initialize with projected starting point    
    n = x0.shape[0]                                             # Get the shape of x0 to match the identity matrix size 
    Hk = np.eye(n)                                              # initialise Hk with identity matrix
    Ak = P.activeIndexSet(xk)                                   # initialise A from the active index
    gradx = f.gradient(xk)                                      # gradient for further calculations

    # Main optimization loop
    while (np.linalg.norm(xk - P.project(xk - gradx)) > eps):                           # checking for descent
        dk = PCG.PrecCGSolver(Hk, -gradx)                                               # using Preconditioned CG solver from LAB01
        if gradx.T @ dk >= 0:                                                           # Ensure descent direction
            dk = -gradx                                                                 # Resetting dx manually to -gradx for descent
            Hk = np.eye(n)                                                              # Reset only when forced to steepest descent
        
        tk = PB.projectedBacktrackingSearch(f, P, xk, dk)                               # calculaiting pos tk using alg 4.17 (ProjectedBackTracking using the other file from this exercise)
        x_plus = P.project(xk + tk * dk)
        A_plus = P.activeIndexSet(x_plus)
        grad_new = f.gradient(x_plus)

        if not np.array_equal(A_plus, Ak):
            Hk[A_plus, :] = np.eye(n)[A_plus, :]                                        # overwrite active rows
            Hk[:, A_plus] = np.eye(n)[:, A_plus]                                        # overwrite active cols

        else:
            delta_gk = grad_new - gradx                                                 # calculate delta gk
            delta_xk = x_plus - xk                                                      # calculate delta_xk
            curvature = delta_gk.T @ delta_xk                                           # common parts of the calculation and we know that its the curvature
            if (curvature <= eps**2):                           
                Hk = np.eye(n)                                                          # Hessian reset

            else:
                rho = 1.0 / curvature                                                   # Helper terms, because these stuffs gets repeated while calculating the update Hk corresponding to the lemma
                Vk = Hk @ delta_xk                                                      # Helper terms
                Hk += (delta_gk @ delta_gk.T) * rho - (Vk @ Vk.T) / (delta_xk.T @ Vk)   # updated Hk according to the lemma
                Hk[A_plus, :] = np.eye(n)[A_plus, :]                                    # reduce after update (rows)
                Hk[:, A_plus] = np.eye(n)[:, A_plus]                                    # reduce after update (columns)

        xk = x_plus
        Ak = A_plus

        countIter += 1
        gradx = f.gradient(xk)
        
        # Safety stop
        if countIter > 1000:
            if verbose:
                print('Warning: max iterations reached')
            break

    # Final output
    if verbose: # print information
        gradx = f.gradient(xk) # get gradient
        stationarity = np.linalg.norm(xk - P.project(xk - gradx)) # get stationarity
        print('projectedBFGSDescent terminated after ', countIter, ' steps with stationarity =', np.linalg.norm(stationarity)) # print termination

    return xk

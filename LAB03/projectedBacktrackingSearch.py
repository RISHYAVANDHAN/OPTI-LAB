# Optimization for Engineers - Dr.Johannes Hild
# projected Wolfe-Powell line search

# Purpose: Find t to satisfy f(P(x+t*d))<f(x) + sigma*gradf(x).T@(P(x+t*d)-x) with P(x+t*d)-x being a descent direction
# and in addition but only if x+t*d is inside the feasible set gradf(x+t*d).T@d >= rho*gradf(x).T@d

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project()
# x: column vector in R ** n(domain point)
# d: column vector in R ** n(search direction)
# sigma: value in (0, 1 / 2), marks quality of decrease. Default value: 1.0e-3
# rho: value in (sigma, 1), marks quality of steepness. Default value: 1.0e-2
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# t: t is set to the biggest 2**m, such that 2**m satisfies the projected sufficient decrease condition
# and in addition if x+t*d is inside the feasible set, the second Wolfe-Powell condition holds

# Required files:
# <none>

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[-2], [1]])
# b = np.array([[2], [2]])
# eps = 1.0e-6
# myBox = projectionInBox(a, b, eps)
# x = np.array([[1], [1]])
# d = np.array([[-1.99], [0]])
# sigma = 0.5
# rho = 0.75
# t = projectedBacktrackingSearch(myObjective, myBox, x, d, sigma, rho, 1)
# should return t = 0.5

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23356687
    return matrnr


def projectedBacktrackingSearch(f, P, xk: np.array, d: np.array, sigma=1.0e-4, rho=1.0e-2, verbose=0):
    xp = P.project(xk) # initialize with projected starting point
    fx = f.objective(xp) # get current objective
    gradx = f.gradient(xp) # get current gradient
    descent = gradx.T @ d # descent direction check value

    if descent >= 0: # if not a descent direction
        raise TypeError('descent direction check failed!')

    if sigma <= 0 or sigma >= 0.5: # if sigma is out of range
        raise TypeError('range of sigma is wrong!')

    if rho <= sigma or rho >= 1: # if rho does not fit to sigma
        raise TypeError('range of rho is wrong!')

    if verbose: # print information
        print('Start projectedBacktracking...') # print start

    t = 1 # starting guess for t

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE

    
    # Check if point is stationary
    if np.array_equal(P.project(xk + d), xk):
        raise TypeError('x is stationary!')
    
    # Define Wolfe-Powell condition 1
    def W1(t):
        xt = P.project(xk + t * d)
        diff = xt - xk
        grad_diff = (gradx.T @ diff)
        if grad_diff >= 0:
            return False
        fxt = f.objective(xt)
        return fxt <= fx + sigma * grad_diff
    
    # Define Wolfe-Powell condition 2
    def W2(t):
        xt = P.project(xk + t * d)
        if not np.array_equal(xk + t * d, xt):
            return True
        grad_xt = f.gradient(xt)
        return grad_xt.T @ d >= rho * descent # descent is already calculated in the beginning, so we can directly use  it.
        
    # Backtracking if W1 fails at t
    if W1(t) == False:                      # condiiton to check if W1 fails
        t /= 2                              # halving the t to get a smaller t and converge it
        while W1(t) == False:               # step to be repeated until W1 is true
            t /= 2                          # if w1 is false, continue halving t again
        t_minus = t                         # update t- as t
        t_plus = 2 * t                      # and update the t+ with 2t
    
    # Return t if both conditions satisfied
    elif W2(t) == True:                                                     # check for w2                  
        if verbose:                                                         # check for verbose
            print('projectedBacktracking terminated with t=', t)            # print completion if verbose is true
        return t                                                            # return t if the w2 is is true
    
    # Fronttracking if W1 passes but W2 fails
    else:                                                                   # if w2 is false
        t = 2 * t                                                           # increase t to 2t
        xt = P.project(xk + t * d)                                          # get the corresponding updated x for the new t
        # Double t while conditions hold and point is feasible
        while ((W1(t) == True) and np.array_equal(xt, (xk + t * d))):       # check if w1 satisfies and the projected x and non-projected x is same
            t = 2 * t                                                       # if yes, increase t to 2t again
            xt = P.project(xk + t * d)                                      # project the x
        t_minus = t / 2                                                     # if not, update t- with t/2
        t_plus = t                                                          # and t+ with t
    
    t = t_minus                                                             # Start refinement from t_minus

    while W2(t) == False:                                                   # Refine until W2 condition is satisfied
        t_mid = (t_minus + t_plus) / 2                                      # get the mid of t+ and t-
        if W1(t_mid) == True:                                               # check if w1 satisfies for this new t, t_mid
            t_minus = t_mid                                                 # if yes, update t- with t_mid
        else:
            t_plus = t_mid                                                  # if not, t+ with t_mid
        t = t_minus                                                         # update the t if w2 is not satisfied with the corresponding t- or the previous t-

    # INCOMPLETE CODE ENDS

    if verbose: # print verbose information
        xt = P.project(xk + t * d) # get x+td for found step size t
        fxt = f.objective(xt) # get objective value at this point
        print('projectedBacktracking terminated with t=', t) # print termination
        print('Sufficient decrease: ', fxt, '<=', fx+t*sigma*descent) # print result of sufficient decrease check

    return t

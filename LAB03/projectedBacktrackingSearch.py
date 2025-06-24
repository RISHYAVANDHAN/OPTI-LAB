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
        t /= 2                              # halving the "time" but not technically t
        while W1(t) == False:               # step to be repeated until W1 is true
            t /= 2                          # halving again
        t_minus = t                         # 
        t_plus = 2 * t
    
    # Return t if both conditions satisfied
    elif W2(t) == True:
        if verbose:
            print('projectedBacktracking terminated with t=', t)
        return t
    
    # Fronttracking if W1 passes but W2 fails
    else:
        t = 2 * t
        xt = P.project(xk + t * d)
        # Double t while conditions hold and point is feasible
        while ((W1(t) == True) and np.array_equal(xt, (xk + t * d))):
            t = 2 * t
            xt = P.project(xk + t * d)
        t_minus = t / 2
        t_plus = t
    
    t = t_minus                             # Start refinement from t_minus

    while W2(t) == False:                        # Refine until W2 condition is satisfied
        t_mid = (t_minus + t_plus) / 2
        if W1(t_mid) == True:
            t_minus = t_mid
        else:
            t_plus = t_mid
        t = t_minus

    # Step 12: Output t_minus
    if verbose: # print verbose information
        xt = P.project(xk + t * d) # get x+td for found step size t
        fxt = f.objective(xt) # get objective value at this point
        print('projectedBacktracking terminated with t=', t) # print termination
        print('Sufficient decrease: ', fxt, '<=', fx+t*sigma*descent) # print result of sufficient decrease check

    return t

# Optimization for Engineers - Dr.Johannes Hild
# Wolfe-Powell line search

# Purpose: Find t to satisfy f(x+t*d)<=f(x) + t*sigma*gradf(x).T@d
# and gradf(x+t*d).T@d >= rho*gradf(x).T@d

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x: column vector in R ** n(domain point)
# d: column vector in R ** n(search direction)
# sigma: value in (0, 1 / 2), marks quality of decrease. Default value: 1.0e-3
# rho: value in (sigma, 1), marks quality of steepness. Default value: 1.0e-2
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# t: t is set, such that t satisfies both Wolfe - Powell conditions

# Required files:
# < none >

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.01], [1]])
# d = np.array([[1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=1

# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.2], [1]])
# d = np.array([[0.1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=16

# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-0.2], [1]])
# d = np.array([[1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=0.25

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23356687
    return matrnr


def WolfePowellSearch(f, x: np.array, d: np.array, sigma=1.0e-3, rho=1.0e-2, verbose=0):
    fx = f.objective(x) # store objective
    gradx = f.gradient(x) # store gradient
    descent = gradx.T @ d # store descent value

    if descent >= 0: # if not a descent direction
        raise TypeError('descent direction check failed!')

    if sigma <= 0 or sigma >= 0.5: # if sigma is out of range
        raise TypeError('range of sigma is wrong!')

    if rho <= sigma or rho >= 1: # if rho does not fit to sigma
        raise TypeError('range of rho is wrong!')

    if verbose: # print information
        print('Start WolfePowellSearch...') # print start

    t = 1 # initial step size guess

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    
    # Define Wolfe-Powell condition 1: sufficient decrease
    def W1(t):
        xt = x + t * d                                                  # Compute new point
        return f.objective(xt) <= fx + t * sigma * descent              # Check condition

    # Define Wolfe-Powell condition 2: curvature condition
    def W2(t):
        xt = x + t * d                                                  # Compute new point
        gradxt = f.gradient(xt)                                         # Compute gradient at new point
        return (gradxt.T @ d) >= rho * descent                          # Check condition

    max_iter = 30  # Set maximum number of refinement iterations
    countIter = 0  # Initialize iteration counter

    # Step 1: Backtracking if W1 fails at t=1
    if not W1(t):
        while not W1(t):
            t = t / 2.0  # halve t until W1 is satisfied
            countIter += 1
            if countIter > max_iter:
                raise Exception('Too many iterations in backtracking')
        t_minus = t
        t_plus = 2 * t
    # Step 2: If W1 holds but W2 fails, fronttracking
    elif not W2(t):
        while W1(t):
            t = 2 * t  # double t until W1 fails
            countIter += 1
            if countIter > max_iter:
                raise Exception('Too many iterations in fronttracking')
        t_minus = t / 2.0
        t_plus = t
    # Step 3: If both W1 and W2 hold at t=1, return t=1
    else:
        if verbose:
            xt = x + t * d
            fxt = f.objective(xt)
            gradxt = f.gradient(xt)
            print('WolfePowellSearch terminated with t=', t)
            print('Wolfe-Powell: ', fxt, '<=', fx+t*sigma*descent, ' and ', gradxt.T @ d, '>=', rho*descent)
        return t

    # Step 4: Refinement (bisection) between t_minus and t_plus
    refine_iter = 0
    t_current = t_minus
    
    while not W2(t_current):
        t_candidate = (t_minus + t_plus) / 2.0  # bisect interval
        if W1(t_candidate):
            t_minus = t_candidate  # move lower bound up
        else:
            t_plus = t_candidate  # move upper bound down
        t_current = t_minus
        refine_iter += 1
        if refine_iter > max_iter:
            raise Exception('Too many iterations in refinement')

    t = t_minus  # final step size

    if verbose:  # print information
        xt = x + t * d  # store solution point
        fxt = f.objective(xt)  # get its objective
        gradxt = f.gradient(xt)  # get its gradient
        print('WolfePowellSearch terminated with t=', t)  # print terminatin and step size
        print('Wolfe-Powell: ', fxt, '<=', fx+t*sigma*descent, ' and ', gradxt.T @ d, '>=', rho*descent)  # print Wolfe-Powell checks

    
    if countIter > 30:
        raise Exception('Its going over the maximum counf of 30')
    
    return t

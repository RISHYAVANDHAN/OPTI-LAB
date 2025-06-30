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

    def WP1(ft, s):                                         # defining w1 
        isWP1 = ft <= fx + s*sigma*descent                  # boolean check for checking if w1 is true
        return isWP1                                        # return the boolean

    def WP2(gradft: np.array):                              # defining W2
        isWP2 = gradft.T @ d >= rho*descent                 # boolean check for w2
        return isWP2                                        # return the boolean

    if gradx.T @ d >= 0:                                    # check the descent direction first before proceedimnmng
        raise TypeError('descent direction check failed!')  # statemnt if the check fails

    if WP1(f.objective(x+t*d), t) == False:                 # check if w1 paases
        t = t/2                                             # update t
        while WP1(f.objective(x + t*d), t) == False:        # check for it again
            t = t/2                                         # update t again
        t_minus = t                                         # if the intermediate check failed, update t_minus
        t_plus = 2*t                                        # and also t_plus
    
    elif WP2(f.gradient(x + t*d)) == True:                  # check for w2
        t_star = t                                          # update the t and return it
        return t_star
    
    else :                                                  # if the check failed, then 
        t = 2*t                                             # update t
        while WP1(f.objective(x+t*d), t) == True:           # check for w1 now (front tracking)
            t = 2*t                                         # update t if passes
        t_minus = t/2                                       # if it failed update t_minus
        t_plus = t                                          # and t_plus

    t = t_minus                                             # updte t with t_minus
    while WP2(f.gradient(x + t*d)) == False:                # check for w2
        t = (t_minus + t_plus)/2                            # update t with the average of t- and t+
        if WP1(f.objective(x + t*d),t) == True:             # check for w1
            t_minus = t                                     # update t- if it passes
        else:                                               # if not
            t_plus = t                                      # update t+
    t_star = t_minus                                        # assign t as t-
    
    # INCOMPLETE CODE ENDS

    if verbose:
        xt = x + t * d
        fxt = f.objective(xt)
        gradxt = f.gradient(xt)
        print('WolfePowellSearch terminated with t=', t)
        print('Wolfe-Powell: ', fxt, '<=', fx+t*sigma*descent, ' and ', gradxt.T @ d, '>=', rho*descent)

    return t_star
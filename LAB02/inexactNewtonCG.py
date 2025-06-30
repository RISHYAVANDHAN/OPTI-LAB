# Optimization for Engineers - Dr.Johannes Hild
# inexact Newton CG

# Purpose: Find xmin to satisfy norm(gradf(xmin))<=eps
# Iteration: x_k = x_k + t_k * d_k
# d_k starts as a steepest descent step and then CG steps are used to improve the descent direction until negative curvature is detected or a full Newton step is made.
# t_k results from Wolfe-Powell

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# dA = directionalHessApprox(f, x, d) from directionalHessApprox.py
# t = WolfePowellSearch(f, x, d) from WolfePowellSearch.py

# Test cases:
# myObjective = noHessianObjective()
# x0 = np.array([[-0.01], [0.01]])
# xmin = inexactNewtonCG(myObjective, x0, 1.0e-6, 1)
# should return
# xmin close to [[0.26],[-0.21]]

import numpy as np
import WolfePowellSearch as WP
import directionalHessApprox as DHA

def matrnr():
    # set your matriculation number here
    matrnr = 23356687
    return matrnr


def inexactNewtonCG(f, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0: # check for positive eps
        raise TypeError('range of eps is wrong!')

    if verbose: # print information
        print('Start inexactNewtonCG...') # print start

    countIter = 0 # counter for number of loop iterations
    xk = x0 # initialize starting iteration

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE

    grad_fk = f.gradient(xk)                                                # Calculate gradient for further use
    norm_grad_fk = np.linalg.norm(grad_fk)                                  # caluclate norm of the gradient for further use
    eta_k = np.min([(0.5, np.sqrt(norm_grad_fk))]) * norm_grad_fk           # set Eta value first

    while norm_grad_fk > eps:                                               # Termination condition
        xj = xk.copy()                                                      # reassign value to use inside the loop
        rj = grad_fk.copy()                                                 # reassign the r to use inside the loop
        dj = -rj.copy()                                                     # get the descent directionn first from rj, then will be updated inside

        while np.linalg.norm(rj) > eta_k:                                   # termination condition

            if np.linalg.norm(dj) < 1e-12:                                  # numerical sanity check
                break                                                       # CG direction broke down numerically

            dA = DHA.directionalHessApprox(f, xk, dj)                       # given as per the readme file
            rhoj = dj.T @ dA                                                # set rho from the output and descent direction

            if not np.isfinite(rhoj) or rhoj<= eps*np.linalg.norm(dj) ** 2: # sanity check also a check included in the algoirithem
                break  # curvature fail or invalid

            tj = (np.linalg.norm(rj) ** 2) / rhoj                           # calculate tj
            xj_new = xj + tj * dj                                           # update x with t and d

            rold = rj.copy()                                                # reuse r for the checks
            rj = rold + tj * dA                                             # update r
            beta_j = (np.linalg.norm(rj) ** 2)/(np.linalg.norm(rold) ** 2)  # calculate beta to update d
            dj = -rj + beta_j * dj                                          # update descent direction d

            xj = xj_new.copy()                                              # reassign x to use later

        dk = xj - xk                                                        # get the diff when xj and xk are not equal

        if np.linalg.norm(dk) < 1e-12:                                      # the other check where xj = xk
            dk = -grad_fk                                                   # fallback to steepest descent if no CG progress (xj = xk)

        tk = WP.WolfePowellSearch(f, xk, dk)                                # update t
        xk = xk + tk * dk                                                   # update x for the last time

        grad_fk = f.gradient(xk)                                            # calculate gradient to use for the next loop
        norm_grad_fk = np.linalg.norm(grad_fk)                              # calculate norm to use for next loop condition check
        eta_k = np.min((0.5, np.sqrt(norm_grad_fk))) * norm_grad_fk         # eta for next loop condition check
        countIter += 1                                                      # update the counter


    # INCOMPLETE CODE ENDS
    
    if verbose: # print information
        stationarity = np.linalg.norm(f.gradient(xk)) # store stationarity value
        print('inexactNewtonCG terminated after ', countIter, ' steps with norm of gradient =', stationarity) # print termination with stationarity value

    return xk

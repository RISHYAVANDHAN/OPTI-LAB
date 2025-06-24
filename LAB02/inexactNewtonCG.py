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

    grad_fk = f.gradient(xk)  # compute gradient at xk
    norm_grad_fk = np.linalg.norm(grad_fk)  # compute norm of gradient

    # Compute initial eta_k
    eta_k = np.min([0.5, np.sqrt(norm_grad_fk)]) * norm_grad_fk  # set initial eta_k

    max_outer_iter = 30  # maximum number of Newton iterations

    while norm_grad_fk > eps and countIter < max_outer_iter:  # main loop
        xj = xk.copy()  # initialize CG variable xj
        rj = grad_fk.copy()  # residual is gradient at xk
        dj = -rj  # initial CG direction is negative gradient

        norm_rj = np.linalg.norm(rj)  # norm of residual

        # CG loop: solve approximately Hessian system
        while norm_rj > eta_k:
            dA = DHA.directionalHessApprox(f, xk, dj)  # approximate Hessian-vector product
            rhoj = float(dj.T @ dA)  # compute curvature

            if rhoj <= eps * np.linalg.norm(dj) ** 2:  # check for negative/poor curvature
                break  # exit CG loop

            tj = float((rj.T @ rj) / rhoj)  # compute step length in CG
            xj_new = xj + tj * dj  # update xj

            rold = rj.copy()  # store old residual
            rj = rold + tj * dA  # update residual

            beta_j = float((rj.T @ rj) / (rold.T @ rold))  # compute beta for CG direction
            dj = -rj + beta_j * dj  # update CG direction

            xj = xj_new  # update xj for next CG step
            norm_rj = np.linalg.norm(rj)  # update norm of residual

        # Compute Newton direction
        if np.allclose(xj, xk):  # if no CG progress, use steepest descent
            dk = -grad_fk
        else:
            dk = xj - xk  # Newton direction from CG

        # Wolfe-Powell line search for step size
        tk = WP.WolfePowellSearch(f, xk, dk)  # get step size

        xk = xk + tk * dk  # update iterate
        grad_fk = f.gradient(xk)  # update gradient
        norm_grad_fk = np.linalg.norm(grad_fk)  # update norm of gradient

        eta_k = np.min([0.5, np.sqrt(norm_grad_fk)]) * norm_grad_fk  # update eta_k

        countIter += 1  # increment iteration counter


    # INCOMPLETE CODE ENDS
    
    if verbose: # print information
        stationarity = np.linalg.norm(f.gradient(xk)) # store stationarity value
        print('inexactNewtonCG terminated after ', countIter, ' steps with norm of gradient =', stationarity) # print termination with stationarity value

    return xk

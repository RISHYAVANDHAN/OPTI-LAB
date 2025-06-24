# Optimization for Engineers - Dr.Johannes Hild
# Levenberg-Marquardt descent

# Purpose: Find pmin to satisfy norm(jacobian_R.T @ R(pmin))<=eps

# Input Definition:
# R: error vector class with methods .residual() and .jacobian()
# p0: column vector in R**n (parameter point), starting point.
# eps: positive value, tolerance for termination. Default value: 1.0e-4.
# alpha0: positive value, starting value for damping. Default value: 1.0e-3.
# beta: positive value bigger than 1, scaling factor for alpha. Default value: 100.
# verbose: bool, if set to true, verbose information is displayed.

# Output Definition:
# pmin: column vector in R**n (parameter point)

# Required files:
# d = PrecCGSolver(A,b) from PrecCGSolver.py

# Test cases:
# p0 = np.array([[180],[0]], dtype=float)
# myObjective =  simpleValleyObjective(p0)
# xk = np.array([[[0], [0]], [[1], [2]]], dtype=float)
# fk = np.array([[2], [3]], dtype=float)
# myErrorVector = leastSquaresModel(myObjective, xk, fk)
# eps = 1.0e-4
# alpha0 = 1.0e-3
# beta = 100
# pmin = levenbergMarquardtDescent(myErrorVector, p0, eps, alpha0, beta, 1)
# should return pmin close to [[1], [1]]

import numpy as np
import PrecCGSolver as PCG


def matrnr():
    # set your matriculation number here
    matrnr = 23356687
    return matrnr


def levenbergMarquardtDescent(R, p0: np.array, eps=1.0e-4, alpha0=1.0e-3, beta=100, verbose=0):
    if eps <= 0: # check for positive eps
        raise TypeError('range of eps is wrong!')

    if alpha0 <= 0: # check for positive alpha0
        raise TypeError('range of alpha0 is wrong!')

    if beta <= 1: # check for sufficiently large beta
        raise TypeError('range of beta is wrong!')

    if verbose: # print information
        print('Start levenbergMarquardtDescent...') # print start

    countIter = 0 # counter for loop iterations

    p = p0 # initialize p with starting point
    alpha = alpha0 # initialize damping parameter alpha
    J = R.jacobian(p) # compute Jacobian at current p
    r = R.residual(p) # compute residual at current p
    grad = J.T @ r # compute gradient of the objective
    grad_norm = np.linalg.norm(grad) # compute norm of the gradient

    while grad_norm > eps:
        J = R.jacobian(p) # compute Jacobian at current p
        r = R.residual(p) # compute residual at current p
        grad = J.T @ r # compute gradient of the objective
        grad_norm = np.linalg.norm(grad) # compute norm of the gradient

        if verbose: # print current iteration and gradient norm
            print(f"Iter {countIter}: ||grad|| = {grad_norm}, alpha = {alpha}")

        if grad_norm <= eps: # check termination criterion
            break # exit loop if gradient norm is small enough

        A = J.T @ J + alpha * np.eye(p.shape[0]) # build LM system matrix
        b = -grad # right-hand side for LM step

        d = PCG.PrecCGSolver(A, b) # solve for step direction using preconditioned CG

        p_new = p + d # compute new candidate point
        r_new = R.residual(p_new) # compute new residual
        if ((r_new.T @ r_new) < (r.T @ r)):
            p = p + d  # Accept step: update p
            alpha = alpha0  # Reset damping
            countIter += 1  # Only increment when step accepted
        else:
            alpha = beta * alpha  # Reject step: increase damping, keep same p

    if verbose: # print information
        gradp = R.jacobian(p).T @ R.residual(p) # store final gradient
        print('levenbergMarquardtDescent terminated after ', countIter, ' steps with norm of gradient =', np.linalg.norm(gradp)) # print termination and gradient information

    return p

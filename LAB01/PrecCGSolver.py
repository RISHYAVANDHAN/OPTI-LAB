# Optimization for Engineers - Dr.Johannes Hild
# Preconditioned Conjugate Gradient Solver

# Purpose: PregCGSolver finds y such that norm(A * y - b) <= delta using incompleteCholesky as preconditioner

# Input Definition:
# A: real valued matrix nxn
# b: column vector in R ** n
# delta: positive value, tolerance for termination. Default value: 1.0e-6.
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# x: column vector in R ^ n(solution in domain space)

# Required files:
# L = incompleteCholesky(A, 1.0e-3, delta) from IncompleteCholesky.py
# y = LLTSolver(L, r) from LLTSolver.py

# Test cases:
# A = np.array([[4, 1, 0], [1, 7, 0], [ 0, 0, 3]], dtype=float)
# b = np.array([[5], [8], [3]], dtype=float)
# delta = 1.0e-6
# x = PrecCGSolver(A, b, delta, 1)
# should return x = [[1], [1], [1]]

# A = np.array([[484, 374, 286, 176, 88], [374, 458, 195, 84, 3], [286, 195, 462, -7, -6], [176, 84, -7, 453, -10], [88, 3, -6, -10, 443]], dtype=float)
# b = np.array([[1320], [773], [1192], [132], [1405]], dtype=float)
# delta = 1.0e-6
# x = PrecCGSolver(A, b, delta, 1)
# should return approx x = [[1], [0], [2], [0], [3]]


import numpy as np
import incompleteCholesky as IC
import LLTSolver as LLT


def matrnr():
    # set your matriculation number here
    matrnr = 23356687
    return matrnr


def PrecCGSolver(A: np.array, b: np.array, delta=1.0e-6, verbose=0):

    if verbose: # print information
        print('Start PrecCGSolver...') # print start

    countIter = 0                                                       # counter for number of loop iterations

    L = IC.incompleteCholesky(A)                                        # Step 2: Preconditioner definition
    xj = np.zeros_like(b)                                               # Step 3: Initial guess for the routine
    rj = A @ xj - b                                                     # Step 3: Initial residual for the initial guessed value
    zj = LLT.LLTSolver(L, rj)                                           # Preconditioned residual using the perconditioned matrix obtained form choleskey
    dj = -zj                                                            # Step 3: Initial direction (descent direction)

    while np.linalg.norm(rj) > delta:
        dj_tilda = A @ dj                                               # Step 4a: calculating ˜dj ← Adj 
        rhoj = float(dj.T @ dj_tilda)                                   # Step 4b: assigning ρj ← dj^⊤ ˜dj
        tj_num = float(rj.T @ LLT.LLTSolver(L, rj))                     # Step 4c: numerator for further use
        tj = tj_num / rhoj                                              # Step 4c: calculating and assigning tj ← rj^⊤LLTSolver(L,rj)/ρj
        xj = xj + tj * dj                                               # Step 4d: updating x: xj ← xj + tjdj
        r_old = rj.copy()                                               # Step 4e: copying old r to use after the loop: rold ← rj
        rj = r_old + tj * dj_tilda                                      # Step 4f: updating new r with old r: rj ← rold + tj ˜dj
        zj = LLT.LLTSolver(L, rj)                                       # Preconditioned residual for new rj after updation
        betaj_num = float(rj.T @ zj)                                    # Step 4g: numerator for further use
        betaj_den = float(r_old.T @ LLT.LLTSolver(L, r_old))            # Step 4g: denominator for further use
        betaj = betaj_num / betaj_den if betaj_den != 0 else 0          # Step 4g: βj with the if-else case
        dj = -zj + betaj * dj                                           # Step 4h: new descent direction dj ← −LLTSolver(L, rj ) + βjdj
        countIter += 1                                                  # Increment iteration counter to check if it has exceeded the limit or not 

    x = xj                                                              # Output x

    if countIter > 30:
        raise Exception('Its going over the maximum count of 30')

    if verbose: # print information
        print('precCGSolver terminated after ', countIter, ' steps with norm of residual being ', np.linalg.norm(rj)) # print termination

    return x
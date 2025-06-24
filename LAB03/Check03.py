# Optimization for Engineers - Dr.Johannes Hild
# Programming Homework Check Script
# Do not change this file

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your programming homework is not working correctly.')

import numpy as np
import simpleValleyObjective as SO
import boxObjective as BO
import noHessianObjective as NO
import multidimensionalObjective as MO
import flatObjective as FO
import projectionInBox as PB
import projectedBacktrackingSearch as PS
import projectedBFGSDescent as PBD

p = np.array([[0], [1]], dtype=float)
myObjective = SO.simpleValleyObjective(p)
a = np.array([[-2], [1]], dtype=float)
b = np.array([[2], [2]], dtype=float)
myBox = PB.projectionInBox(a, b)
x = np.array([[1], [1]], dtype=float)
d = 4*np.array([[-1.99], [0]], dtype=float)
sigma = 0.45
rho = 0.75
t = PS.projectedBacktrackingSearch(myObjective, myBox, x, d, sigma, rho, 1)
te = 0.125
if t == te:
    print('Check 01 okay')
else:
    raise Exception('Your projected backtracking search is not backtracking correctly.')


A = -0.01*np.eye(3)
B = np.array([[0], [0], [0]], dtype=float)
C = 1
a = np.array([[1], [1], [1]], dtype=float)
b = np.array([[40], [30], [20]], dtype=float)
myBox = PB.projectionInBox(a, b)
myObjective = BO.boxObjective(A, B, C, a, b)
x = np.array([[1.0], [1.0], [3.0]], dtype=float)
d = np.array([[1.0], [1.0], [1.0]], dtype=float)
sigma = 1.0e-3
rho = 1.0e-2
t = PS.projectedBacktrackingSearch(myObjective, myBox, x, d, sigma, rho, 1)
te = 24
if t == te:
    print('Check 02 okay')
else:
    raise Exception('Your projected backtracking search is not fronttracking correctly.')

x = np.array([[0.0]])
d = np.array([[1.0]])
a = np.array([[-100], [-100]], dtype=float)
b = np.array([[100], [100]], dtype=float)
myBox = PB.projectionInBox(a, b)
myObjective = FO.flatObjective()
sigma = 49/100
rho = 51/100
t = PS.projectedBacktrackingSearch(myObjective, myBox, x, d, sigma, rho, 1)
te = 6
if t == te:
    print('Check 03 okay')
else:
    raise Exception('Your projected backtracking search is not refining correctly.')

A = -np.eye(3)
B = np.array([[-1.5], [-1.5], [-1.5]], dtype=float)
C = 1
a = np.array([[1], [1], [1]], dtype=float)
b = np.array([[2], [3], [4]], dtype=float)
myBox = PB.projectionInBox(a, b)
myObjective = BO.boxObjective(A, B, C, a, b)
x0 = np.array([[1], [1], [3]], dtype=float)
eps = 1.0e-3
xmin = PBD.projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
xe = np.array([[2], [3], [4]], dtype=float)
if np.linalg.norm(xmin - xe) < 1.0e-2:
    print('Check 04 is okay')
else:
    raise Exception('Your projectedBFGSDescent is not working correctly for a simple example.')

myObjective = NO.noHessianObjective()
x0 = np.array([[0.15], [2.0]], dtype=float)
eps = 1.0e-3
a = np.array([[2], [-4]])
b = np.array([[4], [4]])
myBox = PB.projectionInBox(a, b)
xmin = PBD.projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
xe = np.array([[2.0], [0.0]], dtype=float)
if np.linalg.norm(xmin-xe) < 1.0e-2:
    print('Check 05 is okay')
else:
    raise Exception('Your projectedBFGSDescent is not working for Hessian free objective class.')

A = -0.3*np.eye(3)
B = np.array([[0], [0], [0]], dtype=float)
C = 1
a = np.array([[1], [1], [1]], dtype=float)
b = np.array([[20], [30], [20]], dtype=float)
myBox = PB.projectionInBox(a, b)
myObjective = BO.boxObjective(A, B, C, a, b)
x0 = np.array([[1], [1], [3]], dtype=float)
eps = 1.0e-4
xmin = PBD.projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
xe = np.array([[20], [30], [20]], dtype=float)
if np.linalg.norm(xmin - xe) < 1.0e-2:
    print('Check 06 is okay')
else:
    raise Exception('Your projectedBFGSDescent is not detecting the curvature failure correctly.')


myObjective = MO.multidimensionalObjective()
a = np.array([[1], [1], [1], [1], [-1], [-1], [-1], [-1]], dtype=float)
b = np.array([[2], [2], [2], [2], [2], [2], [2], [2]], dtype=float)
myBox = PB.projectionInBox(a, b)
x0 = np.array([[1], [1], [1], [1], [2], [2], [2], [2]], dtype=float)
eps = 1.0e-6
xmin = PBD.projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
xe = np.array([[1], [1], [1], [1], [-0.40749], [0.01116], [0.04147], [-0.01356]], dtype=float)
if np.linalg.norm(xmin-xe) < 1.0e-2:
    print('Check 07 is okay')
else:
    raise Exception('Your projectedBFGSDescent is not working for higher dimensions.')

if PS.matrnr() == 0:
    raise Exception('Please set your matriculation number in projectedBacktrackingSearch.py!')
elif PBD.matrnr() == 0:
    raise Exception('Please set your matriculation number in projectedBFGSDescent.py!')
else:
    print('Check completed. DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE OF CODE YOU WRITE.')

print('\nWe finished now projectedBFGSDescent, which is a q-superlinear descent algorithm for box constraints, that converges globally for unconstrained problems.')
print('It does not require Hessian information, but a linear system solver like CG.')
print('We will implement an algorithm for finding feasible points in the next LAB.')

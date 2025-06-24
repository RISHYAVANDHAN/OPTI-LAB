# Optimization for Engineers - Dr.Johannes Hild
# Programming Homework Check Script
# Do not change this file

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your programming homework is not working correctly.')

import numpy as np
import simpleValleyObjective as SO
import noHessianObjective as NO
import bananaValleyObjective as BO
import WolfePowellSearch as WP
import multidimensionalObjective as MO
import inexactNewtonCG as ICG
import flatObjective as FO
import quadraticObjective as QO


p = np.array([[0], [1]])
myObjective = SO.simpleValleyObjective(p)
x = np.array([[-1.01], [1]])
d = np.array([[1], [1]])
sigma = 1.0e-3
rho = 1.0e-2
t = WP.WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
te = 1
if t == te:
    print('Check 01 okay')
else:
    raise Exception('Your Wolfe-Powell search is not recognizing t = 1 as valid starting point.')

p = np.array([[0], [1]])
myObjective = SO.simpleValleyObjective(p)
x = np.array([[-1.2], [1]])
d = np.array([[0.1], [1]])
sigma = 1.0e-3
rho = 1.0e-2
t = WP.WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
te = 16
if t == te:
    print('Check 02 okay')
else:
    raise Exception('Your Wolfe-Powell search is not front tracking correctly.')

myObjective = FO.flatObjective()
x = np.array([[0.0]])
d = np.array([[1.0]])
sigma = 49/100
rho = 51/100
t = WP.WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
te = 6
if t == te:
    print('Check 03 okay')
else:
    raise Exception('Your Wolfe-Powell search is not refining correctly.')

myObjective = MO.multidimensionalObjective()
x = np.array([[1], [1], [1], [1], [1], [1], [1], [1]])
d = -myObjective.gradient(x)
sigma = 1.0e-3
rho = 1.0e-2
t = WP.WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
te = 0.0625
if np.abs(t-te) < 1.0e-3:
    print('Check 04 okay')
else:
    raise Exception('Your Wolfe-Powell search is not working for multidimensional objective.')

myObjective = MO.multidimensionalObjective()
x0 = np.array([[1], [1], [1], [1], [1], [1], [1], [1]])
xmin = ICG.inexactNewtonCG(myObjective, x0, 1.0e-6, 1)
xe = np.array([[1.02614], [0], [0], [0], [0], [0], [0], [0]])
if np.linalg.norm(xmin - xe) < 1.0e-2:
    print('Check 05 okay')
else:
    raise Exception('Your inexact Newton-CG does not work for the 8-dimensional test function')

myObjective = NO.noHessianObjective()
x0 = np.array([[-0.01], [0.01]])
xmin = ICG.inexactNewtonCG(myObjective, x0, 1.0e-6, 1)
xe = np.array([[0.26], [-0.21]])
if np.linalg.norm(xmin - xe) < 1.0e-2:
    print('Check 06 okay')
else:
    raise Exception('Your inexact Newton-CG is not working correctly for the Hessian free test function.')

myObjective = BO.bananaValleyObjective()
x0 = np.array([[0], [0]], dtype=float)
xmin = ICG.inexactNewtonCG(myObjective, x0, 1.0e-6, 1)
xe = np.array([[1], [1]], dtype=float)

if np.linalg.norm(xmin - xe) < 1.0e-2:
    print('Check 07 okay')
else:
    raise Exception('Your inexact Newton-CG does not work for the banana valley objective')

A = np.array([[1, 0.5, 0.25, 0.125, 0.0625, 0, 0, 0], [0.5, 2, 1, 0.5, 0.25, 0.125, 0.0625, 0], [0.25, 1, 4, 2, 1, 0.5, 0.25, 0.125], [0.125, 0.5, 2, 8, 4, 2, 1, 0.5],[0.0625, 0.25, 1, 4, 16, 8, 4, 2],[0, 0.125, 0.5, 2, 8, 32, 16, 8],[0, 0.0625, 0.25, 1, 4, 16, 64, 32],[0, 0, 0.125, 0.5, 2, 8, 32, 128]], dtype=float)
b = np.array([[1], [0.5], [0.25], [0.125], [0.0625], [0.125], [0.25], [0.5]], dtype=float)
c = 0
myObjective = QO.quadraticObjective(A, b, c)
x0 = np.ones((8, 1))
xmin = ICG.inexactNewtonCG(myObjective, x0, 1.0e-6, 1)
xe = np.array([[-1], [-2.7e-05], [1.49e-05], [0], [2.23e-03], [-2.79e-03], [-1.67e-03], [-3.34e-03]], dtype=float)

if np.linalg.norm(xmin - xe) < 1.0e-2:
    print('Check 08 okay')
else:
    raise Exception('Your inexact Newton-CG does not work for the flat objective')

if WP.matrnr() == 0:
    raise Exception('Please set your matriculation number in WolfePowellSearch.py!')
elif ICG.matrnr() == 0:
    raise Exception('Please set your matriculation number in inexactNewtonCG.py!')
else:
    print('Check completed. DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE OF CODE YOU WRITE.')

print('\nWe finished now inexactNewtonCG, which is a q-superlinear descent algorithm, that converges globally for unconstrained problems.')
print('It does not require Hessian information or a linear system solver like CG.')
print('We will implement an algorithm that can handle box constraints with a projection in the next LAB.')
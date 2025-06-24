# Optimization for Engineers - Dr.Johannes Hild
# Programming Homework Check Script
# Do not change this file

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your programming homework is not working correctly.')

import numpy as np
import simpleValleyObjective as SO
import multidimensionalObjective as MO
import quadraticObjective as QO
import leastSquaresFeasiblePoint as LSFP
import levenbergMarquardtDescent as LMD

p0 = np.array([[2], [-1]], dtype=float)
myObjectives = np.array([SO.simpleValleyObjective(p0)], dtype=object)
myWeights = np.array([1], dtype=float)
myErrorVector = LSFP.leastSquaresFeasiblePoint(myObjectives, myWeights)
x0 = np.array([[0], [4]], dtype=float)
res = myErrorVector.residual(x0)
rese = np.array([[18]], dtype=float)
if np.linalg.norm(res-rese) < 1.0e-2:
    print('Check 01 is okay')
else:
    raise Exception('Your leastSquaresFeasiblePoint returns a wrong residual for one constraint')

res = myErrorVector.jacobian(x0)
rese = np.array([[0, 12]], dtype=float)
if np.linalg.norm(res - rese) < 1.0e-2:
    print('Check 02 is okay')
else:
    raise Exception('Your leastSquaresFeasiblePoint returns a wrong jacobian for one constraint')

A = 0.4*np.eye(8)
b = 0.7*np.ones((8, 1))
c = 1
myObjectives = np.array([MO.multidimensionalObjective(), QO.quadraticObjective(A, b, c)], dtype=object)
myWeights = np.array([1, 10], dtype=float)
myErrorVector = LSFP.leastSquaresFeasiblePoint(myObjectives, myWeights)
x0 = np.array([[3], [-1], [-1], [0], [-1], [-1], [0], [-1]], dtype=float)
res = myErrorVector.residual(x0)
rese = np.array([[40.01492], [24]], dtype=float)
if np.linalg.norm(res-rese) < 1.0e-4:
    print('Check 03 is okay')
else:
    raise Exception('Your leastSquaresFeasiblePoint returns a wrong residual for two higher dimensional constraints')

res = myErrorVector.jacobian(x0)
rese = np.array([[16, -7, -12, -8, -14, -14, -7, -11], [19, 3, 3, 7, 3, 3, 7, 3]], dtype=float)
if np.linalg.norm(res-rese) < 1.0e-2:
    print('Check 04 is okay')
else:
    raise Exception('Your leastSquaresFeasiblePoint returns a wrong jacobian for two higher dimensional constraints')

p0 = np.array([[2], [-1]], dtype=float)
myObjectives = np.array([SO.simpleValleyObjective(p0)], dtype=object)
myWeights = np.array([1], dtype=float)
myErrorVector = LSFP.leastSquaresFeasiblePoint(myObjectives, myWeights)
x0 = np.array([[0], [4]], dtype=float)
eps = 1.0e-6
alpha0 = 1.0e-8
beta = 1000
xFeasible = LMD.levenbergMarquardtDescent(myErrorVector, x0, eps, alpha0, beta, 1)
feasibleErrorVector = myErrorVector.residual(xFeasible)
if np.linalg.norm(feasibleErrorVector) < 1.0e-4:
    print('Check 05 is okay')
else:
    raise Exception('Your levenbergMarquardtDescent returns a wrong result for one constraint')

A = np.array([[1, 0.5, 0.25, 0.125, 0.0625, 0, 0, 0], [0.5, 2, 1, 0.5, 0.25, 0.125, 0.0625, 0], [0.25, 1, 4, 2, 1, 0.5, 0.25, 0.125], [0.125, 0.5, 2, 8, 4, 2, 1, 0.5],[0.0625, 0.25, 1, 4, 16, 8, 4, 2],[0, 0.125, 0.5, 2, 8, 32, 16, 8],[0, 0.0625, 0.25, 1, 4, 16, 64, 32],[0, 0, 0.125, 0.5, 2, 8, 32, 128]], dtype=float)
b = np.array([[1], [0.5], [0.25], [0.125], [0.0625], [0.125], [0.25], [0.5]], dtype=float)
c = -2
myObjectives = np.array([MO.multidimensionalObjective(), QO.quadraticObjective(A, b, c)], dtype=object)
myWeights = np.array([1, 100], dtype=float)
myErrorVector = LSFP.leastSquaresFeasiblePoint(myObjectives, myWeights)
x0 = np.array([[3], [-1], [-1], [0], [-1], [-1], [0], [-1]], dtype=float)
eps = 1.0e-6
alpha0 = 1.0e-8
beta = 1000
xFeasible = LMD.levenbergMarquardtDescent(myErrorVector, x0, eps, alpha0, beta, 1)
feasibleErrorVector = myErrorVector.residual(xFeasible)
if np.linalg.norm(feasibleErrorVector) < 1.0e-4:
    print('Check 06 is okay')
else:
    raise Exception('Your levenbergMarquardtDescent returns a wrong result for two higher dimensional constraints')

if LSFP.matrnr() == 0:
    raise Exception('Please set your matriculation number in leastSquaresModel.py!')
elif LMD.matrnr() == 0:
    raise Exception('Please set your matriculation number in levenbergMarquardtDescent.py!')
else:
    print('Check completed. DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE OF CODE YOU WRITE.')

print('\nWe finished now levenbergMarquardtDescent, which is a q-superlinear descent algorithm for least squares objectives.')
print('We use it to find a feasible point of a 8 dimensional constraint set.')
print('We will implement an descent algorithm to solve the noisy problem with gradient free methods in the next LAB.')
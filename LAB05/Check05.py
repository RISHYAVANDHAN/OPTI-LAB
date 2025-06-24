# Optimization for Engineers - Dr.Johannes Hild
# Programming Homework Check Script
# Do not change this file

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your programming homework is not working correctly.')

import numpy as np
import SUCSGradient as SUC
import noHessianObjective as NO
import multidimensionalObjective as MO
import noisyObjective as NOI
import projectionInBall as PB
import implicitFiltering as IF

myObjective = MO.multidimensionalObjective()
x = np.array([[1], [1], [1], [1], [1], [1], [1], [1]], dtype=float)
h = 1.0e-6
g = SUC.SUCSGradient(myObjective, x, h)
ge = myObjective.gradient(x)
if np.linalg.norm(g-ge) < 1.0e-3:
    print('Check 01 is okay')
else:
    raise Exception('Your SUCSimplexGradient returns a wrong gradient')

myObjective = NO.noHessianObjective()
x = np.array([[1], [1]], dtype=float)
h = 1.0e-6
g = SUC.SUCSGradient(myObjective, x, h)
ge = myObjective.gradient(x)
if np.linalg.norm(g-ge) < 1.0e-3:
    print('Check 02 is okay')
else:
    raise Exception('Your SUCSimplexGradient returns a wrong gradient')

myObjective = NO.noHessianObjective()
x = np.array([[0.261], [-0.209]], dtype=float)
h = 1.0e-2
g1 = SUC.SUCSStencilFailure(myObjective, x, h)
x = np.array([[1], [1]], dtype=float)
g2 = SUC.SUCSStencilFailure(myObjective, x, h)
if g1 and (not g2):
    print('Check 03 is okay')
else:
    raise Exception('Your SUCSimplexGradient returns a wrong stencil failure check')

origin = np.array([[0.2], [0.0]])
radius = 0.2
myProjection = PB.projectionInBall(origin, radius)
myObjective = NO.noHessianObjective()
x0 = np.array([[0.0], [0.0]], dtype=float)
eps = 1.0e-3
h = np.array([[1], [0.1], [0.01], [0.001], [0.0001], [0.00001]], dtype=float)
xmin = IF.implicitFiltering(myObjective, myProjection, x0, h, eps, 1)
xe = np.array([[0.257], [-0.192]], dtype=float)
if np.linalg.norm(xmin - xe) < 1.0e-2:
    print('Check 04 is okay')
else:
    raise Exception('Your implicitFiltering is not working correctly for the noise free function.')


origin = np.array([[0], [0], [0], [0], [0], [0], [0], [0]], dtype=float)
radius = 0.3
myProjection = PB.projectionInBall(origin, radius)
myObjective = MO.multidimensionalObjective()
x0 = np.array([[1],[1],[1],[1],[1],[1],[1],[1]], dtype=float)
h = np.array([[1], [0.1], [0.01], [0.001], [0.0001], [0.00001]], dtype=float)
eps = 1.0e-3
xmin = IF.implicitFiltering(myObjective, myProjection, x0, h, 1.0e-1, 1)
xe = np.array([[0.290], [0.072], [0.020], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype=float)
if np.linalg.norm(xmin-xe) < 1.0e-2:
    print('Check 05 is okay')
else:
    raise Exception('Your implicitFiltering is not working for noise free higher dimensions.')

origin = np.array([[0], [0], [0], [0], [0], [0], [0], [0]], dtype=float)
radius = 0.3
myProjection = PB.projectionInBall(origin, radius)
myObjective = NOI.noisyObjective()
x0 = np.array([[10],[1],[1],[1],[1],[1],[1],[1]], dtype=float)
h = np.array([[1], [0.1]], dtype=float)
xmin = IF.implicitFiltering(myObjective, myProjection, x0, h, 1.0e-1, 1)
xe = np.array([[0.290], [0.072], [0.020], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype=float)
if np.linalg.norm(xmin-xe) < 5.0e-2:
    print('Check 06 is okay.')
else:
    raise Exception('Your implicitFiltering returns a wrong LMP for the noisy problem. ')

if SUC.matrnr() == 0:
    raise Exception('Please set your matriculation number in SUCSimplexGradient.py!')
elif IF.matrnr() == 0:
    raise Exception('Please set your matriculation number in implicitFiltering.py!')
else:
    print('Check completed.')

print('\nWe finished now SUCSimplexGradient and implicitFiltering, which can solve noisy objectives subject to convex feasible sets.')
print('We use it to solve the noisy problem.')
print('The (LMP) you found for the noisy problem is:\n', xmin)
print('Congratulations! You finished the LAB.')



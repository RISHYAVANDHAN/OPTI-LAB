# Optimization for Engineers - Dr.Johannes Hild
# projection in ball constraints

# Purpose: if x is outside the ball ||x-origin|| <= radius, then it is projected to xp = origin + radius/||x-origin||*(x-origin)

# Class parameters:
# origin: column vector in R ** n.
# radius: positive scalar, radius of the ball

# Input Definition:
# x: column vector in R ** n(domain space)

# Output Definition:
# projectedX: column vector in R ** n, satisfies constraint

# Required files:
# < none >

# Test cases:
# origin = np.array([[1], [1]], dtype=float)
# radius = 2
# eps = 0.01
# myBall = projectionInBall(origin, radius, eps)
# x = np.array([[4], [4]], dtype=float)
# myBall.project(x) should return [[2.414], [2.414]]
# y = np.array([[2.2], [2.2]], dtype=float)
# myBall.project(y) should return [[2.2], [2.2]]

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 0
    return matrnr


class projectionInBall:
  
    def __init__(self, origin: np.array, radius):
        self.origin = origin
        self.radius = radius
        if radius <= 0:
            raise TypeError('radius is not positive.')
        
    def project(self, x: np.array):

        myNorm = np.linalg.norm(x-self.origin)

        if myNorm <= self.radius:
            projectedX = x
        else:
            projectedX = self.origin + self.radius / myNorm * (x - self.origin)

        return projectedX
    


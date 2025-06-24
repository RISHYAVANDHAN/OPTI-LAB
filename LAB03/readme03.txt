---
Optimization for Engineers - Dr.Johannes Hild
LAB03
---

---
Files
---
projectedBacktrackingSearch: Line search method for projection methods, works similar to Wolfe-Powell, needs to be completed.
projectedBFGSDescent: Descent method for box constraints with global q-superlinear convergence rate. Does not require Hessian information, requires PrecCGSolver. Needs to be completed.
projectionInBox: Provides projection into boxes and active index sets.
boxObjective: Test problem that is not defined outside a box. If you get an error from this file, you probably miss a projection.
Check03: Run this to check your files for correctness, requires files from previous LABs.

---
Tasks
---
1. Complete projectedBacktrackingSearch.py and projectedBFGSDescent.py and check them with Check03.py for correctness.
2. The script needs to run without error messages AND the number of iterations of ALL checks is smaller than 25.
3. You need to add a short comment for each line of code you write. In each line you need to explain in your own words what the line does.
4. Upload projectedBacktrackingSearch.py and projectedBFGSDescent.py within the deadline.

---
Hints
---
For projectedBacktrackingSearch implement Algorithm 4.17 on script page 64. 
P is already provided and can project(), you can look in the file projectionInBox.py to see how it works.
For projectedBFGSDescent you have to implement Alg. 11.8 on script page 128.
Make sure that you use the original (BFGS) update formula and that a change in x leads to a proper change in all variables depending on x.
Use numpy.eye(n) to construct a identity matrix, use @ for matrix - vector or matrix - matrix product, use .T to transpose
Use x.shape[0] to get the vector dimension, activeIndexSet() is already provided by the projection.
A call like H[A, :] = eye(n)[A, :] replaces all columns with indexes in A with the corresponding columns of eye(n).
Together with a similar call to replace all rows, you can reduce the BFGS matrix by overwriting cleverly with the identity.
Make sure that you never evaluate the objective or its derivatives on unprojected points and do not forget the reduction of the BGFS matrix.
For all checks you should have fewer than 25 iterations, otherwise you use steepest-descent-like steps too often and end up with a poor q-linear algorithm.

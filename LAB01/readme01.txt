---
Optimization for Engineers - Dr.Johannes Hild
LAB01
---

---
Files
---
PrecCGSolver: Highly effective linear system solver, needs to be completed.
NewtonDescent: Descent method with local q-quadratic convergence rate, but has its issues. Needs to be completed.
bananaValleyObjective: Test problem with vanishing Hessian information.
quadraticObjective: Testproblem with a hill point not bounded from below.
Check01: Run this to check your files for correctness, requires files from previous LABs.

---
Tasks
---
1. Complete PrecCGSolver.py and NewtonDescent.py and check them with Check01.py for correctness.
2. The script needs to run without error messages AND the number of iterations of the last check must be smaller than 30.
If you have 0 iterations, then you forgot to count them in the loop!
3. You need to add a short comment for each line of code you write. In each line you need to explain in your own words what the line does.
4. Upload PrecCGSolver.py and NewtonDescent.py within the deadline.

---
Hints
---
For PrecCGSolver see Algorithm 11.1. on script page 125, use incompleteCholesky.py and LLTSolver.py from LAB00
For NewtonDescent see Algorithm 4.13 on script page 59, with the following modifications: 
In step 3a) Bk is always set to objective.hessian(xk) and use PrecCGSolver to solve Bk*dk=-gradfk
Ignore step 3b) and just set tk=1.

Consult the tutorials for python and numpy
Use numpy.sqrt for sqrt and numpy.abs for abs and numpy.linalg.norm for norms.
Python starts indexes with [0] and not with [1], this is important for the use of range()
Python uses "call by assignment", sometimes you must copy a vector or matrix with .copy()
Multiplication of matrices and vectors is @ (and not *) and use .T to transpose
In your loops count the iterations with countIter = countIter + 1

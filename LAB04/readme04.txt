---
Optimization for Engineers - Dr.Johannes Hild
LAB04
---

---
Files
---
leastSquaresFeasiblePoint: Constructs an objective for Levenberg-Marquardt that demands all equality constraints to be satisfied, needs to be completed.
levenbergMarquardtDescent: Descent method for least squares objectives with global q-superlinear convergence rate. Needs to be completed.
Check04: Run this to check your files for correctness, requires files from previous LABs.

---
Tasks
---
1. Complete leastSquaresFeasiblePoint.py and levenbergMarquardtDescent.py and check them with Check04.py for correctness.
2. The script needs to run without error messages AND the number of iterations of the last check must be smaller than 20.
3. You need to add a short comment for each line of code you write. In each line you need to explain in your own words what the line does.
4. Upload leastSquaresFeasiblePoint.py and levenbergMarquardtDescent.py within the deadline.

---
Hints
---
For leastSquaresFeasiblePoint you need to complete the code that generates the residual and jacobian for the least squares objective of the following type:
f(x) = 0.5*sum_k (p[k]h[k](x))**2, whereas f(x) is the least squares objective and h[k] are equality constraints collected in the array hArray with positive weights p[k] collected in another array.
Remember that each row of the Jacobian is a transposed gradient of the residual component, you will need .gradient() to call these.
For levenbergMarquardtDescent implement Algorithm 6.14 on script page 91.
You will need PrecCGSolver from previous LABs.
Remember the connection f = 1/2 R'*R and grad_f = J'*R.

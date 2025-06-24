---
Optimization for Engineers - Dr.Johannes Hild
LAB02
---

---
Files
---
WolfePowellSearch: Highly effective line search method, needs to be completed.
inexactNewtonCG: Descent method with global q-superlinear convergence rate. Does not require Hessian information or a linear system solver. Needs to be completed.
simpleValleyObjective: Test problem for Wolfe-Powell.
directionalHessApprox: Provides directional Hessian approximations, i.e. Algorithm 11.5.
flatObjective: Test problem for Wolfe-Powell.
noHessianObjective: Test problem without Hessian information.
Check02: Run this to check your files for correctness, requires files from previous LABs.

---
Tasks
---
1. Complete WolfePowellSearch.py and inexactNewtonCG.py and check them with Check02.py for correctness.
2. The script needs to run without error messages AND the number of iterations of the last two checks must be smaller than 30.
3. You need to add a short comment for each line of code you write. In each line you need to explain in your own words what the line does.
4. Upload WolfePowellSearch.py and inexactNewtonCG.py within the deadline.

---
Hints
---
For Wolfe-Powell see Algorithm 4.10 on script page 56
Define inner functions to check the Wolfe-Powell rules
Make sure that a change in t leads to a proper change in all variables depending on t.
MultidimensionalObjective is in LAB00.
For inexactNewtonCG implement Algorithm 6.3 on script page 80. You also need Algorithm 11.5., which is already implemented in directionalHessApprox.
Use numpy.sqrt and numpy.min([,]) for eta_k.
Be careful: numpy arrays are not copied for assignments like "x = x_j".
A statement like  "x_j += td" would then also change "x", even if not intended.
You can avoid this by using ".copy()" and not using "+=".
For the last two checks you should have fewer than 30 iterations, otherwise you use steepest-descent-like steps too often and end up with a poor q-linear algorithm.

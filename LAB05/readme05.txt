---
Optimization for Engineers - Dr.Johannes Hild
LAB05
---

---
Files
---
SUCSGradient: Approximates gradient of f on a scaled unit central simplex, needs to be completed.
implicitFiltering: Inner and outer loop with projected steepest descent update to find the LMP at all scales of a noisy objective. Needs to be completed.
noisyObjective: Test problem in 8 dimensions with noise.
projectionInBall: Projection into ball constraint.
Check05: Run this to check your files for correctness, requires files from previous LABs.


---
Tasks
---
Complete SUCSGradient.py and implicitFiltering.py and check them with Check05.py for correctness.
The script needs to run without error messages. The number of iterations for the outer loops should be below 50.
Upload SUCSGradient.py and implicitFiltering.py within the deadline.

---
Hints
---
For SUCSGradient implement the scaled unit central simplex gradient from Def. 8.8 in script page 104 including the stencil failure check.
use x.shape[0] to get the vector dimension and use a .copy() of a vector if required.
In implicitFiltering you need to complete the outer loop, which is Algorithm 8.11 on page 105.
At the beginning of that loop, x_b is not a minimum at all scales by default.
To get the solution of minimizing f in step 3a)i) you can call SUCSProjectedSteepestDescent(x,h).



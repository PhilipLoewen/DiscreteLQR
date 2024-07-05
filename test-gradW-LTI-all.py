# 2024-07-04 [PDL]

# Consider a time-varying LQR setup with simplified dynamics
#    x[t+1] = A[t] x[t] + B[t] u[t]
# and jointly convex quadratic objective
#    J = sum(  x[t]' * Q[t] x[t] + u[t]' * R[t] u[t] ).
# Given a start point x_0 and a final time T>0,
# we are interested in the gradients of the minimum value of J
# with respect to all elements of the problem data: think of this
# as defining a "value function"
#    W=W(A,B,Q,R,x0).

# We calculate all possible gradients of W using both
# finite differences and forward-backward theory,
# then check the results for consistency

import copy
import numpy as np
import scipy as sp
import DiscreteLQR as mylqr
import PrettyPrinter as ppm
from TensorGradient import grad
import sys

np.random.seed(320101)  # Use this to make the "random" problem reproducible

######################################################################
# Set up the structure parameters for the LQR system.
# Manually adjust these to run independent tests.

n = 4  # State vector dimension
m = 3  # Input/control vector dimension
T = 4  # Literal final time, interesting subscripts are 0,...,T

# Pick an initial state at random:
x0 = np.around(np.random.rand(n, 1), decimals=1).reshape(n, 1)

# Pick a number from 0 to 5 inclusive to control diagnostic output:
printlevel = 0


def diagprint(q, string):
    if q <= printlevel:
        print(string)


######################################################################
# Instantiate a random LTI system with the parameters above.
# (This mentions some elements not anticipated in the intro.)

# Dynamic shift and linear cost terms
f0 = np.random.rand(n, 1)  # same columns for all indices
c0 = np.random.rand(n + m, 1)  # same columns for all indices
f = f0
c = c0

# Main dynamic matrices are just random constants
F0 = np.random.rand(n, n + m) * 10

# Each cost coefficient matrix must be symmetric,
# with a positive-definite uu-block.
# (Scale  and shift to give the diagonal entries expected value 1.5.)
Q0r = np.random.rand(n, n)
Q0 = Q0r.T @ Q0r * 3.0 / n + 0.5 * np.ones((n, n))
R0r = np.random.rand(m, m)
R0 = R0r.T @ R0r * 3.0 / m + 0.5 * np.ones(R0r.shape)
C0 = np.block(
    [
        [np.around(Q0, decimals=2), np.zeros((n, m))],
        [np.zeros((m, n)), np.around(R0, decimals=2)],
    ]
)


#######################################################################
# Invent coefficients for a suitable linear function W=W(C,F,x0).
#######################################################################
wx = np.random.rand(n, 1, T + 1) * 10
wu = np.random.rand(m, 1, T) * 10

ppm.ppm(wx, "wx, giving coefficients for x,")
ppm.ppm(wu, "wu, giving coefficients for u,")


#######################################################################
# Utility function packs x, u, lambda into a KKT-compatible column
#######################################################################
def packxul(traj_x, traj_u, traj_lam):
    T = traj_u.shape[2]
    # First pile x's atop u's and pad bottom with 0's:
    midpart = np.vstack((traj_x[:, 0, 1:T], traj_u[:, 0, 1:T], traj_lam[:, 0, 1:T]))
    # Next stack the columns on top of each other, working left to right
    corecol = midpart.reshape(
        ((T - 1) * (m + n + n), 1), order="F"
    )  # What a minefield.
    # Finally stitch on the short pieces for the top and bottom.
    result = np.vstack(
        (traj_u[:, [0], 0], traj_lam[:, [0], 0], corecol, traj_x[:, [0], T])
    )
    return result


#######################################################################
# Build nominal system and print its key ingredients
#######################################################################
print(f"Sample system has n={n}, m={m}, and T={T}.")
ppm.ppm(x0, "x0, the initial state,")
print(" ")

sys0 = mylqr.DiscreteLQR(C0, c0, F0, f0, T=T)

print("\nSystem construction is complete. Calling printself gives this:")
sys0.printself()

#######################################################################
# Set up to record summary stats.
#######################################################################

tolvals = []
tolstrs = []

if True:
    #######################################################################
    # Gradient of W w.r.t. initial point x0 -- numerical evaluation by finite difference
    #######################################################################
    print(f"\n{10*'*':s} Sensitivity of W to x0 {10*'*':s}")

    def W(x):
        # Input is a tall skinny matrix, (n)x(1)
        bestx, bestu, bestlam = sys0.bestxul(x)
        return np.tensordot(wx[:, 0, :], bestx[:, 0, :]) + np.tensordot(
            wu[:, 0, :], bestu[:, 0, :]
        )

    print(f"Nominal value W(x0) = {W(x0):13.6e}.")

    dWdx0auto = grad(W, x0)

    #######################################################################
    # Gradient of W w.r.t. initial point x0 -- theoretical evaluation from KKT system
    #######################################################################

    dWdx0module = sys0.gradx0W(wx, wu, x0)

    #######################################################################
    # Gradient of W w.r.t. initial point x0 -- report
    #######################################################################

    ppm.ppm(dWdx0auto, f"dWdx0auto, showing automatic finite differences,", sigfigs=7)
    ppm.ppm(
        dWdx0module, f"dWdx0module, showing theoretical findings in module,", sigfigs=7
    )
    ppm.ppm(
        dWdx0module - dWdx0auto, "difference from theory to approximation", sigfigs=2
    )

    tolvals.append(np.linalg.norm(dWdx0module - dWdx0auto) / np.linalg.norm(dWdx0auto))
    tolstrs.append("gradient of W w.r.t. x0")
else:
    print(f"(Skipping sensitivity calculations for x0.)")

if True:
    #######################################################################
    # Gradient of W w.r.t. matrix C -- numerical evaluation by finite difference
    #######################################################################
    print(f"\n{10*'*':s} Sensitivity of W to C {10*'*':s}")

    def W(C):
        # Input is a square symmetric matrix, (n+m)x(n+m)
        mysys = mylqr.DiscreteLQR(C, c0, F0, f0, T=T)
        bestx, bestu, bestlam = mysys.bestxul(x0)
        return np.tensordot(wx[:, 0, :], bestx[:, 0, :]) + np.tensordot(
            wu[:, 0, :], bestu[:, 0, :]
        )

    print(f"Nominal value W(C0) = {W(C0):13.6e}.")

    dWdCtauto = grad(W, C0)

    #######################################################################
    # Gradient of W w.r.t. matrix C -- theoretical evaluation from KKT system
    #######################################################################

    dWdCtmodule = sys0.gradCtW(0, wx, wu, x0)

    #######################################################################
    # Gradient of W w.r.t. matrix C -- report
    #######################################################################

    ppm.ppm(
        dWdCtauto, f"dWdCtauto, showing finite differences done in script,", sigfigs=7
    )
    ppm.ppm(
        dWdCtmodule, f"dWdCtmodule, showing theoretical findings in module,", sigfigs=7
    )
    ppm.ppm(
        dWdCtmodule - dWdCtauto, "difference from theory to approximation", sigfigs=2
    )

    tolvals.append(np.linalg.norm(dWdCtmodule - dWdCtauto) / np.linalg.norm(dWdCtauto))
    tolstrs.append(f"gradient of W w.r.t.  C")

    #######################################################################
    # Gradient of W w.r.t. vector c -- numerical evaluation by finite difference
    #######################################################################
    print(f"\n{10*'*':s} Sensitivity of W to c {10*'*':s}")

    def W(c):
        # Input is a tall skinny matrix, (n+m)x(1)
        mysys = mylqr.DiscreteLQR(C0, c, F0, f, T=T)
        bestx, bestu, bestlam = mysys.bestxul(x0)
        return np.tensordot(wx[:, 0, :], bestx[:, 0, :]) + np.tensordot(
            wu[:, 0, :], bestu[:, 0, :]
        )

    print(f"Nominal value W(c0) = {W(c0):13.6e}.")

    dWdctauto = grad(W, c0)

    #######################################################################
    # Gradient of W w.r.t. vector ct -- theoretical evaluation from KKT system
    #######################################################################

    dWdctmodule = sys0.gradctW(0, wx, wu, x0)

    #######################################################################
    # Gradient of W w.r.t. vector c -- report
    #######################################################################

    ppm.ppm(dWdctauto, f"dWdctauto, showing automatic finite differences,", sigfigs=7)
    ppm.ppm(
        dWdctmodule, f"dWdctmodule, showing theoretical findings in module", sigfigs=7
    )
    ppm.ppm(
        dWdctmodule - dWdctauto, "difference from theory to approximation", sigfigs=2
    )

    tolvals.append(np.linalg.norm(dWdctmodule - dWdctauto) / np.linalg.norm(dWdctauto))
    tolstrs.append(f"gradient of W w.r.t.  c")

    #######################################################################
    # Gradient of W w.r.t. matrix F -- finite difference approach
    #######################################################################
    print(f"\n{10*'*':s} Sensitivity of W to F {10*'*':s}")

    def W(F):
        # Input is a square symmetric matrix, (n+m)x(n+m)
        mysys = mylqr.DiscreteLQR(C0, c0, F, f0, T=T)
        bestx, bestu, bestlam = mysys.bestxul(x0)
        return np.tensordot(wx[:, 0, :], bestx[:, 0, :]) + np.tensordot(
            wu[:, 0, :], bestu[:, 0, :]
        )

    print(f"Nominal value W(F0) = {W(F0):13.6e}.")

    dWdFtauto = grad(W, F0)

    #######################################################################
    # Gradient of W w.r.t. matrix F -- theoretical prediction from system built-in
    #######################################################################

    dWdFtmodule = sys0.gradFtW(0, wx, wu, x0)

    #######################################################################
    # Gradient of W w.r.t. matrix F -- report
    #######################################################################

    ppm.ppm(dWdFtauto, f"dWdFtauto, from automated FD calculations,", sigfigs=7)
    ppm.ppm(
        dWdFtmodule, f"dWdFtmodule, showing theoretical findings in module,", sigfigs=7
    )
    ppm.ppm(
        dWdFtmodule - dWdFtauto,
        "difference from theory to auto-FD approximation,",
        sigfigs=2,
    )

    tolvals.append(np.linalg.norm(dWdFtmodule - dWdFtauto) / np.linalg.norm(dWdFtauto))
    tolstrs.append(f"gradient of W w.r.t.  F")

    #######################################################################
    # Gradient of W w.r.t. vector f -- numerical evaluation by finite difference
    #######################################################################
    print(f"\n{10*'*':s} Sensitivity of W to f {10*'*':s}")

    def W(f):
        # Input is a tall skinny matrix, (n)x(1)
        mysys = mylqr.DiscreteLQR(C0, c0, F0, f, T=T)
        bestx, bestu, bestlam = mysys.bestxul(x0)
        return np.tensordot(wx[:, 0, :], bestx[:, 0, :]) + np.tensordot(
            wu[:, 0, :], bestu[:, 0, :]
        )

    print(f"Nominal value W(f0) = {W(f0):13.6e}.")

    dWdftauto = grad(W, f0)

    #######################################################################
    # Gradient of W w.r.t. vector f -- theoretical evaluation from KKT system
    #######################################################################

    dWdftmodule = sys0.gradftW(0, wx, wu, x0)

    #######################################################################
    # Gradient of W w.r.t. vector f -- report
    #######################################################################

    ppm.ppm(dWdftauto, f"dWdftauto, showing automatic finite differences,", sigfigs=7)
    ppm.ppm(
        dWdftmodule, f"dWdftmodule, showing theoretical findings in module,", sigfigs=7
    )
    ppm.ppm(
        dWdftmodule - dWdftauto, "difference from theory to approximation,", sigfigs=2
    )

    tolvals.append(np.linalg.norm(dWdftmodule - dWdftauto) / np.linalg.norm(dWdftauto))
    tolstrs.append(f"gradient of W w.r.t.  f")

#######################################################################
# Ultimate final report
#######################################################################

print(" ")
print(72 * "*")  # 72 columns in honour of Fortran on punch cards
print(f"Ultimate report after {len(tolvals)} sensitivity calculations:")
print(72 * "*")  # 72 columns in honour of Fortran on punch cards

worstindex = -1
worsterror = 0
for j in range(len(tolvals)):
    print(f"Relative error in {tolstrs[j]} was {tolvals[j]:6.1e}.")
    if tolvals[j] > worsterror:
        worsterror = tolvals[j]
        worstindex = j
print(f"Largest discrepancy was {worsterror:6.1e}, from the {tolstrs[worstindex]}.")
myspec = 1e-5
if worsterror < myspec:
    print(f"That meets my acceptability criterion of {myspec:6.1e}.")
else:
    print(f"Apparently further scrutiny is required.")
print(72 * "*")
print(" ")

# 2024-07-03 [PDL]

# Consider a time-varying LQR setup with simplified dynamics
#    x[t+1] = A[t] x[t] + B[t] u[t]
# and jointly convex quadratic objective
#    J = sum(  x[t]' * Q[t] x[t] + u[t]' * R[t] u[t] ).
# Given a start point x_0 and a final time T>0,
# we are interested in the gradients of a general linear function
# defined in terms of the optimal trajectory:
#    W(A,B,Q,R,x0) = dot(wx,bestx) + dot(wu,bestu)
# Here the constant coefficients wx and wu are given in the setup,
# with shapes compatible with the indicated dot-product operations.

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

n = 3  # State vector dimension
m = 2  # Input/control vector dimension
T = 3  # Literal final time, interesting subscripts are 0,...,T

# Pick an initial state at random:
x0 = np.around(np.random.rand(n, 1), decimals=1).reshape(n, 1)

# Pick a number from 0 to 5 inclusive to control diagnostic output:
printlevel = 0


def diagprint(q, string):
    if q <= printlevel:
        print(string)


######################################################################
# Instantiate a random system with the parameters above.
# (This mentions some elements not anticipated in the intro.)


# Dynamic shift and linear cost terms
f0 = np.random.rand(n, 1, T)  # columns for indices 0,1,...,T-1
c0 = np.random.rand(n + m, 1, T + 1)  # columns for t=0,1,...,T
f = f0
c = c0

# Main dynamic matrices are just random
F0 = np.random.rand(n, n + m, T) * 10

# Each cost coefficient matrix must be symmetric,
# with a positive-definite uu-block.
# (Scale  and shift to give the diagonal entries expected value 1.5.)
C0 = np.zeros((n + m, n + m, T + 1))
for t in range(T + 1):
    Q0r = np.random.rand(n, n)
    Q0 = Q0r.T @ Q0r * 3.0 / n + 0.5 * np.ones((n, n))
    R0r = np.random.rand(m, m)
    R0 = R0r.T @ R0r * 3.0 / m + 0.5 * np.ones(R0r.shape)
    C0[:, :, t] = np.block(
        [
            [np.around(Q0, decimals=2), np.zeros((n, m))],
            [np.zeros((m, n)), np.around(R0, decimals=2)],
        ]
    )
if False:
    # Cleanup of costs with indices t=0 and t=T
    # should be handled automatically in the module
    C0[0:n, 0:n, 0] = np.zeros((n, n))  # Cost is independent of x[0]
    C0[n : n + m, n : n + m, T] = np.zeros((m, m))  # Cost is independent of u[T]

if False:
    # Overwrite the time-varying setup above with
    # a setup that activates the autonomous case of
    # the system object. TODO - This breaks many things now!
    A0 = np.random.rand(n, n) * 10
    B0 = -np.random.rand(n, m) * 10
    F0 = np.hstack((A0, B0))
    C0 = C0[:, :, 0]
    c0 = np.random.rand(n + m, 1)
    c = c0
    f0 = np.random.rand(n, 1)
    f = f0

#######################################################################
# Build nominal system and print its key ingredients
#######################################################################
print(f"Sample system has n={n}, m={m}, and T={T}.")
ppm.ppm(x0, "x0, the initial state,")
print(" ")

sys0 = mylqr.DiscreteLQR(C0, c0, F0, f0, T=T)

print("\nSystem construction is complete. Here are some elements.")
print(f"sys0.autonomous: {sys0.autonomous}")

for t in range(T):
    ppm.ppm(sys0.F(t), f"F_{t}, according to sys0,")
    ppm.ppm(sys0.f(t), f"f_{t}, according to sys0,")
print(" ")

for t in range(T + 1):
    ppm.ppm(sys0.C(t), f"C_{t}, according to sys0,")
    ppm.ppm(sys0.c(t), f"c_{t}, according to sys0,")
print(" ")

bestx, bestu, bestlam = sys0.bestxul(x0)
# KKTsol0 = sys0.xul2kkt(bestx,bestu,bestlam)

#######################################################################
# Invent coefficients for a suitable linear function W=W(C,F,x0).
#######################################################################
wx = np.random.rand(*bestx.shape) * 10
wu = np.random.rand(*bestu.shape) * 10

ppm.ppm(wx, "wx, giving coefficients for x,")
ppm.ppm(wu, "wu, giving coefficients for u,")

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
        dWdx0module - dWdx0auto, "difference from theory to approximation,", sigfigs=2
    )

    tolvals.append(np.linalg.norm(dWdx0module - dWdx0auto))
    tolstrs.append("gradient of W w.r.t.  x0")
else:
    print(f"(Skipping sensitivity calculations for x0.)")

for t in range(T):
    #######################################################################
    # Gradient of W w.r.t. matrix Ct -- numerical evaluation by finite difference
    #######################################################################
    print(f"\n{10*'*':s} Sensitivity of W to C_{t} {10*'*':s}")

    def W(Ct):
        # Input is a square symmetric matrix, (n+m)x(n+m)
        Cmod = copy.deepcopy(C0)
        Cmod[:, :, t] = Ct
        mysys = mylqr.DiscreteLQR(Cmod, c, F0, f)
        bestx, bestu, bestlam = mysys.bestxul(x0)
        return np.tensordot(wx[:, 0, :], bestx[:, 0, :]) + np.tensordot(
            wu[:, 0, :], bestu[:, 0, :]
        )

    Ct0 = sys0.C(t)
    print(f"For t={t}, nominal value W(Ct) = {W(Ct0):13.6e}.")

    dWdCtauto = grad(W, Ct0)
    # ppm.ppm(dWdCtauto,f"grad W w.r.t. C_{t}",sigfigs=7)

    #######################################################################
    # Gradient of W w.r.t. matrix Ct -- theoretical evaluation from KKT system
    #######################################################################

    dWdCtmodule = sys0.gradCtW(t, wx, wu, x0)

    #######################################################################
    # Gradient of W w.r.t. matrix Ct -- report
    #######################################################################

    ppm.ppm(
        dWdCtauto,
        f"dWdCtauto, showing finite differences done in script, t={t},",
        sigfigs=7,
    )
    ppm.ppm(
        dWdCtmodule, f"dWdCtmodule, showing theoretical findings in module,", sigfigs=7
    )
    ppm.ppm(
        dWdCtmodule - dWdCtauto, "difference from theory to approximation,", sigfigs=2
    )

    tolvals.append(np.linalg.norm(dWdCtmodule - dWdCtauto))
    tolstrs.append(f"gradient of W w.r.t. C_{t}")

    #######################################################################
    # Gradient of W w.r.t. vector ct -- numerical evaluation by finite difference
    #######################################################################
    print(f"\n{10*'*':s} Sensitivity of W to c_{t} {10*'*':s}")

    def W(ct):
        # Input is a tall skinny matrix, (n+m)x(1)
        cmod = copy.deepcopy(c0)
        cmod[:, :, t] = ct
        mysys = mylqr.DiscreteLQR(C0, cmod, F0, f)
        bestx, bestu, bestlam = mysys.bestxul(x0)
        return np.tensordot(wx[:, 0, :], bestx[:, 0, :]) + np.tensordot(
            wu[:, 0, :], bestu[:, 0, :]
        )

    ct0 = sys0.c(t)
    print(f"For t={t}, nominal value W(ct) = {W(ct0):13.6e}.")

    dWdctauto = grad(W, ct0)

    #######################################################################
    # Gradient of W w.r.t. vector ct -- theoretical evaluation from KKT system
    #######################################################################

    dWdctmodule = sys0.gradctW(t, wx, wu, x0)

    #######################################################################
    # Gradient of W w.r.t. vector ct -- report
    #######################################################################

    ppm.ppm(
        dWdctauto, f"dWdctauto, showing automatic finite differences, t={t},", sigfigs=7
    )
    ppm.ppm(
        dWdctmodule, f"dWdctmodule, showing theoretical findings in module", sigfigs=7
    )
    ppm.ppm(
        dWdctmodule - dWdctauto, "difference from theory to approximation", sigfigs=2
    )

    tolvals.append(np.linalg.norm(dWdctmodule - dWdctauto))
    tolstrs.append(f"gradient of W w.r.t. c_{t}")

    if t == T:
        break

    #######################################################################
    # Gradient of W w.r.t. matrix Ft -- finite difference approach
    #######################################################################
    print(f"\n{10*'*':s} Sensitivity of W to F_{t} {10*'*':s}")

    def W(Ft):
        # Input is a square symmetric matrix, (n+m)x(n+m)
        Fmod = copy.deepcopy(F0)
        Fmod[:, :, t] = Ft
        mysys = mylqr.DiscreteLQR(C0, c, Fmod, f)
        bestx, bestu, bestlam = mysys.bestxul(x0)
        return np.tensordot(wx[:, 0, :], bestx[:, 0, :]) + np.tensordot(
            wu[:, 0, :], bestu[:, 0, :]
        )

    Ft0 = sys0.F(t)
    print(f"For t={t}, nominal value W(Ft) = {W(Ft0):13.6e}.")

    dWdFtauto = grad(W, Ft0)

    #######################################################################
    # Gradient of W w.r.t. matrix Ft -- theoretical prediction from system built-in
    #######################################################################

    dWdFtmodule = sys0.gradFtW(t, wx, wu, x0)

    #######################################################################
    # Gradient of W w.r.t. matrix Ft -- report
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

    tolvals.append(np.linalg.norm(dWdFtmodule - dWdFtauto))
    tolstrs.append(f"gradient of W w.r.t. F_{t}")

    #######################################################################
    # Gradient of W w.r.t. vector ft -- numerical evaluation by finite difference
    #######################################################################
    print(f"\n{10*'*':s} Sensitivity of W to f_{t} {10*'*':s}")

    def W(ft):
        # Input is a tall skinny matrix, (n)x(1)
        fmod = copy.deepcopy(f0)
        fmod[:, :, t] = ft
        mysys = mylqr.DiscreteLQR(C0, c0, F0, fmod)
        bestx, bestu, bestlam = mysys.bestxul(x0)
        return np.tensordot(wx[:, 0, :], bestx[:, 0, :]) + np.tensordot(
            wu[:, 0, :], bestu[:, 0, :]
        )

    ft0 = sys0.f(t)
    print(f"For t={t}, nominal value W(ft) = {W(ft0):13.6e}.")

    dWdftauto = grad(W, ft0)

    #######################################################################
    # Gradient of W w.r.t. vector ft -- theoretical evaluation from KKT system
    #######################################################################

    dWdftmodule = sys0.gradftW(t, wx, wu, x0)

    #######################################################################
    # Gradient of W w.r.t. vector ft -- report
    #######################################################################

    ppm.ppm(
        dWdftauto, f"dWdftauto, showing automatic finite differences, t={t},", sigfigs=7
    )
    ppm.ppm(
        dWdftmodule, f"dWdftmodule, showing theoretical findings in module,", sigfigs=7
    )
    ppm.ppm(
        dWdftmodule - dWdftauto, "difference from theory to approximation,", sigfigs=2
    )

    tolvals.append(np.linalg.norm(dWdftmodule - dWdftauto))
    tolstrs.append(f"gradient of W w.r.t. f_{t}")

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
    print(f"Discrepancy in {tolstrs[j]} was {tolvals[j]:6.1e}.")
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

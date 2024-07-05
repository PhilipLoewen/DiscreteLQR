# 2024-07-03 [PDL]

# Consider a time-varying LQR setup with simplified dynamics
#    x[t+1] = A[t] x[t] + B[t] u[t]
# and jointly convex quadratic objective
#    J = sum(  x[t]' * Q[t] x[t] + u[t]' * R[t] u[t] ).
# Given a start point x_0 and a final time T>0,
# we are interested in the gradients of the minimum value of J
# with respect to all elements of the problem data: think of this
# as defining a "value function"
#    V=V(A,B,Q,R,x0).

# We calculate all possible gradients of V using both
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
T = 4  # Literal final time, interesting subscripts are 0,...,T

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
# KKTsol0 = packxul(bestx,bestu,bestlam)

#######################################################################
# Set up to record summary stats.
#######################################################################

tolvals = []
tolstrs = []

if True:
    #######################################################################
    # Gradient of V w.r.t. initial point x0 -- numerical evaluation by finite difference
    #######################################################################
    print(f"\n{10*'*':s} Sensitivity of V to x0 {10*'*':s}")

    def V(x):
        # Input is a tall skinny matrix, (n)x(1)
        bestx, bestu, bestlam = sys0.bestxul(x)
        return sys0.J(bestx, bestu)

    print(f"Nominal value V(x0) = {V(x0):13.6e}.")

    dVdx0auto = grad(V, x0)

    #######################################################################
    # Gradient of V w.r.t. initial point x0 -- theoretical evaluation from KKT system
    #######################################################################

    dVdx0module = sys0.gradx0V(x0)

    #######################################################################
    # Gradient of V w.r.t. initial point x0 -- report
    #######################################################################

    ppm.ppm(dVdx0auto, f"dVdx0auto, showing automatic finite differences,", sigfigs=7)
    ppm.ppm(
        dVdx0module, f"dVdx0module, showing theoretical findings in module", sigfigs=7
    )
    ppm.ppm(
        dVdx0module - dVdx0auto, "difference from theory to approximation", sigfigs=2
    )

    tolvals.append(np.linalg.norm(dVdx0module - dVdx0auto))
    tolstrs.append("gradient of V w.r.t.  x0")
else:
    print(f"(Skipping sensitivity calculations for x0.)")

for t in range(T):
    #######################################################################
    # Gradient of V w.r.t. matrix Ct -- numerical evaluation by finite difference
    #######################################################################
    print(f"\n{10*'*':s} Sensitivity of V to C_{t} {10*'*':s}")

    def V(Ct):
        # Input is a square symmetric matrix, (n+m)x(n+m)
        Cmod = copy.deepcopy(C0)
        Cmod[:, :, t] = Ct
        mysys = mylqr.DiscreteLQR(Cmod, c, F0, f)
        bestx, bestu, bestlam = mysys.bestxul(x0)
        return mysys.J(bestx, bestu)

    Ct0 = sys0.C(t)
    print(f"For t={t}, nominal value V(Ct) = {V(Ct0):13.6e}.")

    dVdCtauto = grad(V, Ct0)
    # ppm.ppm(dVdCtauto,f"grad V w.r.t. C_{t}",sigfigs=7)

    #######################################################################
    # Gradient of V w.r.t. matrix Ct -- theoretical evaluation from KKT system
    #######################################################################

    dVdCtmodule = sys0.gradCtV(t, x0)

    #######################################################################
    # Gradient of V w.r.t. matrix Ct -- report
    #######################################################################

    ppm.ppm(
        dVdCtauto,
        f"dVdCtauto, showing finite differences done in script, t={t},",
        sigfigs=7,
    )
    ppm.ppm(
        dVdCtmodule, f"dVdCtmodule, showing theoretical findings in module", sigfigs=7
    )
    ppm.ppm(
        dVdCtmodule - dVdCtauto, "difference from theory to approximation", sigfigs=2
    )

    tolvals.append(np.linalg.norm(dVdCtmodule - dVdCtauto))
    tolstrs.append(f"gradient of V w.r.t. C_{t}")

    #######################################################################
    # Gradient of V w.r.t. vector ct -- numerical evaluation by finite difference
    #######################################################################
    print(f"\n{10*'*':s} Sensitivity of V to c_{t} {10*'*':s}")

    def V(ct):
        # Input is a tall skinny matrix, (n+m)x(1)
        cmod = copy.deepcopy(c0)
        cmod[:, :, t] = ct
        mysys = mylqr.DiscreteLQR(C0, cmod, F0, f)
        bestx, bestu, bestlam = mysys.bestxul(x0)
        return mysys.J(bestx, bestu)

    ct0 = sys0.c(t)
    print(f"For t={t}, nominal value V(ct) = {V(ct0):13.6e}.")

    dVdctauto = grad(V, ct0)

    #######################################################################
    # Gradient of V w.r.t. vector ct -- theoretical evaluation from KKT system
    #######################################################################

    dVdctmodule = sys0.gradctV(t, x0)

    #######################################################################
    # Gradient of V w.r.t. vector ct -- report
    #######################################################################

    ppm.ppm(
        dVdctauto, f"dVdctauto, showing automatic finite differences, t={t},", sigfigs=7
    )
    ppm.ppm(
        dVdctmodule, f"dVdctmodule, showing theoretical findings in module", sigfigs=7
    )
    ppm.ppm(
        dVdctmodule - dVdctauto, "difference from theory to approximation", sigfigs=2
    )

    tolvals.append(np.linalg.norm(dVdctmodule - dVdctauto))
    tolstrs.append(f"gradient of V w.r.t. c_{t}")

    if t == T:
        break

    #######################################################################
    # Gradient of V w.r.t. matrix Ft -- finite difference approach
    #######################################################################
    print(f"\n{10*'*':s} Sensitivity of V to F_{t} {10*'*':s}")

    def V(Ft):
        # Input is a square symmetric matrix, (n+m)x(n+m)
        Fmod = copy.deepcopy(F0)
        Fmod[:, :, t] = Ft
        mysys = mylqr.DiscreteLQR(C0, c, Fmod, f)
        bestx, bestu, bestlam = mysys.bestxul(x0)
        return mysys.J(bestx, bestu)

    Ft0 = sys0.F(t)
    print(f"For t={t}, nominal value V(Ft) = {V(Ft0):13.6e}.")

    dVdFtauto = grad(V, Ft0)

    #######################################################################
    # Gradient of V w.r.t. matrix Ft -- theoretical prediction from system built-in
    #######################################################################

    dVdFtmodule = sys0.gradFtV(t, x0)

    #######################################################################
    # Gradient of V w.r.t. matrix Ft -- report
    #######################################################################

    ppm.ppm(dVdFtauto, f"dVdFtauto, from automated FD calculations,", sigfigs=7)
    ppm.ppm(
        dVdFtmodule, f"dVdFtmodule, showing theoretical findings in module,", sigfigs=7
    )
    ppm.ppm(
        dVdFtmodule - dVdFtauto,
        "difference from theory to auto-FD approximation,",
        sigfigs=2,
    )

    tolvals.append(np.linalg.norm(dVdFtmodule - dVdFtauto))
    tolstrs.append(f"gradient of V w.r.t. F_{t}")

    #######################################################################
    # Gradient of V w.r.t. vector ft -- numerical evaluation by finite difference
    #######################################################################
    print(f"\n{10*'*':s} Sensitivity of V to f_{t} {10*'*':s}")

    def V(ft):
        # Input is a tall skinny matrix, (n)x(1)
        fmod = copy.deepcopy(f0)
        fmod[:, :, t] = ft
        mysys = mylqr.DiscreteLQR(C0, c0, F0, fmod)
        bestx, bestu, bestlam = mysys.bestxul(x0)
        return mysys.J(bestx, bestu)

    ft0 = sys0.f(t)
    print(f"For t={t}, nominal value V(ft) = {V(ft0):13.6e}.")

    dVdftauto = grad(V, ft0)

    #######################################################################
    # Gradient of V w.r.t. vector ft -- theoretical evaluation from KKT system
    #######################################################################

    dVdftmodule = sys0.gradftV(t, x0)

    #######################################################################
    # Gradient of V w.r.t. vector ft -- report
    #######################################################################

    ppm.ppm(
        dVdftauto, f"dVdftauto, showing automatic finite differences, t={t},", sigfigs=7
    )
    ppm.ppm(
        dVdftmodule, f"dVdftmodule, showing theoretical findings in module,", sigfigs=7
    )
    ppm.ppm(
        dVdftmodule - dVdftauto, "difference from theory to approximation,", sigfigs=2
    )

    tolvals.append(np.linalg.norm(dVdftmodule - dVdftauto))
    tolstrs.append(f"gradient of V w.r.t. f_{t}")

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

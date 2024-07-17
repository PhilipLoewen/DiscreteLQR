# test-LTI-Vonly.py
#
# Exercise the optimization part of the LQR module, with an LTI system.
#   * Print the KKT matrix to make sure it matches the theory.
#   * Compare the optimal trajectories from the KKT method
#     with the trajectories from the forward-backward method.
#
# Simple stripped-down version of the original in test-LQR-optim.py
#
# (c) Philip D Loewen, 2024-07-16

import numpy as np

# Local modules
import DiscreteLQR as LQRmodule
import PrettyPrinter as ppm  # PDL 2024-06-03

#######################################################################
# Choose basic system parameters.
# Recommended values for each of m,n,T are in the interval [1,9].
#######################################################################
n = 5  # State vector dimension
m = 3  # Input/control vector dimension
T = 8  # Literal final time. Interesting subscripts are 0,...,T

x0 = np.random.rand(n, 1)  # Initial state

#######################################################################
# Invent system ingredients whose digit patterns will be recognizable
# in the KKT mtx.
#######################################################################
# Dynamics
for t in [0]:
    # Matrix A[t], entry ij, will have value whose digits look like 0.tij,
    # provide all of t,i,j are in the interval [0,9]. Note that "entry ij"
    # uses 1-based indexing like in math writing, not Pythonic indexes.
    # For matrix B[t], the form is the same but the numbers are negative.

    n_ones = np.ones((n, 1))
    Arow = np.array(range(1, n + 1)).reshape((1, n))
    A = t * np.ones((n, n)) / 10.0 + n_ones @ Arow / 1000.0 + Arow.T @ n_ones.T / 100.0
    # ppm.ppm(A,f"A[{t}]",sigfigs=3)

    m_ones = np.ones((m, 1))
    Brow = np.array(range(1, m + 1)).reshape((1, m))
    B = -(
        t * np.ones((n, m)) / 10.0 + n_ones @ Brow / 1000.0 + Arow.T @ m_ones.T / 100.0
    )
    # ppm.ppm(B,f"B[{t}]",sigfigs=3)

    F = np.hstack((A, B))
    f = (0.001 + np.ones((n, 1)) * t * 1.11).reshape((n, 1))

# Costs -- complicated by requirements of symmetry and positivity
for t in [0]:
    # Similar story to the setup for matrix A, but with a bigger diagonal.
    C0 = t * np.ones((n + m, n + m))
    C0 += 10 * np.eye(n + m)
    C0 += (
        np.ones((m + n, 1)) @ np.array(range(1, m + n + 1)).reshape((1, m + n)) / 200.0
    )
    C0 += np.array(range(1, m + n + 1)).reshape((m + n, 1)) @ np.ones((1, m + n)) / 20.0
    C0 = (C0 + C0.T) / 2.0

    C = C0
    c = -(0.001 + np.ones((n + m, 1)) * t * 1.01).reshape((m + n, 1))

#######################################################################
# Instantiate the system and have it print what it internalizes.
#######################################################################
myLQRsystem = LQRmodule.DiscreteLQR(C, c, F, f, T=T)

# myLQRsystem.printself()

###############################################################
Ntests = 20
tolerance = 1E-9
worsttest = 0
worstrelerr = -1.0
worstx0 = np.random.rand(n,1)

for test in range(1,Ntests+1):
    print(f"\n-- Test number {test} --")
    x0 = np.random.rand(n,1) * 10 - 5.0
    ppm.ppm(x0.T,"transpose of the initial point x0")
    Vquad = myLQRsystem.V(x0)

    bestx, bestu, bestlam  = myLQRsystem.bestxul(x0)
    Valt = myLQRsystem.J(bestx,bestu)

    relerr = np.abs((Vquad - Valt) / Vquad )

    print(f"Quadratic prediction of V(x0):   {Vquad:13.6E}")
    print(f"Trajectory-based calc for V(x0): {Valt:13.6E}")
    print(f"Relative discrepancy in these:   {relerr:8.1E}")

    if relerr > worstrelerr:
        worsttest = test
        worstrelerr = relerr
        worstx0 = x0

print(f"\nTest {worsttest} was the worst of these {Ntests}.")
ppm.ppm(worstx0.T,f"transpose of the initial point x0 for test {worsttest}")
print(f"Test {worsttest} had a relative error {worstrelerr:7.1E}.")

if worstrelerr < tolerance:
    print("All the other tests were at least as good. This outcome is acceptable.\n")
else:
    print("Further investigation is required.\n")


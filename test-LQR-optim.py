# test-LQR-optim.py
#
# Exercise the optimization part of the LQR module.
#   * Print the KKT matrix to make sure it matches the theory.
#   * Compare the optimal trajectories from the KKT method
#     with the trajectories from the forward-backward method.
#
# (c) Philip D Loewen, 2024-07-05

import numpy as np
import sys

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
F = np.zeros((n, n + m, T))
f = np.zeros((n, 1, T))
for t in range(T):
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

    F[:, :, t] = np.hstack((A, B))
    # ppm.ppm(F[:,:,t],f"F[{t}] = [A[{t}] B[{t}]]",sigfigs=3)

    f[:, [0], t] = 0.001 + np.ones((n, 1)) * t * 1.11
    # ppm.ppm(f[:,[0],t],f"f[{t}]",sigfigs=3)

# Costs -- complicated by requirements of symmetry and positivity
C = np.zeros((n + m, n + m, T + 1))
c = np.zeros((n + m, 1, T + 1))
for t in range(T + 1):
    # Similar story to the setup for matrix A, but with a bigger diagonal.
    C0 = t * np.ones((n + m, n + m))
    C0 += 10 * np.eye(n + m)
    C0 += (
        np.ones((m + n, 1)) @ np.array(range(1, m + n + 1)).reshape((1, m + n)) / 200.0
    )
    C0 += np.array(range(1, m + n + 1)).reshape((m + n, 1)) @ np.ones((1, m + n)) / 20.0
    C0 = (C0 + C0.T) / 2.0

    C[:, :, t] = C0
    # ppm.ppm(C0,f"C0[{t}]",sigfigs=3)

    c[:, [0], t] = -(0.001 + np.ones((n + m, 1)) * t * 1.01)
    # ppm.ppm(c[:,[0],t],f"c[{t}]",sigfigs=3)

#######################################################################
# Instantiate the system and have it print what it internalizes.
#######################################################################
myLQRsystem = LQRmodule.DiscreteLQR(C, c, F, f)

myLQRsystem.printself()

###############################################################
print("*** Check the setup for Dynamic Programming. ***")
print(f"The final quadratic matrix V_{T} should match ")
print(f"the 1,1-block of the cost matrix C_{T}. So check ...")
ppm.ppm(myLQRsystem.V_mats[:, :, T], f"V_{T}")
ppm.ppm(myLQRsystem.C(T)[:n, :n], f"(C_{T})_11")
print(f"Second, vector v_{T} should match block 1 of cost c_{T}:")
ppm.ppm(myLQRsystem.v_vecs[:, [0], T], f"v_{T}")
ppm.ppm(myLQRsystem.c(T)[:n, [0]], f"(c_{T})_1")
print(f"Finally, expect 0 for the value of beta_{T} = {myLQRsystem.beta[T]}.")

###############################################################
if True:
    print("\n*** Test the basic dynamic process, method openloopxu ***")

    u_vecs = np.around(np.random.rand(m, T), decimals=1)
    print("(Control input u has dimension m={m:d}.)")
    print(f"Random inputs for T={T:d} steps give the transitions below:")
    # ppm.ppm(u_vecs,"u_vecs")

    openx, openu = myLQRsystem.openloopxu(x0, u_vecs)

    for r in range(0, T):
        state = ", ".join(["{:9.2e}".format(q) for q in openx[:, 0, r]])
        ctl = ", ".join(["{:9.2e}".format(q) for q in openu[:, 0, r]])
        print("  State x_{:d}: {:s}; ".format(r, state), end="")
        print("input u_{:d}: {:s}. ".format(r, ctl))
    r = T
    state = ", ".join(["{:9.2e}".format(q) for q in openx[:, 0, r]])
    print("  State x_{:d}: {:s}.".format(r, state))

    lossJ = myLQRsystem.J(openx, openu)
    print(f"The objective value for this trajectory is J = {lossJ:11.5e}.")

###############################################################
print(f"\n*** Test the forward-backward optimizer, method bestxul ***")

V = myLQRsystem.V_mats[:, :, 0]
v = myLQRsystem.v_vecs[:, [0], 0]
b = myLQRsystem.beta[0]
quadJ = (0.5 * x0.T @ V @ x0 + v.T @ x0 + b)[0, 0]
print("\nForward-backward iterative scheme computes the value function")
print("as a quadratic form valid for any initial point.")
print(f"For our initial point, it predicts the minimum value {quadJ:11.5e}.")

print("Forward/backward scheme for T={T:d} steps give the transitions below:")
bestx, bestu, bestlam = myLQRsystem.bestxul(x0)
for r in range(0, T):
    state = ", ".join(["{:9.2e}".format(q) for q in bestx[:, 0, r]])
    ctl = ", ".join(["{:9.2e}".format(q) for q in bestu[:, 0, r]])
    mul = ", ".join(["{:9.2e}".format(q) for q in bestlam[:, 0, r]])
    print("    State x_{:d}: {:s}; ".format(r, state), end="")
    print("input u_{:d}: {:s}; ".format(r, ctl), end="")
    print("mult  lambda_{:d}: {:s}. ".format(r, mul))
r = T
state = ", ".join(["{:9.2e}".format(q) for q in bestx[:, 0, r]])
print("    State x_{:d}: {:s}.".format(r, state))

bestJ = myLQRsystem.J(bestx, bestu)
print(f"Evaluating the objective on this trajectory gives J = {bestJ:11.5e}.")
relerr = abs((bestJ - quadJ) / quadJ)
print(f"The relative error against the quadratic prediction above is {relerr:7.1e}.")

###############################################################
if True:
    print("\n*** Print key elements of the computed feedback plan ***")
    # print(f"V_mats.shape = {myLQRsystem.V_mats.shape}.")

    for t in range(T):
        print(" ")
        ppm.ppm(myLQRsystem.K_mats[:, :, t], f"K_mats[:,:,{t}]")
        ppm.ppm(myLQRsystem.k_vecs[:, [0], t], f"k_vecs[:,[0],{t}]")
        ppm.ppm(myLQRsystem.V_mats[:, :, t], f"V_mats[:,:,{t}]")
    print(" ")
    t = T
    ppm.ppm(myLQRsystem.V_mats[:, :, t], f"V_mats[:,:,{t}]")


if True:
    ###############################################################
    print(f"\n*** Test construction of the KKT matrix, using T={T:d} ***")
    M = myLQRsystem.KKTmtx()
    Mrows, Mcols = M.shape
    if Mrows > 32 or Mcols > 325:
        print(
            f"KKT matrix has shape {Mrows} x {Mcols}. That's too big. Printing suppressed."
        )
    else:
        ppm.ppm(M, "KKT matrix M")
    print(f"The condition number of the KKT matrix is {np.linalg.cond(M):9.2E}.")

    ###############################################################
    print(f"\n*** Test construction of the KKT right-hand side, using T={T:d} ***")
    b = myLQRsystem.KKTrhs(x0)
    brows, bcols = b.shape
    if brows > 32:
        print(f"KKT RHS vector b has shape {brows} x {bcols}. Printing suppressed.")
    else:
        ppm.ppm(b, "KKT RHS vector b")
    ###############################################################
    print(f"\n*** Solve the KKT system, T={T:d} ***\n")
    w0 = np.linalg.solve(M, b)
    residual0 = M @ w0 - b
    relres0 = np.linalg.norm(residual0) / np.linalg.norm(b)
    # print(f"First-round relative residual has norm {relres0:8.2E}.")

    update = np.linalg.solve(M, residual0)
    w = w0 - update
    residual = M @ w - b
    relres = np.linalg.norm(residual) / np.linalg.norm(b)
    # print(f"Improved solution has relative residual with norm {relres:8.2E}.")

    wrows, wcols = w.shape
    if brows > 32:
        print(
            f"KKT solution vector w has shape {wrows} x {wcols}. Printing suppressed."
        )
    else:
        ppm.ppm(w, "KKT solution vector w")

    # Extract components of solution vector w into standard-format arrays
    KKTx = np.zeros((n, 1, T + 1))
    KKTu = np.zeros((m, 1, T))
    KKTl = np.zeros((n, 1, T))

    KKTx[:, [0], 0] = x0.reshape(n, 1)
    for r in range(0, T):
        KKTu[:, [0], r] = w[r * (m + 2 * n) : r * (m + 2 * n) + m, :]
        KKTl[:, [0], r] = w[(m + r * (m + 2 * n)) : (m + r * (m + 2 * n) + n), :][:]
        KKTx[:, [0], r + 1] = w[
            (m + n + r * (m + 2 * n)) : (m + n + r * (m + 2 * n) + n), :
        ][:]

    # ppm.ppm(KKTx,"KKTx") # Unhide these to see the explicit internal representations
    # ppm.ppm(KKTu,"KKTu")
    # ppm.ppm(KKTl,"KKTl")

    print("\nDynamic report from KKT method:\n")
    for r in range(0, T):
        state = ", ".join(["{:9.2e}".format(q) for q in KKTx[:, 0, r]])
        ctl = ", ".join(["{:9.2e}".format(q) for q in KKTu[:, 0, r]])
        mul = ", ".join(["{:9.2e}".format(q) for q in KKTl[:, 0, r]])
        print("  State x_{:d}: {:s}; ".format(r, state), end="")
        print("input u_{:d}: {:s}; ".format(r, ctl), end="")
        print("mult  lambda_{:d}: {:s}. ".format(r, mul))
    r = T
    state = ", ".join(["{:9.2e}".format(q) for q in KKTx[:, 0, r]])
    print("  State x_{:d}: {:s}.".format(r, state))

    KKTJ = myLQRsystem.J(KKTx, KKTu)
    print(
        f"\nEvaluating the objective for the KKT-generated trajectory gives J = {KKTJ:11.5E}."
    )

###############################################################
print("\n*** Comparing optimal values using 3 methods. ***")

print(f"Quadratic prediction J = {quadJ:11.5e}")
print(f"KKT system traj      J = {KKTJ:11.5e}")
print(f"Forward/backward     J = {bestJ:11.5e}")


print("\nComparing states from KKT with states from bestxul:\n")
ppm.ppm(KKTx, "KKTx")
ppm.ppm(bestx, "bestx")
relerr = np.linalg.norm(KKTx - bestx) / np.linalg.norm(KKTx)
print(f"Relative Frobenius norm discrepancy for KKTx vs bestx = {relerr:8.2E}.")
if relerr > 0.2:
    print("Large discrepancy noted. Here is the difference matrix.")
    ppm.ppm(KKTx - bestx, "KKTx-bestx")
    print("")

print("\nComparing controls from KKT with controls from bestxul:\n")
ppm.ppm(KKTu, "KKTu")
ppm.ppm(bestu, "bestu")
relerr = np.linalg.norm(KKTu - bestu) / np.linalg.norm(KKTu)
print(f"Relative Frobenius norm discrepancy for KKTu vs bestu = {relerr:8.2E}.")
if relerr > 0.2:
    print("Large discrepancy noted. Here is the difference matrix.")
    ppm.ppm(KKTu - bestu, "KKTu-bestu")
    print("")

print("\nComparing multipliers from KKT with multipliers from bestxul:\n")
ppm.ppm(KKTl, "KKTl")
ppm.ppm(bestlam, "bestlam")
relerr = np.linalg.norm(KKTl - bestlam) / np.linalg.norm(KKTl)
print(f"Relative Frobenius norm discrepancy for KKTl vs bestlam : {relerr:8.2E}.")
if relerr > 0.2:
    print("Large discrepancy noted. Here is the difference matrix.")
    ppm.ppm(KKTl - bestlam, "KKTl-bestlam")
    print("")

sys.exit(0)

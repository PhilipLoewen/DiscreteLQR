import copy
import numpy as np
import PrettyPrinter as ppm

class DiscreteLQR:
    """
    Record the defining properties and calculate key features
    for a classic discrete-time LQR problem in which a state
    vector x evolves for T time steps like this:
        x[t+1] = A[t]x[t] + B[t]u[t] + f[t],  t=0,1,...,T-1;
        x[0]     outside user input.
    Here u is the time-varying control vector and f allows for
    an additive time-varying perturbation. To set up the system,
    the user must supply A, B, and f; also T.

    We write n for the dimension of x, and m for the dimension of u.

    For convenience, stack vectors x[t] and u[t] into a tall
    column and call the result z[t]. Then a "trajectory" for
    the system has the form (z[0],z[1],...,z[T-1],z[T]),
    subject to the dynamic equations above. [But see note below.]

    Every trajectory as above has an associated cost, given by
        J = sum( (1/2) *z[t]' * C[t] * z[t] + c[t]' * z[t] ).
    Here the coefficient matrices C[0],...,C[T] and vectors
    c[0],...,c[T] are required to set up the system.
    These special notes apply:
        0. Note that the vector u[T] is irrelevant in the dynamics.
           We typically treat trajectories in terms of the
           separated sequences (x[0],...,x[T]) and (u[0],...,u[T-1]).
        1. In the user-specified ingredients C[0] and c[0],
           all the elements that that touch components of
           the initial vector x[0] are ignored when calculating J.
        2. In the user-specified ingredients C[T] and c[T],
           all the elements that that touch components of
           the control value u[T] are ignored when calculating J.

    To set things up, the user must provide the cost coefficients C, c,
    the dynamic ingredients F,f, and the final subscript T. Changing
    any one of these elements requires defining a whole new LQR object.

    For implementation, all non-scalar quantities named above are
    represented as numpy arrays with 2 dimensions. So, for example,
    each n-dimensional state vector x[t] is treated as an n-by-1
    matrix.

    ## INITIALIZATION ##

    The Autonomous Case: When the problem data are time-invariant,
    a system object can be created using simple numpy matrices of
    appropriate dimensions; a final time T is required. E.g., for
    a low-pass filter with m=2 inputs and n=2 states, the dynamics
        x[t+1] = (1-r)*x[T] + r u[T]
    could be expressed in code like this:
        A = np.eye(2) * (1-r)
        B = np.eye(2) * r
        F = np.hstack((A,B))
        f = np.zeros((2,1))
    Standard LQR cost terms like (x[t]'*Q*x[t] + u[t]'*R*u[t])/2
    would be captured by saying
        Q = np.eye(2) * 5.0
        R = np.eye(2)
        C = np.block([[Q, np.zeros((2,2))],[np.zeros((2,2)),R]])
        c = np.ones((4,1))
    Then one could specify the number of time-steps by saying
        Tf= 12
    and create the corresponding system object with this powerful
    command:
        mysystem = mylqr.DiscreteLQR(C,c,F,f,T=Tf)

    The General Case: Each of the matrix-valued inputs in the autonomous
    case just mentioned can be replaced by a numpy.ndarray whose final
    index corresponds to the subscript t in the general notation above.
    So in the general case F would be a numpy array with 3 indices,
    and the Python notation F[:,:,t] would select the n-by-(n+m) matrix
    denoted by F[t] in the general math symbols shown above. Similarly
    f would be a numpy array with 3 indices, and the notation f[:,:,t]
    would select the n-by-1 matrix denoted by f[t] in the math intro.
    Any individual elements that have no t-dependence can be specified
    just as in the autonomous case.

    The Final Time: To define the dynamics and cost with final index T,
    the dynamic coefficints F[t] and f[t] must be available up to t=T-1;
    the cost coefficients C[t] and c[t] must be available up to t=T.
    The keyword parameter T must be compatible with these requirements.
    If it is omitted, the restrictions just mentioned will be used to
    choose the largest value compatible with the given matrix families.
    (This fails in the purely autonomous case, where it would suggest
    T = infinity.)

    ## INTROSPECTION ##

    Methods anchored to the system will return the coefficient matrices
    for any t:
        C(t) - matrix of shape (n+m)x(n+m), defined for any t in 0,...,T;
        c(t) - matrix of shape (n+m)x(1), defined for any t in 0,...,T;
        F(t) - matrix of shape (n)x(n+m), defined for any t in 0,...,T-1;
        f(t) - matrix of shape (n)x(1), defined for any t in 0,...,T-1.
    Note that the values of C(0), c(0), C(T), c(T) have some blocks
    that get assigned the value 0 in compliance with the notes above.

    ## NON-APOLOGY ##

    Every non-scalar quantity in the code is a 2D numpy array.
    So, e.g., an n-component vector will be represented by an array of
    shape (n,1), with ndim=2. This adds a little overhead to the indexing,
    but saves confusion over the built-in behaviours of numpy whe arrays
    with different dimensions need to be combined in a calculation.

    ## DYNAMICS AND COST ##

    A system object has methods for the evolution and the cost, namely
        openloopxu(x0,uvecs) - returns sequence of xvecs and uvecs, given x0
        J(xvec,uvec) - returns scalar value of J

    ## OPTIMIZATION ##

    Two methods are provided: the forward-backward iteration
    and the direct KKT setup.

    Knowing the final time T, a backward iteration determines feedback
    coefficients K_t and k_t for T=0,1,...,T-1, and
    quadratic value coefficients V_t and v_t for t=0,...,T.
    These are all computed as part of creating the DiscreteLQR object,
    and stored as attributes of the system object after that.

    To get the optimal trajectory for a given initial point,
    use the method "bestxul".

    ## THE VALUE FUNCTION ##

    A system object has a method that returns the minimum cost from any
    initial state, namely
        V(x0) - scalar value of V at point x0, where x0 is an n-by-1 2D array.

    ## VALUE FUNCTION GRADIENTS ##

    Method list: Let V(x0) denote the minimum value in the optimization problem
    detailed above. For fixed x0, changing any of the problem data could influence
    the value. For suitable choices of t, the rate of influence (partial derivative)
    is available:
        gradFtV(t,x0) - derivative of value V(x0) w.r.t. F(t)
        gradCtV(t,x0) - derivative of value V(x0) w.r.t. C(t)
        gradftV(t,x0) - derivative of value V(x0) w.r.t. f(t)
        gradctV(t,x0) - derivative of value V(x0) w.r.t. c(t)
        gradx0V(x0)   - derivative of value V(x0) w.r.t. x0

    ## GENERAL GRADIENTS ##

    Given any differentiable function l that maps any trajectory to a number,
    let W denote the value of l along the optimal trajectory. Evidently W
    depends on the initial point x0 and the problem data. Knowing the partial
    derivatives of l with respect to x and u is enough to calculate the following:
        gradFtW(t,wx,wu,x0) - derivative of value W(x0) w.r.t. F(t)
        gradCtW(t,wx,wu,x0) - derivative of value W(x0) w.r.t. C(t)
        gradftW(t,wx,wu,x0) - derivative of value W(x0) w.r.t. f(t)
        gradctW(t,wx,wu,x0) - derivative of value W(x0) w.r.t. c(t)
        gradx0W(wx,wu,x0)   - derivative of value W(x0) w.r.t. x0

    ## GRADIENTS INVOLVING TIME-INVARIANT ELEMENTS ##

    The computed gradients of both V and W are different for problem data
    that are time-invariant. For example, suppose the dynamic matrices F
    are all identical, and this is embedded in the system setup by calling
    the constructor with a single n-by-(n+m) matrix in the appropriate position.
    In this case the matrix returned by gradFtV(t,x0) will be the same for each
    relevant t, and it will represent the sum of the gradients with respect to
    F(0),F(1),...,F(T-1). This is probably what users expect when they ask for
    the sensitivity of a linear time-invariant problem with respect to its
    coefficient matrices. (If you really need just the gradient with respect
    to a single specific F(t) in an LTI problem, simply define the problem
    using a 3D array of shape (n,n+m,T) in the first place. Using the same
    matrix F to define each of that array's [:,:,t] slices will cause the
    module to use the elementwise intepretation in all sensitivity computations.

    ## IMPLEMENTATION NOTES ##

    Various other methods are provided, but they are likely to be less
    frequently used. See their docstrings below for details.

    There are many many print statements hidden in comments
    that helped with the initial construction and may yet be useful
    for debugging in the future.

    Philip D Loewen,
    ... 2024-11-23: Remember more internal elements from the backward pass in __init__
    ... 2024-07-16: Get V(x0) from the initial setup, no trajectory needed!
    ... 2024-07-04: Improve sensitivity functions for the autonomous case
    ... 2024-07-03: Finish some sensitivity functions and testing
    ... 2024-06-30: Check details for the non-apology above
    ... 2024-06-29: Reorganize aggressively, draft this lengthy docstring
    ... 2024-06-25: Add sensitivity matrix as a memoized attribute
    ... 2024-06-25: Change handling of unspecified T to use NoneType
    ... 2024-06-20: New slicing syntax for matrices in forward-backward solution scheme
    ... 2024-06-20: Moved parameter finalT in KKT elements into optional final spot
    ... 2024-06-12: Minor fixes
    ... 2023-08-23: Latest revision
    ... 2023-07-20: First working draft, inspired by Thiago da Cunha Vasco
    """

    def __init__(self, C_mats, c_vecs, F_mats, f_vecs, T=None):
        """
        The state space is n-dimensional and the control space is m-dimensional.

        :param C_mats: numpy array, shape (n+m, n+m, T+1). Quadratic cost terms;   t = 0,...,T.
        :param c_vecs: numpy array, shape (n+m, 1, T+1).   Vector cost terms;      t = 0,...,T.
        :param F_mats: numpy array, shape (n, n+m, T).     System dynamics matrix; t = 0,...,T-1.
        :param f_vecs: numpy array, shape (n, 1, T).       System dynamics vector; t = 0,...,T-1.

        Shortcuts: For each parameter above, it's OK to supply just a single 2D numpy array
        instead of a long list of T or T+1 identical copies. If you do this, the single
        array supplied will be used for all relevant t.

        For the column-vector inputs c_vecs and f_vecs, lower-dimensional numpy arrays
        with obvious interpretations are handled sensibly (but we don't mention this aloud).

        Checking: Consistency of the state and control dimensions is checked
        when an instance is created, but there are probably ways to mess up
        the input that have not been anticipated here.
        """

        if C_mats.shape[0] != C_mats.shape[1]:
            print(f"C_mats.shape = {C_mats.shape}.")
            raise AttributeError("Quadratic cost coefficient matrices must be square.")
        if C_mats.shape[0] != c_vecs.shape[0]:
            print(f"C_mats.shape = {C_mats.shape}; c_vecs.shape = {c_vecs.shape}.")
            raise AttributeError(
                "Cost coefficient matrices and vectors must have compatible dimensions."
            )
        if C_mats.shape[1] != F_mats.shape[1]:
            print(f"C_mats.shape = {C_mats.shape}; F_mats.shape = {F_mats.shape}.")
            raise AttributeError(
                "Matrices for dynamics and costs must have compatible dimensions."
            )
        if F_mats.shape[1] <= F_mats.shape[0]:
            print(f"F_mats.shape = {F_mats.shape}.")
            raise AttributeError(
                "Dynamics matrices leave no room for the input signal!"
            )
        if F_mats.shape[0] != f_vecs.shape[0]:
            print(f"F_mats.shape = {F_mats.shape}; f_vecs.shape = {f_vecs.shape}.")
            raise AttributeError("Dynamics offset term has incompatible dimension.")

        self.autonomous = max(C_mats.ndim, c_vecs.ndim, F_mats.ndim, f_vecs.ndim) < 3
        # print(f"In DiscreteLQR.init, various dimenions are {C_mats.ndim}, {c_vecs.ndim}, {F_mats.ndim}, {f_vecs.ndim}.")
        # print(f"Consequently self.autonomous is {self.autonomous}.")

        if T == None and self.autonomous:
            raise AttributeError(
                "For autonomous systems, a positive final time T must be provided explicitly."
            )

        # A few basic sanity tests have passed.
        # Promote the input parameters to be attributes of the new object we are constructing.
        # Enforce symmetry on the cost coefficient matrices.

        if C_mats.ndim == 2:
            # Autonomous case, all coeffs C_t are this same thing.
            self.C_mats = (C_mats + C_mats.T) / 2.0
        else:
            # Time-varying case, symmetrize each component.
            self.C_mats = np.zeros(C_mats.shape)
            for t in range(C_mats.shape[2]):
                self.C_mats[:, :, t] = (C_mats[:, :, t] + C_mats[:, :, t].T) / 2.0

        self.c_vecs = c_vecs
        self.F_mats = F_mats
        self.f_vecs = f_vecs

        self.state_dim = self.F_mats.shape[0]
        self.input_dim = self.C_mats.shape[0] - self.state_dim
        self.n = self.state_dim
        self.m = self.input_dim

        # Infer largest usable value for final subscript T from other given inputs
        intfinity = 2**30  # Leave just a little headroom below (2**31 - 1).
        Tceiling = []

        # print(f"DEBUG: C_mats has shape {self.C_mats.shape}.")
        if self.C_mats.ndim == 3:
            Tceiling.append(self.C_mats.shape[2] - 1)
        else:
            Tceiling.append(intfinity)

        # print(f"DEBUG: c_vecs has shape {self.c_vecs.shape}.")
        if self.c_vecs.ndim == 3:
            Tceiling.append(self.c_vecs.shape[2] - 1)
        else:
            Tceiling.append(intfinity)

        # print(f"DEBUG: F_mats has shape {self.F_mats.shape}.")
        if self.F_mats.ndim == 3:
            Tceiling.append(self.F_mats.shape[2])
        else:
            Tceiling.append(intfinity)

        # print(f"DEBUG: f_vecs has shape {self.f_vecs.shape}.")
        if self.f_vecs.ndim == 3:
            Tceiling.append(self.f_vecs.shape[2])
        else:
            Tceiling.append(intfinity)
        self.Tmax = min(Tceiling)
        # print(f"DEBUG: Features above lead to Tceiling={Tceiling}, so Tmax={self.Tmax:d}.\n")

        if T != None and T >= 0:
            # print(f"DEBUG: Input declares explicit final T={T}.\n       ",end="")
            if T == 0:
                raise AttributeError(
                    "The DiscreteLQR module requires T>0; given inputs imply T=0."
                )
                return None
            if T <= self.Tmax:
                # print(f"This is compatible with Tmax={self.Tmax}, so we will use T={T}.\n")
                self.Tmax = T
            else:
                # print(f"This exceeds Tmax={self.Tmax}, so we will use T={self.Tmax} instead. Expect trouble!\n")
                pass
        self.T = self.Tmax

        # Sometimes we are interested in the KKT matrix, 
        # but that only needs to be built once (or perhaps never).
        # Declare the name and assign it the default value "None".
        self.KKT = None

        # Reserve space for the sensitivity matrix.
        # This depends on the initial point x0. In code, we use the default value None.
        # If x0 is None, these elements are not yet ready. They get built on first request.
        # If x0 is a n-by-1 array, it means our system object contains an 
        # optimal trajectory and its basic-form sensitivity matrix for that x0.

        self.sol_x0 = None
        self.sol_x = None
        self.sol_u = None
        self.sol_lam = None

        # Reserve space for the Big Olde Product matrix used for general sensitivity calculations.
        self.BOP = None
        self.BOPwx = None
        self.BOPwu = None
        self.senssol = None

        # Run the backward pass detailed in the theoretical writeup.
        # Critical ingredients of the system description that this produces are ...
        #   K_mats ... shape (m,n,T)
        #   k_vecs ... shape (m,,1t)
        # (the optimal input for index t will be  u_t = K[:,:,t]@x_t + k[:,t], t=0,1,...,T-1)
        #   V_mats ... shape (n,n,T+1)
        #   v_vecs ... shape (n,1,T+1)
        #   beta   ... shape (T+1)
        # (so the optimal value from any starting point is easy to calculate)
        #   Q_mats ... shape (m+n,m+n,T-1)
        #   q_vecs ... shape (m+n,1,T-1)
        # Note that the matrix collections named for K, Q, and V
        # are all independent of both the initial state and the
        # coefficients of the linear terms in the problem setup.
        # 
        # The objective sum has T+1 terms, but we only need T feedback
        # coefficients to evaluate it ... so that's all we compute and return.

        Q_mats = np.zeros((self.m+self.n, self.m+self.n, self.T))
        q_vecs = np.zeros((self.m+self.n, 1, self.T))

        K_mats = np.zeros((self.m, self.n, self.T))
        k_vecs = np.zeros((self.m, 1, self.T))

        V_mats = np.zeros((self.n, self.n, self.T + 1))
        v_vecs = np.zeros((self.n, 1, self.T + 1))

        beta = np.zeros(self.T + 1)

        T = self.T

        # Set up for backward iteration by declaring values for t=T:

        V_mats[:, :, T] = self.C(T)[: self.n, : self.n]
        v_vecs[:, 0, T:] = self.c(T)[: self.n, [0]]
        beta[T] = 0.0  # Redundant but readable

        # Our theoretical formulation starts the sum with t=0.
        # This aligns math subscripts with Python indices perfectly.
        # The loop below starts with t=T-1 and counts down to t=0:
        for t in reversed(range(T)):
            C = self.C(t)
            c = self.c(t)
            F = self.F(t)
            f = self.f(t)
            V = V_mats[:, :, t + 1]
            v = v_vecs[:, 0, t + 1].reshape(self.n, 1)

            # Q = self.C(t) + self.F(t).T @ V_mats[:, :, t + 1] @ self.F(t)
            Q = C + F.T @ V @ F

            # q = (
            #    self.c(t)
            #    + self.F(t).T @ V_mats[:, :, t + 1] @ self.f(t)
            #    + self.F(t).T @ v_vecs[:, t + 1].reshape(self.n, 1)
            # )
            q = c + F.T @ V @ f + F.T @ v

            if False:
                # Print for debugging.
                print(f"\nReport from stage t={t}:")
                ppm.ppm(self.C(t),f"C({t})")
                ppm.ppm(self.F(t),f"F({t})")
                ppm.ppm(c,f"c({t})")
                ppm.ppm(f,f"f({t})")
                ppm.ppm(V,f"V({t+1})")
                ppm.ppm(v,f"v({t+1})")
                ppm.ppm(Q,f"Q({t})")
                ppm.ppm(q,f"q({t})")

            Q_xx = Q[: self.n, : self.n]
            Q_ux = Q[self.n :, : self.n]
            Q_xu = Q[: self.n, self.n :]
            Q_uu = Q[self.n :, self.n :]

            q_x = q[: self.n].reshape(self.n, 1)
            q_u = q[self.n :].reshape(self.m, 1)

            K = -np.linalg.solve(Q_uu, Q_ux)
            k = -np.linalg.solve(Q_uu, q_u).reshape(self.m, 1)

            b = beta[t + 1] + 0.5 * q_u.T @ k + 0.5 * f.T @ V @ f + v.T @ f

            # Remember all these things as part of the system object.
            # Some of them may be re-used for later sensitivity calculations
            # of form gradW...

            Q_mats[:, :, t] = Q
            q_vecs[:, 0, [t]] = q

            K_mats[:, :, t] = K
            k_vecs[:, 0, [t]] = k

            V_mats[:, :, t] = Q_xx - K.T @ Q_uu @ K
            v_vecs[:, 0, [t]] = q_x + Q_xu @ k

            beta[t] = b

        self.Q_mats = Q_mats
        self.q_vecs = q_vecs
        self.K_mats = K_mats
        self.k_vecs = k_vecs
        self.V_mats = V_mats
        self.v_vecs = v_vecs
        self.beta = beta

    def C(self, t):
        """Return the quadratic cost coefficient matrix C_t, a matrix of shape (n+m,n+m)."""
        # Caution: Tweaks required in cases t=0 and t=T. Use deep copies to not break original mtx
        m = self.m
        n = self.n
        if self.C_mats.ndim == 2:
            # Constructor got a single 2D array as input
            Ct = copy.deepcopy(self.C_mats)
        elif self.C_mats.shape[2] == 1:
            # Constructor got a 3D array as input, but it only contains 1 matrix
            Ct = copy.deepcopy(self.C_mats[:, :, 0])
        else:
            # Typical time-varying setup.
            Ct = copy.deepcopy(self.C_mats[:, :, t])

        # At times t=0 and t=T, some adjustments are required.
        if t == 0:
            # Dependence on x[0] is forbidden
            Ct[:, 0:n] = np.zeros((m + n, n))
            Ct[0:n, :] = np.zeros((n, m + n))
        if t == self.T:
            # Dependence on u[T] is forbidden
            Ct[:, n : n + m] = np.zeros((m + n, m))
            Ct[n : n + m, :] = np.zeros((m, m + n))
        return Ct

    def c(self, t):
        """Return the quadratic cost linear coefficient c_t, with shape (n+m,1)."""
        m = self.m
        n = self.n
        if self.c_vecs.ndim == 1:
            # Constructor got a 1D array. Treat this as a constant column vector.
            # (Users who follow the documentation should never provide this.)
            ct = copy.deepcopy(self.c_vecs).reshape(self.m + self.n, 1)
        elif self.c_vecs.ndim == 2 and self.c_vecs.shape[1] == 1:
            # Constructor got a 2D column vector. Treat this as a constant column vector.
            # (Compliant users who want a constant c should provide this.)
            ct = copy.deepcopy(self.c_vecs).reshape(self.m + self.n, 1)
        else:
            # Nothing special ... c_vecs is a 3D array of columns.
            ct = copy.deepcopy(self.c_vecs[:, :, t]).reshape(self.m + self.n, 1)

        # At times t=0 and t=T, some adjustments are required.
        if t == 0:
            ct[0:n, [0]] = np.zeros((n, 1))  # Dependence on x[0] is forbidden
        if t == self.T:
            ct[n : n + m, [0]] = np.zeros((m, 1))  # Dependence on u[T] is forbidden
        return ct

    def F(self, t):
        """Return the dynamic coefficient matrix F_t, shape (n,n+m)."""
        if self.F_mats.ndim == 2:
            return self.F_mats
        if self.F_mats.shape[2] == 1:
            return self.F_mats[:, :, 0]
        return self.F_mats[:, :, t]

    def f(self, t):
        """Return the dynamic coefficient offset f_t, shape (n,1)."""
        if self.f_vecs.ndim == 1:
            # Constructor got a 1D array. Treat this as a constant column vector.
            # (Users who follow the documentation should never provide this.)
            return self.f_vecs.reshape(self.n, 1)
        if self.f_vecs.ndim == 2 and self.f_vecs.shape[1] == 1:
            # Constructor got a 2D column vector. Treat this as a constant column vector.
            # (Compliant users who want a constant f should provide this.)
            return self.f_vecs.reshape(self.n, 1)
        # Nothing special ... f_vecs is a 3D array of columns.
        return self.f_vecs[:, :, t].reshape(self.n, 1)

    #########################################################################################################
    ## DYNAMICS AND COST ##
    #########################################################################################################

    def openloopxu(self, x0, u_vecs):
        """Return the state/control sequences and the cost
        for the given initial point and control sequence."""

        if u_vecs.ndim < 2:
            print(
                f"Error in openloopxu: expecting 3D array of shape ({self.m},1,{self.T-1})."
            )
            return None
        elif u_vecs.ndim == 2:
            m, r = u_vecs.shape
            if m != self.m:
                print(
                    f"Error in openloopxu: inputs have dimension {m}, but system wants {self.m}."
                )
                return None
            u = u_vecs.reshape(self.m, 1, r)
            T = r
        else:
            m, o, r = u_vecs.shape
            if m != self.m:
                print(
                    f"Error in openloopxu: inputs have dimension {m}, but system wants {self.m}."
                )
                return None
            if o != 1:
                print(
                    f"Error in openloopxu: expecting 3D array of shape ({self.m},1,{self.T-1})."
                )
                return None
            u = u_vecs
        T = r
        # Now u looks like u_vecs, but with the correct shape.

        x = np.ndarray(
            (self.n, 1, 1 + T)
        )  # Save  1+T slots for the trajectory of n-by-1 columns
        if T > self.Tmax + 1:
            print(f"Warning from openloopxu: requested final time is {T+1}, ", end="")
            print(f"but some system dynamics are only known up to time {self.T}.")
            print(f"Execution continues, but problems should be expected.\n")

        x[:, [0], [0]] = x0[:].reshape(
            self.n, 1
        )  # Copy initial condition into result structure

        for t in range(T):
            z_t = np.vstack([x[:, [0], [t]], u[:, [0], [t]]]).reshape(
                self.n + self.m, 1
            )
            x[:, [0], [t + 1]] = self.F(t) @ z_t + self.f(t)

        return x, u

    #########################################################################################################
    def J(self, xvecs, uvecs):
        if xvecs.shape[0] != self.n:
            print(
                f"Trouble in function J: state vectors have the wrong dimension ",
                end="",
            )
            print(f"(got {xvecs.shape[0]}, expected {self.n}).")
            print(f"Giving up and returning NaN.")
            return float("NaN")
        if uvecs.shape[0] != self.m:
            print(
                f"Trouble in function J: input vectors have the wrong dimension ",
                end="",
            )
            print(f"(got {uvecs.shape[0]}, expected {self.m}).")
            print(f"Giving up and returning NaN.")
            return float("NaN")
        if xvecs.shape[2] - uvecs.shape[2] != 1:
            print(f"Trouble in function J:")
            print(
                f"  x_t is given for t=0,...,{xvecs.shape[1]-1}, suggesting T={xvecs.shape[1]-1},"
            )
            print(
                f"  u_t is given for t=0,...,{uvecs.shape[1]-1}, suggesting T={uvecs.shape[1]}"
            )
            print(f"Giving up and returning NaN.")
            return float("NaN")

        finalT = xvecs.shape[2] - 1
        if finalT != self.Tmax:
            print(
                f"System has final time T={self.Tmax} by design, but given trajectory ends at index t={finalT}."
            )
            print(f"And we do special things with the final index, so be careful!")

        Jval = np.zeros((1, 1))  # J is scalar but numpy ops favour a 1-by-1 array

        # PDL, 2024-06-25 ... repackaged this into a single loop.
        # Old alternative had first term, loop for middle, last term.

        for t in range(finalT + 1):
            if t > 0:
                x_t = xvecs[:, [0], t].reshape(self.n, 1)
            else:
                # For t=0, ignore all sum entries that explicitly involve x[0]
                x_t = np.zeros((self.n, 1))
            if t < finalT:
                u_t = uvecs[:, [0], t].reshape(self.m, 1)
            else:
                # For t=Tmax, ignore all sum entries that explicitly involve u[Tmax]
                u_t = np.zeros((self.m, 1))
            z_t = np.vstack((x_t, u_t))
            Jval += 0.5 * z_t.T @ self.C(t) @ z_t + self.c(t).T @ z_t

        return Jval[0, 0]

    #########################################################################################################
    ## OPTIMIZATION -- FORWARD/BACKWARD ITERATIONS DEFINE AN OPTIMAL TRAJECTORY AND ITS ASSOCIATED VALUE
    #########################################################################################################
    def bestxul(self, x0):
        """
        Find optimal state and control sequences and multipliers,
        given initial state.

        :param x0:  (n,1) array. The initial state.
        :return:    tuple (x,u,lam), where ...
                           x is an array of shape (n,1,T+1),
                           u is an array of shape (m,1,T),
                           lam is an array of shape (n,1,T).
        (Can't use lambda for a variable name, because it's a Python keyword.)
        """

        # Skip all the work below if the results are already on file:
        dx0 = 6.02e23  # Avogadro's Number. Geeky joke: any enormous scalar will do.
        if self.sol_x0 is not None:
            dx0 = np.linalg.norm(x0 - self.sol_x0)
        if dx0 < np.finfo(float).eps * 100:
            if self.sol_x is not None \
            and self.sol_u is not None \
            and self.sol_lam is not None:
                return self.sol_x, self.sol_u, self.sol_lam

        m = self.m
        n = self.n
        T = self.T

        x = np.zeros((n, 1, T + 1))
        u = np.zeros((m, 1, T))
        lam = np.zeros((n, 1, T))

        x[:, [0], 0] = x0
        for t in range(T):
            #            x_t = x[:, 0, t].reshape(n, 1)
            x_t = x[:, [0], t]  # 2024-06-30
            u_t = self.K_mats[:, :, t] @ x[:, 0, t].reshape(n, 1) + self.k_vecs[
                :, 0, t
            ].reshape(m, 1)
            u[:, 0, [t]] = u_t
            z_t = np.vstack((x_t, u_t))
            x[:, 0, [t + 1]] = self.F(t) @ z_t + self.f(t)

        x_T = x[:, 0, T].reshape(n, 1)
        u_T = np.zeros((m, 1))
        z_T = np.vstack((x_T, u_T))

        lam[:, [0], T - 1] = self.C(T)[0:n, 0:n] @ x[:, [0], T] + self.c(T)[0:n, [0]]

        for s in range(T - 1):
            t = T - 1 - s
            lam[:, 0, [t - 1]] = (
                self.F(t)[0:n, 0:n].T @ lam[:, 0, [t]]
                + self.C(t)[0:n, 0:n] @ x[:, 0, [t]]
                + self.C(t)[0:n, n : n + m] @ u[:, 0, [t]]
                + self.c(t)[0:n, [0]]
            )

        # Record the initial point and results in the object state
        self.sol_x0 = x0
        self.sol_x, self.sol_u, self.sol_lam = x, u, lam

        return x, u, lam


    def V(self,x0):
        """
        Return the minimum cost of a trajectory starting from the point x0.
        """
        # print(f"V_mats.shape = {self.V_mats.shape}; v_vecs.shape = {self.v_vecs.shape}.")
        mincost = 0.5 * x0.T @ self.V_mats[:,:,0] @ x0 + self.v_vecs[:,[0],0].T @ x0 + self.beta[0]
        # print(f"mincost = {mincost}")
        return( mincost[0,0] )
    #########################################################################################################
    ## OPTIMIZATION -- LAGRANGE MULTIPLIER APPROACH
    #########################################################################################################
    def KKTmtx(self):
        """
        Construct the massive block-matrix K from known elements.
        """
        T = self.Tmax
        m = self.m
        n = self.n

        Kmtxsize = (2 * n + m) * T
        # print("Apparently n={:d} and m={:d}, with T={:d},".format(n,m,T))
        # print("so the KKT matrix will have shape {:d}x{:d}.".format(Kmtxsize,Kmtxsize))

        if self.KKT is not None:
            if self.KKT.size == Kmtxsize**2:
                # We have already built and saved this matrix. Just return a copy.
                return self.KKT

        P = np.block([np.eye(n), np.zeros((n, m))])

        # Block rows with time index t=0
        t = 0
        C0_22 = self.C(0)[n:, n:]
        B0 = self.F(0)[:, n:]

        # Writing the top two block-rows like this captures all cases where T >= 1.
        topblock = np.block([[C0_22, B0.T, np.zeros((m, Kmtxsize - n - m))]])
        nextblock = np.block(
            [[B0, np.zeros((n, n)), -np.eye(n), np.zeros((n, Kmtxsize - 2 * n - m))]]
        )
        KKT = np.block([[topblock], [nextblock]])

        # Simple interior block rows with indices t=1,...,T-1 (seems ok for any T>0)
        for t in range(1, T):
            L = m + (t - 1) * (m + 2 * n)  # Count of zero cols on left
            R = Kmtxsize - L - 4 * n - 2 * m  # Count of zero cols on right
            midbit = np.block(
                [
                    [-P.T, self.C(t), self.F(t).T, np.zeros((n + m, n + m))],
                    [np.zeros((n, n)), self.F(t), np.zeros((n, n)), -P],
                ]
            )
            # ppm.ppm(midbit,"midbit with t={:d}, L={:d}, R={:d},".format(t,L,R))

            if R > 0:
                # Most block rows need padding with R 0-cols on the right
                nextblock = np.block(
                    [np.zeros((2 * n + m, L)), midbit, np.zeros((2 * n + m, R))]
                )
                # ppm.ppm(nextblock,"nextblock with interior t={:d},".format(t))
            else:
                # The block row for t=T-1 needs shrinking from the right, not padding.
                # print("Special Case t={:d}: L={:d}, R={:d}.".format(t,L,R))
                nextblock = np.block([np.zeros((2 * n + m, L)), midbit[:, :-m]])
                # ppm.ppm(nextblock,"nextblock with interior t={:d},".format(t))
            KKT = np.block([[KKT], [nextblock]])

        # Build a short row for t=T, to finish at the bottom
        t = T
        L = Kmtxsize - 2 * n
        midbit = np.block([-np.eye(n), self.C(T)[:n, :n]])
        # ppm.ppm(midbit,"midbit with t={:d},".format(t))
        nextblock = np.block([np.zeros((n, L)), midbit])
        # ppm.ppm(nextblock,"nextblock with t=T={:d},".format(t))

        KKT = np.block([[KKT], [nextblock]])

        self.KKT = KKT

        return KKT

    #########################################################################################################
    def KKTrhs(self, x0kkt):
        """
        Construct RHS vector for KKT system from known elements of system object.
        This depends on the initial point of interest. Allow an arbitrary choice,
        without prejudice to the "native" initial point for the system.
        """
        #        if finalT < 0:
        #            T = self.T
        #        else:
        #            T = int(finalT)

        T = self.Tmax

        n = self.F_mats.shape[0]
        m = self.F_mats.shape[1] - n

        x0 = x0kkt
        if x0.shape != (n, 1):
            print("WARNING: Given initial point has shape {0}.".format(x0.shape))
            x0 = x0kkt.reshape(n, 1)
            print("         Using a local copy with shape {0}.".format(x0.shape))

        Kmtxsize = (2 * n + m) * T
        # print("We have n={:d} and m={:d}, with T={:d},".format(n,m,T))
        # print("so the RHS vector will have shape {:d}x1.".format(Kmtxsize))

        # ppm.ppm(x0,"x0 (initial point)")

        # print("Shape of self.f(0) is {0}.".format(self.f(0).shape))
        # ppm.ppm(self.f(0),"f(0)")

        C0_21 = self.C(0)[n:, :n]  # Wrinkle: This should be zero. See theory writeup.
        # print(" ")
        # ppm.ppm(C0_21,"C0_21")
        # ppm.ppm(self.c(0),"c(0)")
        # print(" ")

        # C021x0 = C0_21 @ x0
        # ppm.ppm(C021x0,"C0_21 @ x0")

        RHS = -self.c(0)[n:, :] - C0_21 @ x0
        # print("Computed top block of RHS has native shape {0}.".format(RHS.shape))
        # RHS = RHS.reshape(m,1)

        A0 = self.F(0)[:n, :n]
        # A0x0= A0 @ x0
        # print("Native shape of A0 @ x0 is {0}.".format(A0x0.shape))

        b2 = -self.f(0) - A0 @ x0
        #        b2  = -self.f(0) - A0x0.reshape(n,1)
        #        b2  = b2.reshape(n,1)

        # print(" ")
        # ppm.ppm(A0,"A0")
        # ppm.ppm(A0@x0,"A0@x0")
        # ppm.ppm(b2,"b2")

        # print(" ")

        #        RHS = np.block([ [RHS], [b2.reshape(n,1)]])
        RHS = np.block([ [RHS],
                          [b2] ])  # fmt: skip

        # print(" ")
        # ppm.ppm(RHS,"RHS top 2 blocks")
        # print(" ")

        for t in range(1, T):
            # print("Entering loop with t={0}, RHS.shape = {1},".format(t,RHS.shape))
            # print("c({0}).shape = {1}".format(t,self.c(t).shape))
            # print("f({0}).shape = {1}".format(t,self.f(t).shape))
            # ppm.ppm(self.c(t),"c({:d})".format(t))
            # ppm.ppm(self.f(t),"f({:d})".format(t))

            RHS = np.block([ [RHS],
                             [-self.c(t)],
                             [-self.f(t)]  ])  # fmt: skip

        t = T
        # print("Final time t={0}, RHS.shape = {1},".format(t,RHS.shape))
        # print("c({0}).shape = {1}".format(t,self.c(t).shape))
        # ppm.ppm(self.c(t),"c({:d})".format(t))

        RHS = np.block([ [RHS],
                         [-self.c(t)[:n,:]]  ])  # fmt: skip

        # ppm.ppm(RHS,"ultimate finished RHS")
        # print(" ")

        return RHS


    #########################################################################################################
    ## INTERFACE BETWEEN DIRECT APPROACH AND KKT METHOD ... CONVERT SOLUTION REPRESENTATIONS
    #########################################################################################################

    def xul2kkt(self, traj_x, traj_u, traj_lam):
        '''
        Stack optimal control-state trajectories and Lagrange multipliers
        into a tall column vector compatible with the KKT matrix system
        detailed in the accompanying theoretical notes.
        '''
        n = traj_x.shape[0]
        m = traj_u.shape[0]
        T = traj_u.shape[2]
        # First pile interior x's atop u's and lambda's:
        midpart = np.vstack((traj_x[:, 0, 1:T], traj_u[:, 0, 1:T], traj_lam[:, 0, 1:T]))
        # Next stack those tall columns on top of each other, working left to right
        corecol = midpart.reshape(
            ((T - 1) * (m + n + n), 1), order="F"
        )  # What a minefield.
        # Finally stitch on the short pieces for the top and bottom.
        result = np.vstack(
            (traj_u[:, [0], 0], traj_lam[:, [0], 0], corecol, traj_x[:, [0], T])
        )
        return result

    def kkt2xul(self, x0, KKTsol):
        """
        Break apart the tall column-vector solving the KKT system into
        separate containers of standard form for the state, control,
        and multipliers. Notes:
          * the KKT solution does not include the initial state x0,
            so that must be provided separately, and
          * the control dimension m and step-count T cannot be reliably
            inferred from x0 and KKTsol only, so we look them up in the
            known characteristics of the system object.
        """
        n = self.n
        m = self.m
        T = self.T
        KKTx = np.zeros((n, 1, T + 1))
        KKTu = np.zeros((m, 1, T))
        KKTl = np.zeros((n, 1, T))

        KKTx[:, [0], 0] = x0.reshape(n, 1)
        for r in range(0, T):
            KKTu[:, [0], r] = KKTsol[r * (m + 2 * n) : r * (m + 2 * n) + m, :]
            KKTl[:, [0], r] = KKTsol[(m + r * (m + 2 * n)) : (m + r * (m + 2 * n) + n), :][:]
            KKTx[:, [0], r + 1] = KKTsol[
                (m + n + r * (m + 2 * n)) : (m + n + r * (m + 2 * n) + n), :
            ][:]
        return (KKTx, KKTu, KKTl)

    #########################################################################################################
    ## SENSITIVITY ANALYSIS -- GRADIENTS OF THE MINIMUM VALUE FUNCTION "V" w.r.t. DYNAMIC ELEMENTS
    #########################################################################################################
    def sensVsetup(self, x0):
        # The sensitivity matrix for V is built from the
        # solutions and multipliers in the nominal problem.
        # Calculate those efficiently by backward-forward pass,
        # repackage result into KKT format, and build the matrix.
        # (Efficiency concern: actually forming that sparse
        # rank-1 matrix wastes time and storage. Maybe fix later.)

        bestx, bestu, bestlam = self.bestxul(x0)
        kktsol = self.xul2kkt(bestx,bestu,bestlam)
        return kktsol @ kktsol.T

    #########################################################################################################
    def gradx0V(self, x0):
        grad = self.V_mats[:, :, 0] @ x0 + self.v_vecs[:, [0], 0]
        return grad

    #########################################################################################################
    def gradctV(self, t, x0):
        T = self.Tmax
        m = self.m
        n = self.n

        _,_,_ = self.bestxul(x0)  # Refresh nominal solution and multipliers

        tlist = [t]
        if self.c_vecs.ndim == 2:
            # In the autonomous case, add contributions for *all* relevant t
            tlist = [t for t in range(T + 1)]

        grad = np.zeros((n + m, 1))

        for t in tlist:
            if 0 < t and t < T:
                grad[0:n, 0] += self.sol_x[:, 0, t]
                grad[n : n + m, 0] += self.sol_u[:, 0, t]

            if t == 0:
                grad[n : n + m, 0] += self.sol_u[:, 0, t]

            if t == T:
                grad[0:n, 0] += self.sol_x[:, 0, t]

        return grad

    #########################################################################################################
    def gradCtV(self, t, x0):
        T = self.Tmax
        m = self.m
        n = self.n

        S = self.sensVsetup(x0)

        tlist = [t]
        if self.C_mats.ndim == 2:
            # In the autonomous case, add contributions for *all* relevant t
            tlist = [t for t in range(T + 1)]

        grad = np.zeros((m + n, m + n))

        for t in tlist:
            if 0 < t and t < T:
                i0 = m + n + (t - 1) * (m + 2 * n)
                j0 = i0
                Sblock = S[i0 : i0 + m + n, j0 : j0 + m + n]
                # if printlevel > 1:
                #    ppm.ppm(Sblock,f"Sblock_{t} starts at ({i0},{j0}),")
                grad += Sblock / 2.0

            if t == 0:
                # if printlevel > 1:
                #    print(f"For t={t}, focus on R_{t}.")
                Sblock = S[0:m, 0:m]
                dVdC = np.block(
                    [[np.zeros((n, n)), np.zeros((n, m))], [np.zeros((m, n)), Sblock]]
                )
                grad += dVdC / 2.0

            if t == T:
                # if printlevel > 1:
                #    print(f"For t={t}, focus on Q_{t}.")
                Sblock = S[
                    (2 * n + m) * T - n :,
                    (2 * n + m) * T - n :,
                ]
                dVdC = np.block(
                    [[Sblock, np.zeros((n, m))], [np.zeros((m, n)), np.zeros((m, m))]]
                )
                grad += dVdC / 2.0

        return grad

    #########################################################################################################
    def gradFtV(self, t, x0):
        T = self.Tmax
        m = self.m
        n = self.n

        S = self.sensVsetup(x0)

        tlist = [t]
        if self.F_mats.ndim == 2:
            # In the autonomous case, add contributions for *all* relevant t
            tlist = [t for t in range(T)]

        grad = np.zeros((n, m + n))

        for t in tlist:
            if 0 < t and t < T:
                i0 = 2 * m + 2 * n + (t - 1) * (m + 2 * n)
                j0 = (m + n) + (t - 1) * (m + 2 * n)
                Sblock = S[i0 : i0 + n, j0 : j0 + m + n]
                STblock = S[j0 : j0 + m + n, i0 : i0 + n]
                grad += (Sblock + STblock.T) / 2.0

            if t == 0:
                lambda0 = self.sol_lam[:, [0], 0]
                Ablock = lambda0 @ x0.T
                ATblock = Ablock.T
                gradA = (Ablock + ATblock.T) / 2.0

                Bblock = S[m : m + n, 0:m]
                BTblock = S[0:m, m : m + n]
                gradB = (Bblock + BTblock.T) / 2.0
                grad += np.hstack((gradA, gradB))

        return grad

    #########################################################################################################
    def gradftV(self, t, x0):
        T = self.Tmax
        m = self.m
        n = self.n

        _,_,_ = self.bestxul(x0)  # Refresh nominal solution and multipliers

        tlist = [t]
        if self.f_vecs.ndim == 2:
            # In the autonomous case, add contributions for *all* relevant t
            tlist = [t for t in range(T)]

        grad = np.zeros((n, 1))

        for t in tlist:
            grad[0:n, 0] += self.sol_lam[:, 0, t]

        return grad


    #########################################################################################################
    ## SENSITIVITY ANALYSIS -- GRADIENTS OF GENERAL TRAJECTORY FUNCTION "W" w.r.t. DYNAMIC ELEMENTS
    #########################################################################################################
    #
    def sensWsetup(self, wx, wu, x0):
        if self.BOP is not None:
            tol = 100 * np.finfo(float).eps
            if (
                np.linalg.norm(wx - self.wx) < tol
                and np.linalg.norm(wu - self.wu) < tol
                and np.linalg.norm(x0 - self.sol_x0) < tol
            ):
                return self.BOP

        # We need the nominal solution for the KKT system.
        # The backward/forward passes find that efficiently
        _ = self.sensVsetup(x0)  # Build or just recall the optimal trajectory
        KKTsol0 = self.xul2kkt(self.sol_x, self.sol_u, self.sol_lam)

        # The corresponding sensitivity system has the same coefficient matrix.
        KKT = self.KKTmtx()
        # Stack coeffs of wx and wu into the right slots of the RHS vector.
        sensrhs = self.xul2kkt(wx, wu, np.zeros(wx.shape))
        senssol = np.linalg.solve(KKT, sensrhs)
        # ppm.ppm(senssol,"senssol, printed by function sensWsetup,")

        # Calculate the Big Olde Product and make it part of the system object
        self.BOP = -senssol @ KKTsol0.T
        self.wx = wx
        self.wu = wu
        self.senssol = senssol

        return self.BOP


    def gradx0W(self, wx, wu, x0):
        # - derivative of value W(x0) w.r.t. x0
        T = self.Tmax
        m = self.m
        n = self.n

        BOP = self.sensWsetup(wx, wu, x0)
        senslam0 = self.senssol[m : m + n, [0]]
        wx0 = wx[:, [0], 0]
        mtxA0 = self.F(0)[:, 0:n]
        grad = wx0 - mtxA0.T @ senslam0

        return grad

    def gradCtW(self, t, wx, wu, x0):
        # - derivative of value W(x0) w.r.t. C(t)
        T = self.Tmax
        m = self.m
        n = self.n

        BOP = self.sensWsetup(wx, wu, x0)

        tlist = [t]
        if self.C_mats.ndim == 2:
            # In the autonomous case, add contributions for *all* relevant t
            tlist = [t for t in range(T + 1)]

        grad = np.zeros((n + m, n + m))
        for t in tlist:
            if 0 < t and t < T:
                i0 = m + n + (t - 1) * (m + 2 * n)
                j0 = i0
                Sblock = BOP[i0 : i0 + m + n, j0 : j0 + m + n]
                gradCtL = Sblock / 2.0

            if t == 0:
                Sblock = BOP[0:m, 0:m]
                dWdC = np.block(
                    [[np.zeros((n, n)), np.zeros((n, m))], [np.zeros((m, n)), Sblock]]
                )
                gradCtL = dWdC / 2.0

            if t == T:
                Sblock = BOP[
                    (2 * n + m) * T - n :,
                    (2 * n + m) * T - n :,
                ]
                dWdC = np.block(
                    [[Sblock, np.zeros((n, m))], [np.zeros((m, n)), np.zeros((m, m))]]
                )
                gradCtL = dWdC / 2.0

            grad += gradCtL + gradCtL.T

        return grad

    def gradctW(self, t, wx, wu, x0):
        # - derivative of value W(x0) w.r.t. c(t)
        T = self.Tmax
        m = self.m
        n = self.n

        BOP = self.sensWsetup(wx, wu, x0)
        ppm.ppm(self.senssol, f"senssol, printed in gradctW,")

        tlist = [t]
        if self.c_vecs.ndim == 2:
            # In the autonomous case, add contributions for *all* relevant t
            tlist = [t for t in range(T + 1)]

        grad = np.zeros((n + m, 1))
        for t in tlist:
            if 0 < t and t < T:
                i0 = t * (2 * n + m) - n
                grad[0 : n + m, [0]] += -self.senssol[i0 : i0 + (n + m)]

            if t == 0:
                grad[n : n + m, [0]] += -self.senssol[0:m, [0]]

            if t == T:
                grad[0:n, [0]] += -self.senssol[(2 * n + m) * T - n :]

        return grad

    def gradFtW(self, t, wx, wu, x0):
        # derivative of value W(x0) w.r.t. F(t)
        T = self.Tmax
        m = self.m
        n = self.n

        BOP = self.sensWsetup(wx, wu, x0)

        tlist = [t]
        if self.F_mats.ndim == 2:
            # In the autonomous case, add contributions for *all* relevant t
            tlist = [t for t in range(T)]

        grad = np.zeros((n, n + m))
        for t in tlist:
            if 0 < t and t < T:
                i0 = 2 * m + 2 * n + (t - 1) * (m + 2 * n)
                j0 = (m + n) + (t - 1) * (m + 2 * n)
                Sblock = BOP[i0 : i0 + n, j0 : j0 + m + n]
                STblock = BOP[j0 : j0 + m + n, i0 : i0 + n]
                grad += 2.0 * ((Sblock + STblock.T) / 2.0)

            if t == 0:
                ss0 = self.senssol[m : m + n]
                gradA = -ss0 @ x0.T

                Bblock = BOP[m : m + n, 0:m]
                BTblock = BOP[0:m, m : m + n]
                gradB = Bblock + BTblock.T
                grad += np.hstack((gradA, gradB))

        return grad

    def gradftW(self, t, wx, wu, x0):
        # - derivative of value W(x0) w.r.t. f(t)
        T = self.Tmax
        m = self.m
        n = self.n

        BOP = self.sensWsetup(wx, wu, x0)

        # ppm.ppm(self.senssol,f"senssol, printed in gradftW,")

        tlist = [t]
        if self.f_vecs.ndim == 2:
            # In the autonomous case, add contributions for *all* relevant t
            tlist = [t for t in range(T)]

        grad = np.zeros((n, 1))  # Container for result to return

        for t in tlist:
            if 0 < t and t < T:
                i0 = t * (2 * n + m) + m
                grad[:, [0]] += -self.senssol[i0 : i0 + n, [0]]

            if t == 0:
                grad[:, [0]] += -self.senssol[m : m + n, [0]]

        return grad

    #########################################################################################################
    ## INTROSPECTION -- CALL THIS TO PRINT SYSTEM INFO
    #########################################################################################################
    def printself(self):
        print("*** SYSTEM DYNAMICS AND LOSS COEFFICIENTS IN GENERIC NOTATION ***")

        print(f"State dimension: n = {self.n:d}.")
        print(f"Input dimension: m = {self.m:d}.")
        print(f"Final time:      T = {self.Tmax:d}. ")
        print(" ")
        print(f"F_mats.shape = {self.F_mats.shape}.")
        print(f"f_vecs.shape = {self.f_vecs.shape}.")
        print(f"C_mats.shape = {self.C_mats.shape}.")
        print(f"c_vecs.shape = {self.c_vecs.shape}.")

        tlist = range(self.T)
        if self.F_mats.ndim == 2:
            tlist = [0]
            print("\nAll matrices F_t are the same. Here's a representative.")
        else:
            print(" ")
        for t in tlist:
            ppm.ppm(self.F(t), f"F_{t:d} = [A_{t:d}  B_{t:d}]")

        tlist = range(self.T)
        if self.f_vecs.ndim == 2:
            tlist = [0]
            print("\nAll vectors f_t are the same. Here's a representative.")
        else:
            print(" ")
        for t in tlist:
            ppm.ppm(self.f(t), f"f_{t:d}")

        tlist = range(1 + self.T)
        if self.C_mats.ndim == 2:
            print(
                f"\nAll matrices C_t are the same, except for tweaks to C_0 and C_{self.T}."
            )
            tlist = sorted(set([0, 1, self.T]))
        else:
            print(" ")
        for t in tlist:
            ppm.ppm(self.C(t), f"C_{t:d}")

        tlist = range(1 + self.T)
        if self.c_vecs.ndim == 2:
            print(
                f"\nAll vectors c_t are the same, except for tweaks to c_0 and c_{self.T}."
            )
            tlist = sorted(set([0, 1, self.T]))
        else:
            print(" ")
        for t in tlist:
            ppm.ppm(self.c(t), f"c_{t:d}")

        return

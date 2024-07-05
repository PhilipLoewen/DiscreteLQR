# DiscreteLQR

The title is the name of a Python class that encodes discrete-time Linear-Quadratic Regulator (LQR) problems.
An object in this class includes all the dynamic coefficients, and comes with methods to compute
the optimal trajectory for any given initial point. Also provided are methods that return gradients
of the optimal value with respect to each and every one of the coefficient matrices in the system definition.
Additional methods will return the gradients with respect to the system parameters for an arbitrary smooth
function defined in terms of the optimal trajectory.

Details of the design and notation are provided in the docstring for the `__init__` function.

Inspiration came from _OptNet: Differentiable Optimization as a Layer in Neural Networks_ by Brandon Amos and J. Zico Kolter,
online at https://arxiv.org/pdf/1703.00443.

This code refreshes and extends original work by Thiago da Cunha Vasco, https://github.com/thiagodcv.

## Dependencies

This is a pure Python module. It relies on Numpy, and on two other modules here on the same GitHub site,
- PhilipLoewen/PrettyPrinter, to display the results in an attractive format, and
- PhilipLoewen/TensorGradient, to double-check theoretical gradients by finite-difference approximations.

## Try it Yourself

To see what the module can do, download these files and the ones in the dependencies,
and then try running the testing scripts. Start with `test-LTI-optim.py`. Look at the
docstring for the `__init__` function in module DiscreteLQR for an explanation of the
output this produces, then try `test-LQR-optim.py` and move on to `test-gradV-LTI-all.py`
and `test-gradW-LTI-all.py`.

## Discussion

The approach to numerical accuracy here is completely naive. It seems to work very well when 
all the size-measures of interest (state dimension _n_, control dimension _m_, and final time _T_)
are not very large. Things go wrong in interesting ways when any of these gets too big.

## Shortcomings

Theoretical gradients for the optimal value function _V_ are calculated efficiently,
as a byproduct of the forward-backward solution for the relevant trajectory.
This is not the case for the general objective function _W_: for this, all gradients
are calculated by using the linear algebra methods built into Numpy to solve a system
of linear equations based on the KKT matrix. Perhaps some day this branch of the code
will be replaced with something more accurate and efficient.

_Philip D Loewen, 2024-07-05_

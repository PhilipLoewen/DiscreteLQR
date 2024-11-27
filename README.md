# DiscreteLQR

The title is the name of a Python class that encodes 
discrete-time Linear-Quadratic Regulator (LQR) problems.
An object in this class includes all the dynamic coefficients, 
and comes with methods to compute the optimal trajectory 
for any given initial point. 
But that's just the beginning!

Also provided are methods that return _gradients of the problem's optimal value, V,_
with respect to each and every one of the coefficient matrices in the system definition.
Additional methods will return the gradients with respect to the system parameters 
for an _arbitrary smooth function W_ defined in terms of the optimal trajectory.

Details of the design and notation are provided in the docstring for the `__init__` function.

Inspiration came from 
_OptNet: Differentiable Optimization as a Layer in Neural Networks_ 
by Brandon Amos and J. Zico Kolter,
online at https://arxiv.org/pdf/1703.00443.

This code refreshes and extends original work by Thiago da Cunha Vasco, https://github.com/thiagodcv.

## Dependencies

This is a pure Python module. 
It relies on Numpy, and on two other modules here on the same GitHub site,
- PhilipLoewen/PrettyPrinter, to display the results in an attractive format, and
- PhilipLoewen/TensorGradient, to double-check theoretical gradients by finite-difference approximations.

## Try it Yourself

To see what the module can do, download these files and the ones in the dependencies, 
and then try running the testing scripts.
Start with `test-LTI-optim.py`.
Look at the docstring for the `__init__` function in module DiscreteLQR 
for an explanation of the output this produces, 
then try `test-LQR-optim.py` and move on to `test-gradV-LTI-all.py` and `test-gradW-LTI-all.py`.

## Accuracy

For a system with _n_ state components, _m_ control components, and _T_ time steps,
a KKT formulation of the problem will generate a square matrix with _N=(2n+m)T_ columns.
When _N_ is not too large, the methods here work quite well.
However, things go wrong in interesting ways when _N_ is large.

## Efficiency

The KKT matrix mentioned above is block-tridiagonal and sparse.
This makes it possible to solve the KKT equations using
an efficent two-pass method.
In fact, keeping track of the KKT structure makes it possible to
proceed without ever explicitly forming (or writing, or storing)
the fully KKT matrix.
The code takes advantage of this,
but it also includes a function that will build and print the matrix.
(That was useful in the early stages of testing.)

## Documentation

I have some mathy notes on the ideas and symbols that come up in the code.
Some day I might clean them up and post them.

## Alternatives

Several authors have implementations of these ideas.
This one is totally independent, so it may provide a
useful comparison to check the consistency of different
approaches.
It would be interesting to compare the results available
here with automated sensitivity analysis tools like the
ones built into these recent packages from DeepMind:
[jax](https://github.com/jax-ml/jax) and
[optax](https://github.com/google-deepmind/optax).

_Philip D Loewen, 2024-11-27_

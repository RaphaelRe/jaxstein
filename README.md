# Implementation of Stein Variational inference using JAX

A method proposed by [Liu & Wang (2016)](https://arxiv.org/abs/1608.04471). 
The idea is, to fit a set of particles to a given density. The algorithm only uses the derivative of the log-density.
Therefore, it is not necessary to thik about the normalizing constant. The implementation relies on JAX's [automatic differentiation](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html).
One has only to provide a log-density function using [the JAX version of numpy](https://jax.readthedocs.io/en/latest/jax.numpy.html).

_At the moment, it is just a small side project as I like the method._


The idea of the algorithm can roughly summarized by this image:




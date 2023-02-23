# Implementation of Stein Variational inference using JAX

A method proposed by [Liu & Wang (2016)](https://arxiv.org/abs/1608.04471). The idea is, to fit a set of particles to a given density. The algorithm only uses the derivative of the log-density. Therefore, it is not necessary to thik about the normalizing constant. 

The implementation relies on JAX's [automatic differentiation](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html).
One has only to provide a log-density function using [the JAX version of numpy](https://jax.readthedocs.io/en/latest/jax.numpy.html).

_At the moment, it is just a small side project as I like the method._


The idea of the algorithm can roughly summarized by this image:

![steinVI](https://user-images.githubusercontent.com/33098451/220856070-2f1aba48-19cc-43c4-bf80-e0368baf889a.gif)


## Usage
The method reuqires a log-density function and a kernel. The rest is done directly by the jaxstein.


```python
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn
from jax.random import PRNGKey
import jaxstein 

# define a log-density and a kernel
mu = jnp.ones(2)
sigma = jnp.eye(2)

def logdensity(x: jnp.array):
    return mvn.logpdf(x, mu, sigma)


def kernel(x: jnp.ndarray, y: jnp.ndarray, gamma: float = 1.) -> float:
    return jnp.exp(-gamma * jnp.sum((x - y) ** 2))


stein = jaxstein.SteinVi(logdensity, kernel)  # define algorithm object with log-density and the kernel
stein.init(30, 2, PRNGKey(1))  # initial particle positions
stein.fit(20)  # make 20 steps
p_fit = stein.get_particles() 
print(p_fit.mean(axis=0))

```


import jax.numpy as jnp
import jax.random as random
from jax.random import PRNGKey
from jax.scipy.stats import multivariate_normal as mvn

from sklearn.datasets import make_biclusters

import sys
sys.path.append('..')
import jaxstein 
from jaxstein.util import rbf_kernel

import matplotlib.pyplot as plt


#############################
# example is stolen from the BlackJAX repr - but with SteinVI as inference method
# original can be found here: https://github.com/blackjax-devs/sampling-book/blob/main/book/models/logistic_regression.md
#############################



num_points = 50
X, rows, cols = make_biclusters(
    (num_points, 2), 2, noise=0.6, random_state=314, minval=-3, maxval=3
)
y = rows[0] * 1.0  # y[i] = whether point i belongs to cluster 1

Phi = jnp.c_[jnp.ones(num_points)[:, None], X]
N, M = Phi.shape


def sigmoid(z):
    return jnp.exp(z) / (1 + jnp.exp(z))


def log_sigmoid(z):
    return z - jnp.log(1 + jnp.exp(z))


def logdensity_fn(w, alpha=1.0):
    """The log-probability density function of the posterior distribution of the model."""
    log_an = log_sigmoid(Phi @ w)
    an = Phi @ w
    log_likelihood_term = y * log_an + (1 - y) * jnp.log(1 - sigmoid(an))
    prior_term = alpha * w @ w / 2

    return -prior_term + log_likelihood_term.sum()

logdensity_fn(random.multivariate_normal(PRNGKey(1), 0.1 + jnp.zeros(M), jnp.eye(M)))




stein = jaxstein.SteinVi(logdensity_fn, rbf_kernel, epsilon=0.001)
stein.init(100, M, PRNGKey(1))
stein.fit(500)
p = stein.get_particles()
p.mean(axis=0)


xmin, ymin = X.min(axis=0) - 0.1
xmax, ymax = X.max(axis=0) + 0.1
step = 0.1
Xspace = jnp.mgrid[xmin:xmax:step, ymin:ymax:step]
_, nx, ny = Xspace.shape
Phispace = jnp.concatenate([jnp.ones((1, nx, ny)), Xspace])
Z_mcmc = sigmoid(jnp.einsum("mij,sm->sij", Phispace, p))
Z_mcmc = Z_mcmc.mean(axis=0)
plt.contourf(*Xspace, Z_mcmc, cmap='Blues')
plt.scatter(*X.T, c="orange")
plt.xlabel(r"$X_0$")
plt.ylabel(r"$X_1$")
plt.show()



import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn
from jax.random import PRNGKey
import jaxstein 

import matplotlib.pyplot as plt


# basically intro example with viz
mu = jnp.ones(2)
sigma = jnp.eye(2)

def logdensity(x: jnp.array):
    return mvn.logpdf(x, mu, sigma)


def kernel(x: jnp.ndarray, y: jnp.ndarray, gamma: float = 1.) -> float:
    return jnp.exp(-gamma * jnp.sum((x - y) ** 2))


stein = jaxstein.SteinVi(logdensity, kernel)
stein.init(30, 2, PRNGKey(1))
p_init =stein.get_particles()  # get initial particles to compare them later
stein.fit(20)
p_fit = stein.get_particles()

p_fit.mean(axis=0)


x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
Z = mvn.pdf(pos, mean=mu, cov=sigma)

plt.contourf(X,Y,Z, cmap='Blues')
plt.scatter(p_init[:,0], p_init[:,1], color="orange")
plt.scatter(p_fit[:,0], p_fit[:,1], color="red")
plt.show()



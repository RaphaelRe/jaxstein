import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn
from jax.random import PRNGKey
import jaxstein 
from jaxstein.util import rbf_kernel

import matplotlib.pyplot as plt


### basically intro example with viz

# define logdensity and kernel
mu = jnp.ones(2)
sigma = jnp.eye(2)

def logdensity(x: jnp.array):
    return mvn.logpdf(x, mu, sigma)

kernel = jaxstein.util.rbf_kernel


stein = jaxstein.SteinVi(logdensity, rbf_kernel)
stein.init(30, 2, PRNGKey(1))
p_init =stein.get_particles()  # get initial particles to compare them later
stein.fit(20)
p_fit = stein.get_particles()

print("Estimated mean values:")
print(p_fit.mean(axis=0))
print("Estimated covariance:")
print(jnp.cov(p_fit, rowvar=False))


# vizualize the density with initial starting points and final estimate
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
Z = mvn.pdf(pos, mean=mu, cov=sigma)

plt.contourf(X,Y,Z, cmap='Blues')
plt.scatter(p_init[:,0], p_init[:,1], color="green")
plt.scatter(p_fit[:,0], p_fit[:,1], color="orange")
plt.show()



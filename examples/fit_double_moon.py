import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn
from jax.random import PRNGKey
import jax
import sys
sys.path.append('..')
import jaxstein 
from jaxstein.util import rbf_kernel
from jaxstein.util import rbf_kernel_auto_h

import matplotlib.pyplot as plt



# double moon log density
def logdensity(x):
    eplus = jnp.exp(-.5 * (x[0] + 3) ** 2)
    eminus = jnp.exp(-.5 * (x[0] - 3) ** 2)
    nx = jnp.linalg.norm(x)
    pre = jnp.exp(-5 * (nx - 2) ** 2)
    tmp = pre * (eplus + eminus)
    return jnp.log(tmp)




kernel = jaxstein.util.rbf_kernel
kernel = rbf_kernel_auto_h


# import pudb; pu.db
stein = jaxstein.SteinVi(logdensity, kernel, epsilon=0.1)
stein.init_particle_positions(num_particles=30, p=2, rng_key=PRNGKey(1))
p_init = stein.get_particles()  # get initial particles to compare them later
stein.fit(200)
p_fit = stein.get_particles()

print("Estimated mean values:")
print(p_fit.mean(axis=0))
print("Estimated covariance:")
print(jnp.cov(p_fit, rowvar=False))


# vizualize the density with initial starting points and final estimate
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

pos = np.empty((2,) + X.shape)
pos[1, :, :] = X
pos[0, :, :] = Y

# Z = mvn.pdf(pos, mean=mu, cov=sigma)
Z = jax.vmap(jax.vmap(lambda x: jnp.exp(logdensity(x)), 1), 2)(pos)



plt.contourf(X,Y,Z, cmap='Blues')
plt.scatter(p_init[:,0], p_init[:,1], color="lightblue")
plt.scatter(p_fit[:,0], p_fit[:,1], color="orange")
plt.show()



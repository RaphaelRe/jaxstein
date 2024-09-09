import numpy as np
import jax.numpy as jnp
from jax.random import PRNGKey
import jax
import sys
sys.path.append('..')
import jaxstein 
from jaxstein.util import rbf_kernel_auto_h

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


jax.config.update("jax_enable_x64", True)



# double moon log density
def logdensity(x):
    eplus = jnp.exp(-.5 * (x[0] + 3) ** 2)
    eminus = jnp.exp(-.5 * (x[0] - 3) ** 2)
    nx = jnp.linalg.norm(x)
    pre = jnp.exp(-5 * (nx - 2) ** 2)
    tmp = pre * (eplus + eminus)
    return jnp.log(tmp)




kernel = rbf_kernel_auto_h

num_particles = 40
iters = 5000
# import pudb; pu.db
stein = jaxstein.SteinVi(logdensity, kernel, epsilon=0.01, stochastic=True)
stein.init_particle_positions(num_particles=num_particles, p=2, rng_key=PRNGKey(1))
stein.particles = stein.particles * 0.1
p_init = stein.get_particles()  # get initial particles to compare them later
stein.fit(iters, jit_update=True, debug=False)

p_fit = stein.get_particles()
p_trajectories = stein.get_trajectories()



x = np.linspace(-3, 3, 1000)
y = np.linspace(-3, 3, 1000)
X, Y = np.meshgrid(x, y)

pos = np.empty((2,) + X.shape)
pos[1, :, :] = X
pos[0, :, :] = Y
Z = jax.vmap(jax.vmap(lambda x: jnp.exp(logdensity(x)), 1), 2)(pos)

tr = jnp.array(p_trajectories)






fig, ax = plt.subplots(figsize=(6, 4))
contour = ax.contourf(X, Y, Z, levels=30, cmap="Blues_r")  # Blues
scatter = ax.scatter(tr[0,:,0], tr[0,:,1], color='orange')
header_text = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')

def animate(frame):
    scatter.set_offsets(tr[frame,:,:])
    # plt.title(f"Iteration: {i}")
    header_text.set_text(f"Current Frame: {frame}")
    return scatter,

ani = FuncAnimation(fig=fig, func=animate, frames=range(0, iters, 10), interval=1, repeat=True, cache_frame_data=False)

plt.show()



t1, t2 = tr[200::10,:,0].flatten(), tr[200::10,:,1].flatten()
fig, ax = plt.subplots(2)
ax[0].scatter(jnp.arange(t1.__len__()), t1)
ax[1].scatter(jnp.arange(t1.__len__()), t2)
plt.show()

import seaborn as sns
sns.displot(x=t1, y=t2, kind="kde", rug=True)
plt.show()

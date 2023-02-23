import jax.numpy as jnp
from jax import grad
from jax import vmap
from jax import jit
from jax.lax import scan
from jax.random import normal
from jax.random import PRNGKey


from tqdm import tqdm

class SteinVi:
    """
    Stein Variational inference

    :log_density: a function representing a log-density
    :kernel: a function representing a kernel
    :epsilon: learning rate
    :repulse: float representing the repulse in XXXX
    """

    def __init__(self, log_density, kernel, epsilon: float = 1., repulse: float = 1.):
        ### TODO: checks
        ### TODO: checks
        ### TODO: checks
        self.log_density = log_density
        self.kernel = kernel 
        self.epsilon = epsilon
        self.repulse = repulse

        # gradients for inference
        self.grad_log_density = grad(log_density)
        self.grad_kernel = grad(kernel)


    def _update_particle_i(self, xi, xl, repulse, eps):

        kernel_i = vmap(lambda x: self.kernel(x, xi))(xl)
        grad_logdensity_i = vmap(self.grad_log_density)(xl)
        grad_kernel_i = vmap(lambda x: self.grad_kernel(x, xi))(xl)

        push = eps * jnp.mean(kernel_i[:, None] * grad_logdensity_i + grad_kernel_i * repulse, axis=0)
        xi += push
        # print(f"Push: {push}, New Point: {xi}")
        return xi


    def update_particles(self, xl, repulse, eps):
        def foo(xl, i, repulse=repulse):
            return xl.at[i,:].set(self._update_particle_i(xl[i,:], xl, repulse, eps)), None  # use unjitted version here as the whole function itself is jitted
        return scan(foo, xl, jnp.arange(xl.shape[0]))[0]


    def init(self, num_particles, p, rng_key, initializer=None, **kwargs):
        # TODO: Make a meaningful initialization
        if initializer is not None:
            self.particles = initializer(**kwargs)
        else:
            # atm, I start at zero mean gaussian
            self.particles = normal(rng_key, (num_particles, p))

    
    def get_particles(self):
        return self.particles

    def fit(self, iterations, jit_update=True):
        if jit_update:
            update_particles = jit(self.update_particles)
        else:
            update_particles = self.update_particles
     
        repulse = self.repulse
        eps = self.epsilon
        print(f"Start fitting process...")
        for _ in tqdm(range(iterations)):
            self.particles = update_particles(self.particles, repulse, eps)

        print("Finished!")


    # Viz functions?

        

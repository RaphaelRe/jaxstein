import jax.numpy as jnp
import jax
from jax import grad
from jax import vmap
from jax import tree_map
from jax import jit
from jax.lax import scan
from jax.random import normal
import jax.random as random
from jax.flatten_util import ravel_pytree

import matplotlib.pyplot as plt

from tqdm import tqdm

class SteinVi:
    """
    Stein Variational inference

    :log_density: a function representing a log-density
    :kernel: a function representing a kernel
    :epsilon: learning rate
    :repulse: float representing the repulse in XXXX
    """

    def __init__(self, log_density, kernel, stochastic: bool = True, epsilon: float = 1., repulse: float = 1.):
        ### TODO: checks
        ### TODO: checks
        ### TODO: checks
        self.log_density = log_density
        self.kernel = kernel 
        self.epsilon = epsilon
        self.repulse = repulse

        # gradient of density
        self.grad_log_density = grad(log_density)

        self.stochastic = stochastic
        ### TODO: check for (nested) pytree structure
        ### TODO: depending on that call the appropriate function for particle update 



    def _update_particle_i(self, xi, xl, repulse, eps):
        """
        This function updates the position of the i-th particle xi given all particles in matrix form xl (shape=(n, p), i.e. n particles in p dimensions)
        """

        kernel_i = vmap(lambda x: self.kernel(x, xi))(xl)
        grad_logdensity_i = vmap(self.grad_log_density)(xl)
        grad_kernel_i = vmap(lambda x: self.grad_kernel(x, xi))(xl)

        push = eps * jnp.mean(kernel_i[:, None] * grad_logdensity_i + grad_kernel_i * repulse, axis=0)
        # print(f"kernel: {kernel_i} \n")
        # print(f"grad_dens: {grad_logdensity_i} \n")
        # print(f"grad_kernel: {grad_kernel_i} \n")
        # print(f"Push: {push}, New Point: {xi} \n")
        return xi + push


    def _update_particle_i_pytree(self, xi, xl, repulse, eps):
        """
        This function updates the position of the i-th particle xi given all particles xl which are represented by a pytree (n leafs, each particle has dimension p)
        grad_logdensity is computed on the pytree. afterwards it is flatted out and then proceeds as for the normal function
        """
        kernel_i = vmap(lambda x: self.kernel(x, xi))(xl) 
        # grad_logdensity_i, _ = ravel_pytree(self.grad_log_density(self.unravel_foo(xi)))
        grad_logdensity_i = vmap(lambda x: ravel_pytree(self.grad_log_density(self.unravel_foo(x)))[0])(xl)
        grad_kernel_i = vmap(lambda x: self.grad_kernel(x, xi))(xl)
        push = eps * jnp.mean(kernel_i[:, None] * grad_logdensity_i + grad_kernel_i * repulse, axis=0)

        # print(f"kernel: {kernel_i} \n")
        # print(f"grad_dens: {grad_logdensity_i} \n")
        # print(f"grad_kernel: {grad_kernel_i} \n")
        # print(f"Push: {push}, New Point: {xi} \n")
        return xi + push



    def _update_particles(self, xl, repulse, eps, stochastic: bool, key = None, debug=False):
        """
        Function to update the positions of all particles.
        This function only works when the particles are represented as matrix, i.e. xl.shape = (n, p) with n particles in p dimensions 
        """

        ### TODO: adaptive step size (e.g. adam, adagrad, etc?)
        grad_log_density = vmap(self.grad_log_density)(xl)
        k, grad_k = self.kernel(xl)
        push = (k @ grad_log_density + repulse * grad_k) / xl.shape[0]

        stein_velocity =  eps * push

        diffusion = jax.lax.cond(stochastic,
                                lambda k, key, eps: self.calc_stochastic_diffusion(k, key) * jnp.sqrt(eps),
                                lambda k, key, eps: jnp.zeros(xl.shape), # adding zero pertubation
                                 k, key, eps)

        # if stochastic:
        #     diffusion = self.calc_stochastic_diffusion(k, key)
        #     velocity += diffusion * jnp.sqrt(eps)

        # if debug:
        #     print("\n === grad_log_density ===")
        #     print(grad_log_density.shape)
        #     print(grad_log_density)
        #
        #     print("\n === kernel ===")
        #     print(k.shape)
        #     print(k)
        #
        #     print("\n === grad kernel ===")
        #     print(grad_k.shape)
        #     print(grad_k)
        #
        #     if stochastic:
        #         print("\n === diffusion ===")
        #         print(diffusion)
        #         print(diffusion.shape)
        #
        #     print("\n === full velocity ===")
        #     print(velocity)
        #     print(velocity.shape)

        return xl + stein_velocity + diffusion 




    def calc_stochastic_diffusion(self, kernel_dists: jax.Array, key: jax.Array) -> jax.Array:
        """
        Calculates the stochastic diffusive noise for sSteinVI
        """
        chol = jnp.linalg.cholesky(kernel_dists)
        noise = normal(key, self.particles.shape)
        
        return chol @ noise * jnp.sqrt(2/self.num_particles)









    def _update_particles_pytree(self, xl, repulse, eps):
        """
        Function to update the positions of all particles.
        Each particle is a pytree. It is assumed that all particles are collected in a list
        :param xl: Matrix where each row represents a particle (each particle is assumed flattened pytree)
        """
        def foo(xl, i, repulse=repulse, eps=eps):
            return xl.at[i,:].set(self._update_particle_i_pytree(xl[i,:], xl, repulse, eps)), None  # use unjitted version here as the whole function itself is jitted
        # for i in range(xl.shape[0]): # loop for debug as scan is nasty to debug
            # xl = foo(xl, i)[0]
            # print(xl)
        # return xl
        return scan(foo, xl, jnp.arange(xl.shape[0]))[0] 


    def _update_particles_pytree(self, xl, repulse, eps):
        """
        Function to update the positions of all particles.
        Each particle is a pytree. It is assumed that all particles are collected in a list
        :param xl: Matrix where each row represents a particle (each particle is assumed flattened pytree)
        """
        grad_logdensity_i = vmap(lambda x: ravel_pytree(self.grad_log_density(self.unravel_foo(x)))[0])(xl)
        grad_log_density = vmap(self.grad_log_density)(xl)
        k, grad_k = self.kernel(xl)
        push = (k @ grad_log_density + repulse * grad_k) / xl.shape[0]
        # print(push)
        return xl + eps * push


    def init_particle_positions(self, initial_particles: jax.Array = None, initializer=None, num_particles=None, p=None, rng_key=None, **kwargs):
        # TODO: Make a meaningful initialization as dfault
        # TODO: Add support for objects from numpyro and so on. probably use ravel_prytree to flatten it out (1-dim). 
        #       Then intialize with p particles
        if initializer is not None and initial_particles is None:
            initial_particles = initializer(**kwargs)
        elif p is not None and initial_particles is None:
            print("no particles given. Trying to initialize as Gaussian...")
            initial_particles = normal(rng_key, (num_particles, p))

        self.particles = initial_particles 

        ### TODO: Only works for arrays! not pytrees!
        self.num_particles = self.particles.shape[0]
        self.num_dimension = self.particles.shape[1]

    
    def get_particles(self):
        return self.particles


    def fit(self, iterations, pytree=False, jit_update=True, debug=False):
        if jit_update:
            print("Jit compile update...")
            update_particles = jit(self._update_particles)
            print("Done")
        else:
            print("jit_update is set to False. Consider to jit, to get more speed")
            update_particles = self._update_particles
     
        repulse = self.repulse
        eps = self.epsilon

        print(f"Start fitting process...")

        if pytree:
            raise NotImplementedError("Stuff does not work atm")
            # self.particles is assumed to be list of pytrees
            _, self.unravel_foo = ravel_pytree(self.particles[0])  # flat the pytree out to get the unravel function (is used in the particle updates)
            xl_flat = jnp.stack(list(map(lambda x: ravel_pytree(x)[0], self.particles)))
            self.particles = xl_flat
            for _ in tqdm(range(iterations)):
                self.particles = self._update_particles_pytree(self.particles, repulse, eps)
            # self.particles = self.unravel_foo(particles)
            self.particles = list(map(self.unravel_foo, self.particles))
        else:
            for _ in tqdm(range(iterations)):
                self.particles = update_particles(self.particles, repulse, eps)

        print("Finished!")



    def fit(self, iterations, key=random.PRNGKey(42), pytree=False, jit_update=True, debug=False):
        if jit_update:
            print("Jit compile update...")
            update_particles = jit(self._update_particles)  # TODO: Fix some arguments like srochastic here using partial?
            print("Done")
        else:
            print("jit_update is set to False. Consider to jit, to get more speed")
            update_particles = self._update_particles
     
        repulse = self.repulse
        eps = self.epsilon

        print(f"Start fitting process...")

        if pytree:
            raise NotImplementedError("Stuff does not work atm")
            # self.particles is assumed to be list of pytrees
            _, self.unravel_foo = ravel_pytree(self.particles[0])  # flat the pytree out to get the unravel function (is used in the particle updates)
            xl_flat = jnp.stack(list(map(lambda x: ravel_pytree(x)[0], self.particles)))
            self.particles = xl_flat
            for _ in tqdm(range(iterations)):
                self.particles = self._update_particles_pytree(self.particles, repulse, eps, debug)
            # self.particles = self.unravel_foo(particles)
            self.particles = list(map(self.unravel_foo, self.particles))
        else:
            keys = random.split(key, iterations)
            for i in tqdm(range(iterations)):
                self.particles = update_particles(self.particles, repulse, eps, self.stochastic, keys[i], debug)

        print("Finished!")

    # Viz functions?

        

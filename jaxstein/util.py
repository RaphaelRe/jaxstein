import jax.numpy as jnp


def rbf_kernel(x: jnp.ndarray, y: jnp.ndarray, gamma: float = 1.) -> float:
    return jnp.exp(-gamma * jnp.sum((x - y) ** 2))

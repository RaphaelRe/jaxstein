import jax.numpy as jnp
from jax import vmap

def rbf_kernel(x: jnp.ndarray, y: jnp.ndarray, gamma: float = 1.) -> float:
    return jnp.exp(-gamma * jnp.sum((x - y) ** 2))


def rbf_kernel_auto_h(X: jnp.ndarray):
    """
    calculates an RBF kernel for a given matrix with n x d where each point is given in a row. It also returns the derivative of the kernel for the given X.
    """
    dist_mat = vmap(lambda v: jnp.sum((v-X)**2, axis=1))(X)
    h = jnp.median(dist_mat)
    h = jnp.sqrt(0.5 * h / jnp.log(X.shape[0]+1))
    kernel_dist = jnp.exp( -dist_mat / h**2 / 2)

    # calc derivative by hand - no need for grad
    dxkxy = -kernel_dist @ X
    sumkxy = jnp.sum(kernel_dist, axis=1)
    dxkxy = vmap(lambda d, x: d + x * sumkxy, in_axes=1, out_axes=1)(dxkxy, X) / (h**2)

    return kernel_dist, dxkxy


# X = jnp.array([
        # [1,1],
        # [2,2],
        # [3,3],
        # ], dtype=float)
# Y = np.array([
        # [1,1],
        # [2,2],
        # [3,3],
        # ])

# rbf_kernel_auto_h(X)
# svgd_kernel(Y)



# vmap(lambda x, y: x + y, in_axes=1)(x, x)


# dxkxy = -np.matmul(Kxy, theta)
# sumkxy = np.sum(Kxy, axis=1)
# for i in range(theta.shape[1]):
    # dxkxy[:, i] = dxkxy[:,i] + np.multiply(theta[:,i],sumkxy)
# dxkxy = dxkxy / (h**2)





# import numpy as np
# from scipy.spatial.distance import pdist, squareform

# def svgd_kernel(theta, h = -1):
    # sq_dist = pdist(theta)
    # pairwise_dists = squareform(sq_dist)**2
    # if h < 0: # if h < 0, using median trick
        # h = np.median(pairwise_dists)  
        # h = np.sqrt(0.5 * h / np.log(theta.shape[0]+1))

    # # compute the rbf kernel
    # Kxy = np.exp( -pairwise_dists / h**2 / 2)

    # dxkxy = -np.matmul(Kxy, theta)
    # sumkxy = np.sum(Kxy, axis=1)
    # for i in range(theta.shape[1]):
        # dxkxy[:, i] = dxkxy[:,i] + np.multiply(theta[:,i],sumkxy)
    # dxkxy = dxkxy / (h**2)
    # return (Kxy, dxkxy)


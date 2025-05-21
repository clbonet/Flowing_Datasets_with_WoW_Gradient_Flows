import jax.numpy as jnp

from .sliced_wasserstein import sliced_wasserstein


def gaussian_kernel_sw(x, y, rng, n_projs=500, h=1, p=2):
    sw = sliced_wasserstein(x, y, rng, n_projs=n_projs, p=p)
    return jnp.exp(-sw / (2*h))


def laplace_kernel_sw(x, y, rng, n_projs=500, h=1):
    sw = sliced_wasserstein(x, y, rng, n_projs=n_projs, p=1)
    return jnp.exp(-sw / h)


def imq_kernel_sw(x, y, rng, n_projs=500, h=1):
    sw2 = sliced_wasserstein(x, y, rng, n_projs=n_projs, p=2)
    return 1 / jnp.sqrt(h + sw2)


def riesz_kernel_sw(x, y, rng, n_projs=500, p=2, r=1):
    sw2 = sliced_wasserstein(x, y, rng, n_projs=n_projs, p=p)
    return - sw2 ** (r/2)

import sys
import jax

import numpy as np
import jax.numpy as jnp

from copy import deepcopy
from tqdm import trange

sys.path.append("../")
from lib.kernels import gaussian_kernel_sw
from lib.sliced_wasserstein import sliced_wasserstein_value_and_grad
from lib.kernels import riesz_kernel_sw

def compute_full_pairwise_kernel(x, y, kernel):
    """
        Parameters
        ----------
        x: dict {key: array of size (n,d)}
        y: dict {key: array of size (m,d)}
        kernel: function taking as input (array of size (d,), array of size (d,))
    """
    result = np.zeros((len(x.keys()), len(y.keys())))
    for i, (k1, v1) in enumerate(x.items()):
        for j, (k2, v2) in enumerate(y.items()):
            result[i, j] = kernel(v1, v2)
    return result


def mmd_dict(x, y, kernel, x_weights=None, y_weights=None):
    """
        Parameters
        ----------
        x: dict {key: array of size (n,d)}
        y: dict {key: array of size (m,d)}
        kernel: function taking as input (array of size (d,), array of size (d,))
    """
    Kxx = compute_full_pairwise_kernel(x, x, kernel)
    Kyy = compute_full_pairwise_kernel(y, y, kernel)
    Kxy = compute_full_pairwise_kernel(x, y, kernel)

    n = len(x.keys())
    m = len(y.keys())

    if x_weights is None:
        x_weights = jnp.ones(n) / n
    if y_weights is None:
        y_weights = jnp.ones(m) / m

    cpt1 = jnp.einsum("n, nm, m", x_weights, Kxx, x_weights)
    cpt2 = jnp.einsum("n, nm, m", y_weights, Kyy, y_weights)
    cpt3 = jnp.einsum("n, nm, m", y_weights, Kxy, x_weights)

    return (cpt1+cpt2-2*cpt3)/2


def grad_mmd_dict(x, x_tgt, rng, sum_kernel_grad):
    """
        Parameters
        ----------
        x: dict {key: array of size (n,d)}
        x_tgt: dict {key: array of size (m,d)}
        sum_kernel_grad: function taking as input (dict, dict, rng)
    """
    n = len(x.keys())
    m = len(x_tgt.keys())

    grad_x = sum_kernel_grad(x, x, rng)
    grad_tgt = sum_kernel_grad(x, x_tgt, rng)

    nabla_mmd = {}
    for k in grad_x.keys():
        nabla_mmd[k] = grad_x[k] / n - grad_tgt[k] / m

    return nabla_mmd


def gaussian_kernel_sw_grad(xs, ys, rng, n_projs=500, h=1, p=2):
    """
        Parameters
        ----------
        xs: dict {key: array of size (n,d)}
        ys: dict {key: array of size (m,d)}
    """
    out = {}
    key = rng
    for i, (kx, x) in enumerate(xs.items()):
        cpt = 0
        for j, (ky, y) in enumerate(ys.items()):
            # rng, key = jax.random.split(rng)
            sw, grad_sw = sliced_wasserstein_value_and_grad(x, y, key, n_projs=n_projs, p=p)
            grad_kernel = - grad_sw * jnp.exp(-sw/(2*h)) / h
            cpt += grad_kernel

        out[kx] = cpt  # / j

    return out


def target_value_and_grad_gaussian_by_hand_dict(x, y, rng, h=0.1, n_projs=500):
    """
        Parameters
        ----------
        x: dict {key: array of size (n,d)}
        y: dict {key: array of size (m,d)}
    """
    master_key, key1, key2 = jax.random.split(rng, num=3)

    kernel = lambda k, l: gaussian_kernel_sw(k, l, key1, h=h)
    grad_kernel = lambda k, l, key: gaussian_kernel_sw_grad(k, l, key, h=h, n_projs=n_projs)

    loss = mmd_dict(x, y, kernel)
    grad = grad_mmd_dict(x, y, key2, grad_kernel)
    return loss, grad


def riesz_kernel_sw_grad(xs, ys, rng, n_projs=500, p=2):
    """
        Parameters
        ----------
        xs: dict {key: array of size (n,d)}
        ys: dict {key: array of size (m,d)}
    """
    out = {}
    key = rng
    for i, (kx, x) in enumerate(xs.items()):
        cpt = 0
        for j, (ky, y) in enumerate(ys.items()):
            # rng, key = jax.random.split(rng)
            grad_kernel = jax.grad(lambda z: riesz_kernel_sw(z, y, key))(x)
            # sw, grad_sw = sliced_wasserstein_value_and_grad(x, y, key, n_projs=n_projs, p=p)
            # grad_kernel = - grad_sw * jnp.exp(-sw/(2*h)) / h
            cpt += grad_kernel

        out[kx] = cpt  # / j

    return out


def target_value_and_grad_riesz_dict(x, y, rng, h=0.1, n_projs=500):
    """
        Parameters
        ----------
        x: dict {key: array of size (n,d)}
        y: dict {key: array of size (m,d)}
    """
    master_key, key1, key2 = jax.random.split(rng, num=3)

    kernel = lambda k, l: riesz_kernel_sw(k, l, key1)
    grad_kernel = lambda k, l, key: riesz_kernel_sw_grad(k, l, key,
                                                         n_projs=n_projs)

    loss = mmd_dict(x, y, kernel)
    grad = grad_mmd_dict(x, y, key2, grad_kernel)
    return loss, grad


def wasserstein_gradient_descent_dict(x0, x_tgt, target_value_and_grad, rng,
                                      n_epochs=101, lr=1):
    """
        Allows to have distribution with different number of samples

        Parameters
        ----------
        x0: dict {key: array of size (n,d)}
        x_tgt: dict {key: array of size (m,d)}
        target_value_and_grad: function taking as input (dict, dict, rng)
    """
    xk = deepcopy(x0)

    # L_particles = [x0]
    L_loss = []

    keys = jax.random.split(rng, num=n_epochs)
    pbar = trange(n_epochs)

    for e in pbar:
        l, grad = target_value_and_grad(xk, x_tgt, keys[e])
        for k in grad.keys():
            xk[k] = xk[k] - lr * grad[k]

        # L_particles.append(deepcopy(xk))
        L_loss.append(l)
        pbar.set_postfix_str(f"loss = {l:.3f}")

    return L_loss, xk  # L_particles

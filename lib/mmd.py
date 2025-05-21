import jax
import jax.numpy as jnp

from .sliced_wasserstein import sliced_wasserstein, sliced_wasserstein_value_and_grad
from .kernels import gaussian_kernel_sw, laplace_kernel_sw, imq_kernel_sw, riesz_kernel_sw


def mmd(x, y, kernel, x_weights=None, y_weights=None):
    vmapped_kernel = jax.vmap(kernel, in_axes=(0, None))
    pairwise_kernel = jax.vmap(vmapped_kernel, in_axes=(None, 0))
    
    Kxx = pairwise_kernel(x, x)
    Kyy = pairwise_kernel(y, y)
    Kxy = pairwise_kernel(x, y)

    n = x.shape[0]
    m = y.shape[0]

    if x_weights is None:
        x_weights = jnp.ones(n) / n
    if y_weights is None:
        y_weights = jnp.ones(m) / m

    cpt1 = jnp.einsum("n, nm, m", x_weights, Kxx, x_weights)
    cpt2 = jnp.einsum("n, nm, m", y_weights, Kyy, y_weights)
    cpt3 = jnp.einsum("n, nm, m", y_weights, Kxy, x_weights)

    return (cpt1+cpt2-2*cpt3)/2


def target_grad_mmd(x, y, kernel, rng, x_weights=None, y_weights=None, n_sample_batch=None):
    """
        Use autodifferentiation.
        
        Parameters
        ----------
        x: array of size (n_distr, n_samples_by_distr, d)
        y: array of size (m_distr, m_samples_by_distr, d)
        kernel: function taking x,y as input
        rng: key
        x_weights: array of size (n_distr,)
        y_weights: array of size (m_distr,)
        n_sample_batch: number of particles to sample from y, default None
    """
    n, n_sample, _ = x.shape
    m, m_sample, _ = y.shape

    if n_sample_batch is None:
        n_sample_batch = m_sample

    keys = jax.random.split(rng, num=m+1)
    y_tgt = jax.vmap(lambda z, key: jax.random.choice(key, z, (n_sample_batch,), replace=False))(y, keys[1:m+1])
    
    out, grad = jax.value_and_grad(lambda z: mmd(z, y_tgt, kernel, x_weights, y_weights))(x)
    return out, n * n_sample * grad ## multiply also by number of inner sample


def target_value_and_grad_gaussian(x, y, rng, x_weights=None, h=0.1, n_projs=500, n_sample_batch=None):
    master_key, key = jax.random.split(rng, num=2)
    kernel = lambda k, l: gaussian_kernel_sw(k, l, key, n_projs=n_projs, h=h)
    l, grad = target_grad_mmd(x, y, kernel, master_key, x_weights=x_weights, n_sample_batch=n_sample_batch)
    return l, grad


def sum_grad_kernel_sw(x, ys, rng, n_projs=500, h=1, p=2):
    sw_vmapped = jax.vmap(lambda y: sliced_wasserstein_value_and_grad(x, y, rng, n_projs=n_projs, p=2))
    sw, grad_sw = sw_vmapped(ys)    
    return - jnp.sum(grad_sw * jnp.exp(-sw[:,None,None]/(2*h)), axis=0) / h

def gaussian_kernel_sw_grad(xs, ys, rng, n_projs=500, h=1, p=2):
    out = jax.vmap(lambda x: sum_grad_kernel_sw(x, ys, rng, n_projs, h, p))(xs)
    return out

def grad_mmd(x, x_tgt, rng, sum_kernel_grad):
    n = x.shape[0]

    grad_x = sum_kernel_grad(x, x, rng)
    grad_tgt = sum_kernel_grad(x, x_tgt, rng)
    nabla_mmd = (grad_x - grad_tgt) / n

    return nabla_mmd

def target_value_and_grad_gaussian_by_hand(x, y, rng, h=0.1, n_projs=500):
    master_key, key1, key2 = jax.random.split(rng, num=3)
    
    kernel = lambda k, l: gaussian_kernel_sw(k, l, key1, h=h)
    grad_kernel = lambda k, l, key: gaussian_kernel_sw_grad(k, l, key, h=h, n_projs=n_projs)
    
    loss = mmd(x, y, kernel)
    grad = grad_mmd(x, y, key2, grad_kernel)
    return loss, grad


def target_value_and_grad_laplace(x, y, rng, h=0.5, n_projs=500, n_sample_batch=None):
    master_key, key = jax.random.split(rng, num=2)
    kernel = lambda k, l: laplace_kernel_sw(k, l, key, h=h, n_projs=n_projs)
    l, grad = target_grad_mmd(x, y, kernel, master_key, n_sample_batch=n_sample_batch)
    return l, grad


def target_value_and_grad_imq(x, y, rng, h=1, n_projs=500, n_sample_batch=None):
    master_key, key = jax.random.split(rng, num=2)
    kernel = lambda k, l: imq_kernel_sw(k, l, key, h=h, n_projs=n_projs)
    l, grad = target_grad_mmd(x, y, kernel, master_key, n_sample_batch=n_sample_batch)
    return l, grad


def target_value_and_grad_riesz(x, y, rng, x_weights=None, r=1, n_projs=500, n_sample_batch=None):
    master_key, key = jax.random.split(rng, num=2)
    kernel = lambda k, l: riesz_kernel_sw(k, l, key, r=r, n_projs=n_projs)
    l, grad = target_grad_mmd(x, y, kernel, master_key, x_weights=x_weights, n_sample_batch=n_sample_batch)
    return l, grad

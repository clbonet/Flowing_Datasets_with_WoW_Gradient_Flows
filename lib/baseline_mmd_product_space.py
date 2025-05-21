# Implementation of https://arxiv.org/abs/2302.00061

import jax
import ot

import jax.numpy as jnp
import numpy as np

from jax_tqdm import scan_tqdm
from functools import partial
from sklearn.decomposition import PCA

from .utils_bw import exp_bw, bures_wasserstein_batch
from .datasets import get_moments_from_dataset
from .utils_labels import get_labels_mmd_product


@partial(jax.jit, static_argnums=[2, 4])
def wasserstein_gradient_descent_product(x0, x_tgt, target_value_and_grad, rng,
                                         n_epochs=101, lr=1, m=0, v0=None):
    @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk, vk, key = carry
        master_key, subkey = jax.random.split(key)
        l, grad = target_value_and_grad(xk, x_tgt, subkey)

        v_x, v_mu, v_sigma = vk
        xk_x, xk_mu, xk_sigma = xk

        # Allows momentum
        v_x = grad[0] + m * v_x
        v_mu = grad[1] + m * v_mu
        # v_sigma = 2 * grad[2] + m * v_sigma

        xk_x = xk_x - lr * v_x
        xk_mu = xk_mu - lr * v_mu
        xk_sigma = exp_bw(xk_sigma, -2 * lr * grad[2])

        xk = (xk_x, xk_mu, xk_sigma)
        vk = (v_x, v_mu, v_sigma)

        return (xk, vk, master_key), l

    # Initial state
    if v0 is None:
        v0 = jnp.zeros_like(x0)

    # Use `lax.scan` to loop over epochs
    (xk, _, _), L_loss = jax.lax.scan(step, (x0, v0, rng), jnp.arange(n_epochs))

    return L_loss, xk


def mmd(x, y, kernel, x_weights=None, y_weights=None):
    vmapped_kernel = jax.vmap(kernel, in_axes=(0, None))
    pairwise_kernel = jax.vmap(vmapped_kernel, in_axes=(None, 0))

    Kxx = pairwise_kernel(x, x)
    Kyy = pairwise_kernel(y, y)
    Kxy = pairwise_kernel(x, y)

    n = x[0].shape[0]
    m = y[0].shape[0]

    if x_weights is None:
        x_weights = jnp.ones(n) / n
    if y_weights is None:
        y_weights = jnp.ones(m) / m

    cpt1 = jnp.einsum("n, nm, m", x_weights, Kxx, x_weights)
    cpt2 = jnp.einsum("n, nm, m", y_weights, Kyy, y_weights)
    cpt3 = jnp.einsum("n, nm, m", y_weights, Kxy, x_weights)

    return (cpt1+cpt2-2*cpt3)/2


def target_grad_mmd(x, y, kernel, rng, x_weights=None, y_weights=None,
                    n_sample_batch=None):
    """
        Use autodifferentiation.

        Parameters
        ----------
        x: tuple (x, mu, cov) with x array of size (n, d)
        y: tuple (y, mu, cov) with y array of size (m, d)
        kernel: function taking x,y as input
        rng: key
        x_weights: array of size (n,)
        y_weights: array of size (m,)
    """
    out, grad = jax.value_and_grad(lambda z: mmd(z, y, kernel, x_weights,
                                                 y_weights))(x)
    return out, grad


def gaussian_kernel(x, y, h=1):
    return jnp.exp(-jnp.sum(jnp.square(x-y), axis=-1) / (2*h))


def gaussian_product_kernel(x, mu_x, sigma_x, y, mu_y, sigma_y, h=1, h_mu=1,
                            h_sigma=1):
    kernel_pos = gaussian_kernel(x, y, h)
    kernel_mu = gaussian_kernel(mu_x, mu_y, h_mu)
    kernel_sigma = gaussian_kernel(sigma_x.reshape(-1,), sigma_y.reshape(-1,), h_sigma)
    return kernel_pos * kernel_mu * kernel_sigma


def target_value_and_grad_product(x, y, rng, x_weights=None, h=1, h_mu=1,
                                  h_sigma=1,
                                  n_sample_batch=None, noise_level=0.01):
    """
        Parameters
        ----------
        x: tuple of (x0, mu0, cov0)
        y: tuple of (y0, mu0, cov0)
        rng: PRNGKey
        x_weights: weights of x, default None
        h: float, bandwidth for Gaussian kernel of x
        h_mu: float, bandwidth for Gaussian kernel of mu
        h_sigma: float, bandwidth for Gaussian kernel of cov
        noise_level: level of noise to inject

        Output
        ------
        l: loss
        tuple: tuple of (grad w.r.t xk,grad w.r.t muk,grad w.r.t covk)
    """
    master_key, key_x, key_mu, key_sigma = jax.random.split(rng, num=4)
    n, d_x = x[0].shape
    n, d_mu_x = x[1].shape
    n, d_sigma_x, _ = x[2].shape

    def kernel(x, y):
        x, mu_x, sigma_x = x
        y, mu_y, sigma_y = y
        return gaussian_product_kernel(x, mu_x, sigma_x, y, mu_y, sigma_y, h, h_mu, h_sigma)

    noise_x = jax.random.normal(key_x, (n, d_x))
    noise_mu_x = jax.random.normal(key_mu, (n, d_mu_x))
    noise_sigma_x = jax.random.normal(key_sigma, (n, d_sigma_x, d_sigma_x))

    xk_x, mu_x, sigma_x = x
    xk_x = xk_x + noise_level * noise_x
    mu_x = mu_x + noise_level * noise_mu_x
    sigma_x = sigma_x + noise_level * noise_sigma_x

    x = (xk_x, mu_x, sigma_x)

    l, grad = target_grad_mmd(x, y, kernel, master_key, x_weights=x_weights,
                              n_sample_batch=n_sample_batch)
    return l, (grad[0] * n, grad[1] * n, grad[2] * n)


def mmd_product_flow(key_wgd, X_data_src, y_src, X_data_tgt, y_tgt,
                     reduced_dim=2, lr=1, m=0.9, n_epochs=5000):
    """
        Run the flow of the MMD with product Gaussian kernel from [1].
        The label distributions are embedded using PCA.

        Parameters
        ----------
        key_wgd: PRNGKey
        X_data_src: ndarray of shape (n_class, n_data_by_class, d)
        y_src: ndarray of shape (n_class, n_data_by_class)
        X_data_tgt: ndarray of shape (n_class, m_data_by_class, d)
        y_tgt: ndarray of shape (n_class, m_data_by_class)
        reduced_dim: int, dimension for PCA
        lr: float, step size
        m: float, momentum
        n_epochs: int, number of steps

        Outputs
        -------
        xk: ndarray of shape (10, n_data_by_class, reduced_dim)
        yk: ndarray of shape (10, n_data_by_class) (aligned with y_tgt)
        L_loss: ndarray of shape (n_epochs,), loss value

        [1] Hua, X., Nguyen, T., Le, T., Blanchet, J., & Nguyen, V. A. (2023).
        Dynamic flows on curved space generated by labeled data.
        arXiv preprint arXiv:2302.00061.
    """
    n_class, n_data_by_class, d = X_data_src.shape
    _, m_data_by_class, _ = X_data_tgt.shape

    X_concat = np.concatenate([X_data_src.reshape(-1, d), X_data_tgt.reshape(-1, d)], axis=0)
    pca = PCA(n_components=reduced_dim)
    X_concat_pca = pca.fit_transform(X_concat)

    X_data_src_emb = X_concat_pca[:n_class*n_data_by_class]
    X_data_tgt_emb = X_concat_pca[n_class*n_data_by_class:]
    X_data_src_emb = X_data_src_emb.reshape(n_class, n_data_by_class,
                                            reduced_dim)
    X_data_tgt_emb = X_data_tgt_emb.reshape(n_class, m_data_by_class,
                                            reduced_dim)

    _, _, mu_src, cov_src = get_moments_from_dataset(X_data_src_emb.reshape(-1, reduced_dim), y_src)
    mu_class_tgt, cov_class_tgt, mu_tgt, cov_tgt = get_moments_from_dataset(X_data_tgt_emb.reshape(-1, reduced_dim), y_tgt)

    if m_data_by_class == 1:
        cov_class_tgt = np.concatenate([np.eye(reduced_dim)[None] for k in range(n_class)], axis=0)
        cov_tgt = np.concatenate([np.eye(reduced_dim)[None] for k in range(n_class)], axis=0)

    X_src = (X_data_src.reshape(-1, d), mu_src, cov_src)
    X_tgt = (X_data_tgt.reshape(-1, d), mu_tgt, cov_tgt)

    def target_grad(x, y, key):
        return target_value_and_grad_product(x, y, key, h=100, h_mu=50,
                                             h_sigma=1000)

    v0 = (jnp.zeros_like(X_src[0]), jnp.zeros_like(X_src[1]), jnp.zeros_like(X_src[2]))

    L_loss, tuple_xk = wasserstein_gradient_descent_product(
        X_src, X_tgt, target_grad, key_wgd, lr=lr, m=m,
        n_epochs=n_epochs, v0=v0)

    xk_labels = get_labels_mmd_product(mu_class_tgt, cov_class_tgt, y_tgt,
                                       tuple_xk)

    z = np.clip(tuple_xk[0], 0, 1)

    xk = np.zeros((n_class, n_data_by_class, z.shape[-1]))
    yk = np.zeros((n_class, n_data_by_class))
    for k in range(n_class):
        xk[k] = z[xk_labels == k]
        yk[k] = np.ones((len(xk[k]),)) * k

    return xk, yk, L_loss

import jax
import optax

import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu

from functools import partial
from jax_tqdm import scan_tqdm
from tqdm import trange
from sklearn.decomposition import PCA

from ott.geometry import costs, pointcloud
from ott.geometry.costs import CostFn
from ott.solvers.linear import sinkhorn, acceleration
from ott.problems.linear import linear_problem
from typing import Any, Callable, Dict, Optional, Tuple

from .utils_bw import bures_wasserstein
from .datasets import get_moments_from_dataset
from .utils_labels import get_labels_mmd_product


@partial(jax.jit, static_argnums=[2, 4])
def wasserstein_gradient_descent_otdd(x0, x_tgt, target_value_and_grad, rng,
                                      n_epochs=101, lr=1, m=0, v0=None):
    """
        Parameters
        ----------
        - x0: tuple of the form (pos_x, mu_x, Sigma_x)
        - x_tgt: tuple of the form (pos_y, mu_y, Sigma_y)
        - target_value_and_grad: function taking x,y,rng as input and returning loss,gradient (e.g. target_value_and_grad_otdd)
        - rng
        - n_epochs
        - lr
        - m: momentum
        - v0: initial value momentum

        Returns
        -------
        - L_loss: list loss
        - xk: tuple (pos_x, mu_x, Sigma_x)
    """
    @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk, vk, key = carry
        master_key, subkey = jax.random.split(key)
        l, grad = target_value_and_grad(xk, x_tgt, subkey)

        v_x, v_mu, v_sigma = vk
        xk_x, xk_mu, xk_sigma = xk

        v_x = grad[0] + m * v_x  # Allows momentum
        v_mu = grad[1] + m * v_mu
        v_sigma = 2 * grad[2] + m * v_sigma

        xk_x = xk_x - lr * v_x
        xk_mu = xk_mu - lr * v_mu
        xk_sigma = xk_sigma - 2 * lr * v_sigma

        xk = (xk_x, xk_mu, xk_sigma)
        vk = (v_x, v_mu, v_sigma)

        return (xk, vk, master_key), l

    # Initial state
    if v0 is None:
        v0 = jnp.zeros_like(x0)

    # Use `lax.scan` to loop over epochs
    (xk, _, _), L_loss = jax.lax.scan(step, (x0, v0, rng), jnp.arange(n_epochs))

    return L_loss, xk


@partial(jax.jit, static_argnums=[2, 4, 5])
def wasserstein_gradient_descent_otdd_optax(x0, x_tgt, target_value_and_grad, rng, optimizer,
                                            n_epochs=101):
    """
        Parameters
        ----------
        - x0: tuple of the form (pos_x, mu_x, Sigma_x)
        - x_tgt: tuple of the form (pos_y, mu_y, Sigma_y)
        - target_value_and_grad: function taking x,y,rng as input and returning loss,gradient
        - rng
        - optimizer: optax optimizer

        Returns
        -------
        - L_loss: list loss
        - xk: tuple (pos_x, mu_x, Sigma_x)
    """
    @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk, state, key = carry
        master_key, subkey = jax.random.split(key)
        l, grad = target_value_and_grad(xk, x_tgt, subkey)

        ## TODO? Apply to each part of the tuple with 3 optimizers?
        updates, state = optimizer.update(grad, state, xk)
        xk = optax.apply_updates(xk, updates)

        return (xk, state, master_key), l

    opt_state = optimizer.init(x0)

    # Use `lax.scan` to loop over epochs
    (xk, _, _), L_loss = jax.lax.scan(step, (x0, opt_state, rng), jnp.arange(n_epochs))

    return L_loss, xk


def wasserstein_gradient_descent_otdd_optax_no_jit(x0, x_tgt, target_value_and_grad, rng, optimizer,
                                            n_epochs=101):
    """
        Parameters
        ----------
        - x0: tuple of the form (pos_x, mu_x, Sigma_x)
        - x_tgt: tuple of the form (pos_y, mu_y, Sigma_y)
        - target_value_and_grad: function taking x,y,rng as input and returning loss,gradient
        - rng
        - optimizer: optax optimizer

        Returns
        -------
        - L_loss: list loss
        - xk: tuple (pos_x, mu_x, Sigma_x)
    """
    opt_state = optimizer.init(x0)

    L_loss = []
    
    xk = x0
    state = opt_state
    key = rng

    pbar = trange(n_epochs)
    for k in pbar:
        master_key, subkey = jax.random.split(key)
        l, grad = target_value_and_grad(xk, x_tgt, subkey)

        updates, state = optimizer.update(grad, state, xk)
        xk = optax.apply_updates(xk, updates)

        L_loss.append(l)

    return L_loss, xk


def x_to_means_and_covs(x: jnp.ndarray,
                        dimension_init, dimension_gaussian: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Extract means and covariance matrices of Gaussians from raveled vector.

    From https://ott-jax.readthedocs.io/en/latest/_modules/ott/geometry/costs.html#SqEuclidean
    """
    x = jnp.atleast_2d(x)

    x_pos = x[:, :dimension_init]
    means = x[:, dimension_init:dimension_init + dimension_gaussian]

    covariances = jnp.reshape(
      x[:, dimension_init+dimension_gaussian:dimension_init + dimension_gaussian + dimension_gaussian ** 2], (-1, dimension_gaussian, dimension_gaussian)
    )
    return jnp.squeeze(x_pos), jnp.squeeze(means), jnp.squeeze(covariances)


@jtu.register_pytree_node_class
class otdd_cost(CostFn):
    """
        For dataset embedded in R^d\times BW(\R^d), computes \|x-y\|_2^2 + W_2^2(N(\mu_x,\Sigma_x), N(\mu_y,\Sigma_y))
    """
    def __init__(self, d_x: int, d_mu: int):
        super().__init__()
        self.d_x = d_x
        self.d_mu = d_mu

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
        pos_x, mu_x, sigma_x = x_to_means_and_covs(x, self.d_x, self.d_mu)
        pos_y, mu_y, sigma_y = x_to_means_and_covs(y, self.d_x, self.d_mu)

        dist2_pos = jnp.sum((pos_x-pos_y)**2, axis=-1)
        dist2_bw = bures_wasserstein(mu_x, mu_y, sigma_x, sigma_y)**2

        return dist2_pos + dist2_bw

    def tree_flatten(self):
        return (), (self.d_x, self.d_mu)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(*aux_data)


def compute_ot(x, y, d_x, d_mu):
    """
        Return optimal transport cost with OTDD groundcost

        Parameters
        ----------
        - x: array of shape (n, d_x + d_mu + d_mu^2)
        - y: array of shape (m, d_y + d_mu + d_mu^2)
        - d_x: dimension samples
        - d_mu: dimension gaussian

        Returns
        -------
        - Optimal transport cost        
    """
    geom = pointcloud.PointCloud(x, y,
                                 cost_fn=otdd_cost(d_x=int(d_x), d_mu=int(d_mu)),
                                 scale_cost="mean")

    ot_prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn(ot_prob, momentum = acceleration.Momentum(start=20, value=1.2))
    ot_sol = solver(ot_prob)

    return ot_sol.reg_ot_cost


def target_value_and_grad_otdd(x, y, rng):
    """
        Value and grad of OTDD

        Parameters
        ----------
        x: tuple of the form (pos_x, mu_x, Sigma_x)
        y: tuple of the form (pos_y, mu_y, Sigma_y)
        rng

        Returns
        -------
        l: loss
        grad: tuple of the gradients
    """
    master_key, key_x, key_mu, key_sigma = jax.random.split(rng, num=4)
    n, d_x = x[0].shape
    n, d_mu_x = x[1].shape
    n, d_sigma_x, _ = x[2].shape

    m, d_y = y[0].shape

    x_cat = jnp.concatenate([x[0], x[1], x[2].reshape(n, -1)], axis=-1)
    y_cat = jnp.concatenate([y[0], y[1], y[2].reshape(m, -1)], axis=-1)

    l, grad = jax.value_and_grad(lambda z: compute_ot(z, y_cat, d_x, d_mu_x))(x_cat)
    grad_x = grad[:, :d_x]
    grad_mu = grad[:, d_x:d_x+d_mu_x]
    grad_sigma = grad[:, d_x+d_mu_x:].reshape(-1, d_sigma_x, d_sigma_x)

    return l, (grad_x * n, grad_mu * n, grad_sigma * n)



def otdd_flow(key_wgd, X_data_src, y_src, X_data_tgt, y_tgt,
             reduced_dim=2, lr=1e-3, n_epochs=5000):
    """
        OTDD flow from [1] using an embedding of the labels as in [2].
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
        n_epochs: int, number of steps

        Outputs
        -------
        xk: ndarray of shape (10, n_data_by_class, reduced_dim)
        yk: ndarray of shape (10, n_data_by_class) (aligned with y_tgt)
        L_loss: ndarray of shape (n_epochs,), loss value

        [1] Alvarez-Melis, David, and Nicolò Fusi.
        "Dataset dynamics via gradient flows in probability space." 
        International conference on machine learning. PMLR, 2021.

        [2] Hua, X., Nguyen, T., Le, T., Blanchet, J., & Nguyen, V. A. (2023).
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

    optimizer = optax.adamw(lr)
    
    L_loss, tuple_xk = wasserstein_gradient_descent_otdd_optax(X_src, X_tgt, target_value_and_grad_otdd, key_wgd,
                                                               optimizer, n_epochs=n_epochs)

    xk_labels = get_labels_mmd_product(mu_class_tgt, cov_class_tgt, y_tgt,
                                       tuple_xk)
    
    z = np.clip(tuple_xk[0], 0, 1)
    
    xk = np.zeros((n_class, n_data_by_class, z.shape[-1]))
    yk = np.zeros((n_class, n_data_by_class))
    for k in range(n_class):
        xk[k] = z[xk_labels == k]
        yk[k] = np.ones((len(xk[k]),)) * k

    return xk, yk, L_loss

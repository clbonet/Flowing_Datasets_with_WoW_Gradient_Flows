import jax
import jax.numpy as jnp

from ott.geometry import costs, pointcloud
from ott.solvers import linear
from functools import partial
from tqdm import trange
from copy import deepcopy

from .mmd import mmd

def compute_ot(x, y):
    geom = pointcloud.PointCloud(x, y, cost_fn=None)
    solve_fn = linear.solve
    ot_object = solve_fn(geom)
    return ot_object.primal_cost

vmapped_compute_ot = jax.vmap(compute_ot, in_axes=(0, None))
pairwise_compute_ot = jax.vmap(vmapped_compute_ot, in_axes=(None, 0))


def semi_uot_mmd_objective(P, C, x, y, y_weights, kernel):
    n = P.shape[0]
    x_weights = jnp.einsum("mn, m -> n", P, jnp.ones(n))
    # x_weights = P.T @ jnp.ones(n)
    return jnp.sum(P*C) + mmd(x, y, kernel, x_weights, y_weights)


@partial(jax.jit, static_argnums=[2,5])
def optim_plan(xk, x_tgt, kernel, xk_weights, x_tgt_weights=None, n_epochs=100, lr=1, rng=None):
    n, n_inner, d = xk.shape
    
    C = jnp.fill_diagonal(pairwise_compute_ot(xk, xk), 0, inplace=False)

    ## Init with independent coupling
    P = jnp.outer(xk_weights, xk_weights)

    def objective(plan):
        return semi_uot_mmd_objective(plan, C, xk, x_tgt, x_tgt_weights, kernel)

    value_and_grad_objective = jax.value_and_grad(objective)

    L_loss = []

    for e in range(n_epochs):
        loss, grad = value_and_grad_objective(P)
        P_ = P * jnp.exp(-lr * grad)

        marginal = jnp.clip(P_ @ jnp.ones(n), 1e-8)
        P = jnp.diag(xk_weights / marginal) @ P_

        L_loss.append(loss)

    return P, L_loss


def wgd_over_plans(x0, x_tgt, kernel, target_value_and_grad, rng, 
                   n_epochs=501, n_inner_epochs=20, lr=0.1, inner_lr=1, start_optim_plan=0):
    pbar = trange(n_epochs)
    
    n = len(x0)
    xk = deepcopy(x0)
    xk_weights = jnp.ones(n) / n
    
    L_particles, L_weights = [deepcopy(x0)], [deepcopy(xk_weights)]
    L_loss = []

    keys = jax.random.split(rng, num=n_epochs)
        
    for e in pbar:
        key_grad, key_plans = jax.random.split(keys[e])
        l, grad = target_value_and_grad(xk, x_tgt, key_grad, xk_weights)
        xk = xk - lr * grad

        if e >= start_optim_plan:
            P, L_inner_loss = optim_plan(xk, x_tgt, kernel, xk_weights, n_epochs=n_inner_epochs, lr=inner_lr, rng=key_plans)
            xk_weights = P.T @ jnp.ones(n)
    
        L_particles.append(deepcopy(xk))
        L_weights.append(deepcopy(xk_weights))
        L_loss.append(l)

        pbar.set_postfix_str(f"loss = {l:.3f}")

    return L_particles, L_weights, L_loss

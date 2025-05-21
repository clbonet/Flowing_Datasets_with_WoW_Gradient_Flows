import jax
import optax

import jax.numpy as jnp

from copy import deepcopy
from tqdm import trange
from jax_tqdm import scan_tqdm
from functools import partial


@partial(jax.jit, static_argnums=[2,4])
def wasserstein_gradient_descent_jit(x0, x_tgt, target_value_and_grad, rng, n_epochs=101, lr=1, m=0):
    @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk, vk, key = carry
        master_key, subkey = jax.random.split(key)
        l, grad = target_value_and_grad(xk, x_tgt, subkey)
        vk = grad + m * vk  # Allows momentum
        xk = xk - lr * vk
        return (xk, vk, master_key), (xk, l)

    # Initial state
    v0 = jnp.zeros_like(x0)

    # Use `lax.scan` to loop over epochs
    (xk, _, _), L = jax.lax.scan(step, (x0, v0, rng), jnp.arange(n_epochs))

    L_particles, L_loss = L
    return L_loss, jnp.insert(L_particles, 0, x0, axis=0)


def wasserstein_gradient_descent(x0, x_tgt, target_value_and_grad, rng, n_epochs=101, lr=1, m=0):
    xk = deepcopy(x0)
    
    L_particles = [x0]
    L_loss = []

    keys = jax.random.split(rng, num=n_epochs)
    pbar = trange(n_epochs)

    vk = jnp.zeros(x0.shape)
    
    for e in pbar:
        l, grad = target_value_and_grad(xk, x_tgt, keys[e])
        vk = grad + m * vk  ## allows momentum
        xk = xk - lr * vk
        
        L_particles.append(deepcopy(xk))    
        L_loss.append(l)
        pbar.set_postfix_str(f"loss = {l:.3f}")

    return L_loss, L_particles


def wasserstein_gradient_descent_lbfgs(x0, x_tgt, target_value_and_grad, rng, n_epochs=101, lr=1):
    xk = deepcopy(x0)
    
    L_particles = [x0]
    L_loss = []

    keys = jax.random.split(rng, num=n_epochs)
    pbar = trange(n_epochs)

    opt = optax.scale_by_lbfgs()
    state = opt.init(xk)
    
    for e in pbar:
        l, grad = target_value_and_grad(xk, x_tgt, keys[e])
        u, state = opt.update(grad, state, xk)
        xk = xk - lr * u
        
        L_particles.append(deepcopy(xk))    
        L_loss.append(l)
        pbar.set_postfix_str(f"loss = {l:.3f}")

    return L_loss, L_particles

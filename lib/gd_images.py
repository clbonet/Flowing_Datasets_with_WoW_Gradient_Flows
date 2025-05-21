import jax
import optax
import jax.numpy as jnp

from jax_tqdm import scan_tqdm
from functools import partial


@partial(jax.jit, static_argnums=[2,4])
def wasserstein_gradient_descent(x0, x_tgt, target_value_and_grad, rng, n_epochs=101, lr=1, m=0):
    """
        Gradient descent with momentum
        
        Parameters
        ----------
        - x0: array of shape (n, d)
        - x_tgt: array of shape (n, d)
        - target_value_and_grad: function taking as input (x,y,key) and return (loss, grad)
        - rng
        - n_epochs
        - lr
        - m: momentum

        Returns
        -------
        - L_loss: list loss
        - xk: array of shape (n, d)
    """
    @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk, vk, key = carry
        master_key, subkey = jax.random.split(key)
        l, grad = target_value_and_grad(xk, x_tgt, subkey)
        vk = grad + m * vk  # Allows momentum
        xk = xk - lr * vk
        return (xk, vk, master_key), l

    # Initial state
    v0 = jnp.zeros_like(x0)

    # Use `lax.scan` to loop over epochs
    (xk, _, _), L_loss = jax.lax.scan(step, (x0, v0, rng), jnp.arange(n_epochs))

    return L_loss, xk


@partial(jax.jit, static_argnums=[2,4,7])
def wasserstein_gradient_descent_save(x0, x_tgt, target_value_and_grad, rng, n_epochs=101, lr=1, m=0, save_interval=1):
    """
        Gradient descent with momentum. Save samples every save_interval.
        
        Parameters
        ----------
        - x0: array of shape (n, d)
        - x_tgt: array of shape (n, d)
        - target_value_and_grad: function taking as input (x,y,key) and return (loss, grad)
        - rng
        - n_epochs
        - lr
        - m: momentum
        - save_interval: int, interval to save samples

        Returns
        -------
        - L_loss: list loss
        - L_xk: list of array of shape (n, d)
    """
    
    num_saves = (n_epochs // save_interval) + 1
    L_particles = jnp.zeros((num_saves,) + x0.shape)
    
    @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk, vk, key, L_particles = carry
        master_key, subkey = jax.random.split(key)
        l, grad = target_value_and_grad(xk, x_tgt, subkey)
        vk = grad + m * vk  # Allows momentum
        xk = xk - lr * vk

        save_particle = (iter_num % save_interval == 0)
        particle_idx = (iter_num // save_interval)
        L_particles = L_particles.at[particle_idx].set(jax.device_get(xk)) * save_particle + L_particles * (1 - save_particle)
        
        return (xk, vk, master_key, L_particles), l

    # Initial state
    v0 = jnp.zeros_like(x0)
    L_particles = L_particles.at[0].set(jax.device_get(x0))
    
    # Use `lax.scan` to loop over epochs
    (xk, _, _, L_xk), L_loss = jax.lax.scan(step, (x0, v0, rng, L_particles), jnp.arange(n_epochs))

    L_xk = L_xk.at[-1].set(jax.device_get(xk))

    return L_loss, L_xk


@partial(jax.jit, static_argnums=[2,4,5])
def wasserstein_gradient_descent_optax(x0, x_tgt, target_value_and_grad, rng, optimizer, n_epochs=101):
    """
        Gradient descent with optax optimizer
        
        Parameters
        ----------
        - x0: array of shape (n, d)
        - x_tgt: array of shape (n, d)
        - target_value_and_grad: function taking as input (x,y,key) and return (loss, grad)
        - rng
        - optimizer: optax optimizer
        - n_epochs

        Returns
        -------
        - L_loss: list loss
        - xk: array of shape (n, d)
    """
    @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk, state, key = carry
        master_key, subkey = jax.random.split(key)
        
        l, grad = target_value_and_grad(xk, x_tgt, subkey)
        updates, state = optimizer.update(grad, state, xk)
        xk = optax.apply_updates(xk, updates)

        return (xk, state, master_key), l

    opt_state = optimizer.init(x0)
    
    (xk, _, _), L_loss = jax.lax.scan(step, (x0, opt_state, rng), jnp.arange(n_epochs))

    return L_loss, xk

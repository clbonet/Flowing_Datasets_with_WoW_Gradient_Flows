import functools 
import jax
import jax.numpy as jnp


@jax.custom_jvp
def norm_1d(z):
    return jnp.abs(z)

@norm_1d.defjvp
def norm_1d_jvp(primals, tangents):
    z, = primals
    z_is_zero = jnp.all(jnp.logical_not(z))
    clean_z = jnp.where(z_is_zero, jnp.ones_like(z), z)
    primals, tangents = jax.jvp(
      functools.partial(jnp.abs),
      (clean_z,), tangents
    )
    return jnp.abs(z), jnp.where(z_is_zero, 0.0, tangents)


def get_displacement(
    source_proj: jnp.ndarray, 
    quantiles_proj_source: jnp.ndarray, quantiles_proj_target: jnp.ndarray, 
    percentiles: jnp.ndarray
):
    cdf_source = jax.vmap(jnp.interp, in_axes=(0, 0, None))(
        source_proj, quantiles_proj_source, percentiles
    )
    x_transported = jax.vmap(jnp.interp, in_axes=(0, None, 0))(
        cdf_source, percentiles, quantiles_proj_target,
    )
    return source_proj - x_transported


def sliced_wasserstein_from_quantiles(
    quantiles_proj_source: jnp.ndarray, quantiles_proj_target: jnp.ndarray, 
    percentiles: jnp.ndarray, rng: jax.random.PRNGKey, 
    p: float = 2
):   
    unif = jax.random.uniform(rng, quantiles_proj_source.shape)
    proj_source_icdf = jax.vmap(jnp.interp, in_axes=(0, None, 0))(
        unif, percentiles, quantiles_proj_source
    )
    proj_target_icdf = jax.vmap(jnp.interp, in_axes=(0, None, 0))(
        unif, percentiles, quantiles_proj_target
    )
    return jnp.mean(
        # jnp.abs(proj_source_icdf - proj_target_icdf) ** p
        norm_1d(proj_source_icdf - proj_target_icdf) ** p
    )


def sliced_wasserstein(
    source: jnp.ndarray, target: jnp.ndarray, rng: jax.random.PRNGKey,
    n_projs: int = 50, p: float = 2,
):
    # generate directions on the sphere
    directions = jax.random.normal(rng, (n_projs, source.shape[1]))
    directions = directions / jnp.linalg.norm(
        directions, axis=-1, keepdims=True
    )

    # slice source and targets along the directions
    proj_target = (target @ directions.T).T
    proj_source = (source @ directions.T).T

    # compute quantiles of sliced distributions
    percentiles = jnp.linspace(0, 1, 100)
    # add weights in percentile to take into account non uniform distributions??
    quantiles_proj_source = jnp.percentile(proj_source, percentiles*100, axis=1).T
    quantiles_proj_target = jnp.percentile(proj_target, percentiles*100, axis=1).T

    # compute sliced wasserstein value 
    return sliced_wasserstein_from_quantiles(
        quantiles_proj_source, quantiles_proj_target, percentiles, rng, p
    )


def sliced_wasserstein_value_and_grad(
    source: jnp.ndarray, target: jnp.ndarray, rng: jax.random.PRNGKey,
    n_projs: int = 50, p: float = 2,
    return_value: bool = True,
):

    # generate directions on the sphere
    directions = jax.random.normal(rng, (n_projs, source.shape[1]))
    directions = directions / jnp.linalg.norm(
        directions, axis=-1, keepdims=True
    )

    # slice source and targets along the directions
    proj_target = (target @ directions.T).T
    proj_source = (source @ directions.T).T

    # compute quantiles of sliced distributions
    percentiles = jnp.linspace(0, 1, 100)
    quantiles_proj_source = jnp.percentile(proj_source, percentiles*100, axis=1).T
    quantiles_proj_target = jnp.percentile(proj_target, percentiles*100, axis=1).T

    # compute sliced wasserstein value and gradients
    displacement  = get_displacement(
        proj_source, quantiles_proj_source, quantiles_proj_target, percentiles
    )
    grads = jnp.mean(
        displacement [:, :, None] * directions[:, None, :], axis=0
    )
    value = sliced_wasserstein_from_quantiles(
        quantiles_proj_source, quantiles_proj_target, percentiles, rng, p
    ) if return_value else None
    
    return value, grads

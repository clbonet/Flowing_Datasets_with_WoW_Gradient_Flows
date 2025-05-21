## Adapted from https://github.com/VICO-UoE/DatasetCondensation/blob/master/utils.py#L643

import jax
import jax.numpy as jnp
import numpy as np

def rand_flip(rng, x, prob_flip=0.5):
    bool_flip = jax.random.uniform(rng, 1)[0]

    def flip(x):
        return jnp.flip(x, -1)

    def id_func(x):
        return x
    
    return jax.lax.cond(bool_flip<prob_flip, flip, id_func, x)


def create_rotation_matrix(key, ratio_rotate=180.):
    """Creates a single 2D rotation matrix.
    
    Args:
        key: JAX PRNGKey
        ratio_rotate: Maximum rotation in degrees (default: 180)
    
    Returns:
        Array of shape (2, 3) containing rotation matrix
    """
    # Generate random angle between -ratio_rotate/2 and ratio_rotate/2 degrees
    theta = (jax.random.uniform(key) - 0.5) * 2 * ratio_rotate / 180. * jnp.pi
    
    # Create rotation matrix
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    
    # Create 2x3 matrix
    matrix = jnp.array([
        [cos_theta, -sin_theta, 0.],
        [sin_theta, cos_theta, 0.]
    ])
    
    return matrix

def affine_grid_single(theta, height, width):
    """Creates sampling grid for affine transformation.
    
    Args:
        theta: Transformation matrix (2, 3)
        height: Image height
        width: Image width
    
    Returns:
        Grid coordinates of shape (height, width, 2)
    """
    x = jnp.linspace(-1, 1, width)
    y = jnp.linspace(-1, 1, height)
    
    # Create grid of coordinates
    x_coords, y_coords = jnp.meshgrid(x, y)
    
    # Reshape coordinates and add homogeneous coordinate
    coords = jnp.stack([x_coords, y_coords, jnp.ones_like(x_coords)])
    
    # Apply transformation
    transformed = jnp.einsum('ij,jkl->ikl', theta, coords)
    
    # Reshape to expected format (H, W, 2)
    grid = jnp.stack([transformed[0], transformed[1]], axis=-1)
    
    return grid

def grid_sample_single(image, grid):
    """Samples image using grid coordinates.
    
    Args:
        image: Input image of shape (channels, height, width)
        grid: Sampling coordinates of shape (height, width, 2)
    
    Returns:
        Transformed image of same shape as input
    """
    channels, H, W = image.shape
    
    # Convert grid coordinates from [-1, 1] to [0, H/W]
    grid = (grid + 1) / 2
    grid = grid * jnp.array([W - 1, H - 1])  # Note: switched H and W here
    
    # Get integer and fractional parts
    grid_i = jnp.floor(grid).astype(jnp.int32)
    grid_f = grid - grid_i
    
    # Clip coordinates
    grid_i = jnp.clip(grid_i, 0, jnp.array([W - 1, H - 1]))  # Note: switched H and W
    
    # Get corner indices
    x0 = grid_i[..., 0]
    y0 = grid_i[..., 1]
    x1 = jnp.clip(x0 + 1, 0, W - 1)
    y1 = jnp.clip(y0 + 1, 0, H - 1)
    
    # Get weights
    wx1 = grid_f[..., 0:1]
    wy1 = grid_f[..., 1:2]
    wx0 = 1 - wx1
    wy0 = 1 - wy1
    
    # Sample and combine
    image = jnp.moveaxis(image, 0, -1)
    
    c00 = image[y0, x0]
    c01 = image[y1, x0]
    c10 = image[y0, x1]
    c11 = image[y1, x1]
    
    out = (c00 * (wx0 * wy0) + 
           c01 * (wx0 * wy1) + 
           c10 * (wx1 * wy0) + 
           c11 * (wx1 * wy1))
    
    return jnp.moveaxis(out, -1, 0)

def rand_rotate(key, x, ratio_rotate=15.):
    """Applies differentiable random rotation to a single image.
    
    Args:
        key: JAX PRNGKey
        x: Input image of shape (channels, height, width)
        ratio_rotate: Maximum rotation in degrees (default: 15°)
    
    Returns:
        Rotated image of same shape as input
    """
    # Create transformation matrix
    theta = create_rotation_matrix(key, ratio_rotate)
    
    # Create sampling grid
    grid = affine_grid_single(theta, x.shape[1], x.shape[2])
    
    # Sample image using grid
    return grid_sample_single(x, grid)


def rand_scale(rng, x, ratio_scale=1.2):
    master_key, key_sx, key_sy = jax.random.split(rng, 3)
    sx = jax.random.uniform(key_sx) * (ratio_scale - 1.0 / ratio_scale) + 1.0 / ratio_scale
    sy = jax.random.uniform(key_sy) * (ratio_scale - 1.0 / ratio_scale) + 1.0 / ratio_scale

    theta = jnp.array([[sx, 0,  0], [0,  sy, 0]])

    # Create sampling grid
    grid = affine_grid_single(theta, x.shape[1], x.shape[2])
    
    # Sample image using grid
    return grid_sample_single(x, grid)    


def rand_saturation(rng, x, ratio=2.0):
    master_key, key = jax.random.split(rng)
    rands = jax.random.uniform(key)

    x_mean = jnp.mean(x, axis=0, keepdims=True)
    x_transformed = (x-x_mean) * rands * ratio + x_mean
    return x_transformed


def rand_contrast(rng, x, contrast=0.5):
    master_key, key = jax.random.split(rng)
    randc = jax.random.uniform(key)

    x_mean = jnp.mean(x, keepdims=True)
    x_transformed = (x-x_mean) * (randc + contrast) + x_mean
    return x_transformed


def rand_crop(rng: jax.random.PRNGKey, x: jnp.ndarray, ratio_crop_pad: float=0.125) -> jnp.ndarray:
    """Random crop function for JAX.
    
    Args:
        rng: JAX random key
        x: Input image of shape (c, h, w)
        ratio_crop_pad: Ratio for padding and cropping
    
    Returns:
        Randomly cropped image of same shape as input
    """
    c, h, w = x.shape
    
    # Calculate shifts
    shift_x = int(w * ratio_crop_pad + 0.5)
    shift_y = int(h * ratio_crop_pad + 0.5)
    
    # Split RNG key for x and y translations
    rng, rng_x, rng_y = jax.random.split(rng, 3)
    
    # Generate random translations
    translation_x = jax.random.randint(rng_x, shape=(1, 1), minval=-shift_x, maxval=shift_x + 1)
    translation_y = jax.random.randint(rng_y, shape=(1, 1), minval=-shift_y, maxval=shift_y + 1)
    
    # Create meshgrid with distinct variable names
    grid_y, grid_x = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing='ij')
    
    # Apply translations and clamp values
    grid_x_shifted = jnp.clip(grid_x + translation_x + 1, 0, w + 1)
    grid_y_shifted = jnp.clip(grid_y + translation_y + 1, 0, h + 1)
    
    # First transpose to (h, w, c), pad, then transpose back
    x_hwc = jnp.transpose(x, (1, 2, 0))
    x_pad = jnp.pad(x_hwc, ((1, 1), (1, 1), (0, 0)), mode='constant')
    x_pad = jnp.transpose(x_pad, (2, 0, 1))
    
    # Create indices for sampling
    indices_c = jnp.arange(c)[:, None, None]
    indices_y = grid_y_shifted[None, :, :].repeat(c, axis=0)
    indices_x = grid_x_shifted[None, :, :].repeat(c, axis=0)
    
    # Sample from padded image using the shifted coordinates
    output = x_pad[indices_c, indices_y, indices_x]
    
    return output


def rand_cutout(rng, x, ratio_cutout=0.5):
    master_key, key_x, key_y = jax.random.split(rng, 3)

    cutout_size = int(x.shape[1] * ratio_cutout + 0.5), int(x.shape[2] * ratio_cutout + 0.5)
    
    offset_x = jax.random.randint(key_x, (1,), 0, x.shape[1] + (1 - cutout_size[0]%2))
    offset_y = jax.random.randint(key_y, (1,), 0, x.shape[2] + (1 - cutout_size[1]%2))

    grid_x, grid_y = jnp.meshgrid(jnp.arange(x.shape[1]), jnp.arange(x.shape[2]))
    grid_x = jnp.clip(grid_x + offset_x - cutout_size[0]//2, 0, x.shape[1]-1)
    grid_y = jnp.clip(grid_y + offset_y - cutout_size[1]//2, 0, x.shape[2]-1)

    mask = jnp.ones((x.shape[1], x.shape[2]))
    indices = (grid_x.ravel(), grid_y.ravel())
    mask = mask.at[indices].set(0)
    
    return x * mask[None]


def rand_brightness(rng, x, brightness=1):
    master_key, key = jax.random.split(rng)
    randb = jax.random.uniform(key)

    return x + (randb - 0.5) * brightness


def rand_color(rng, x, brightness=1, ratio_saturation=2.0, contrast=0.5):
    master_key, key_b, key_s, key_c = jax.random.split(rng, 4)
    x = rand_brightness(key_b, x, brightness)
    x = rand_saturation(key_s, x, ratio_saturation)
    x = rand_contrast(key_c, x, contrast)

    return x

    
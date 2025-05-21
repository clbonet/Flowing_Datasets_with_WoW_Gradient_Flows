import sys
import jax
import jax.numpy as jnp

from utils_augmentation import rand_flip, rand_rotate, rand_scale, rand_crop, \
      rand_cutout, rand_color

sys.path.append("../")
from lib.kernels import riesz_kernel_sw
from lib.classif_nn import ConvNet


# def mmd_dataset_distillation(rng, x, y, kernel, x_weights=None, y_weights=None, c=1, w_img=28, h_img=28):
#     """
#         Parameters
#         ----------
#         rng: key
#         x: array of size (n_distr, n_samples_by_distr, d)
#         y: array of size (m_distr, m_samples_by_distr, d)
#         kernel: function taking x,y as input
#         x_weights: array of size (n_distr,)
#         y_weights: array of size (m_distr,)
#     """
#     vmapped_kernel = jax.vmap(kernel, in_axes=(0, None))
#     pairwise_kernel = jax.vmap(vmapped_kernel, in_axes=(None, 0))
# 
#     master_key, key_nn = jax.random.split(rng)
#     model = ConvNet(key_nn, channel=c, im_size=(w_img, h_img))
# 
#     n_classes, n_samples, d = x.shape
#     m_classes, m_samples, d = y.shape
# 
#     def wrap_embedding(x):
#         x = jnp.reshape(x, (-1, w_img, h_img))
#         x_emb = model.embed(x)
#         return jnp.ravel(x_emb)
# 
#     wrap_embedding_vmap = jax.jit(jax.vmap(wrap_embedding))
# 
#     x_emb = wrap_embedding_vmap(x.reshape((-1, d))).reshape((n_classes, n_samples, -1))
#     y_emb = wrap_embedding_vmap(y.reshape((-1, d))).reshape((m_classes, m_samples, -1))
# 
#     Kxx = pairwise_kernel(x_emb, x_emb)
#     Kyy = pairwise_kernel(y_emb, y_emb)
#     Kxy = pairwise_kernel(x_emb, y_emb)
# 
#     n = x.shape[0]
#     m = y.shape[0]
# 
#     if x_weights is None:
#         x_weights = jnp.ones(n) / n
#     if y_weights is None:
#         y_weights = jnp.ones(m) / m
# 
#     cpt1 = jnp.einsum("n, nm, m", x_weights, Kxx, x_weights)
#     cpt2 = jnp.einsum("n, nm, m", y_weights, Kyy, y_weights)
#     cpt3 = jnp.einsum("n, nm, m", y_weights, Kxy, x_weights)
# 
#     return (cpt1+cpt2-2*cpt3)/2
# 
# 
# def target_grad_mmd_dataset_distillation(x, y, kernel, rng, x_weights=None, y_weights=None, n_sample_batch=None, c=1, w_img=28, h_img=28):
#     """
#         Use autodifferentiation.
# 
#         Parameters
#         ----------
#         x: array of size (n_distr, n_samples_by_distr, d)
#         y: array of size (m_distr, m_samples_by_distr, d)
#         kernel: function taking x,y as input
#         rng: key
#         x_weights: array of size (n_distr,)
#         y_weights: array of size (m_distr,)
#         n_sample_batch: number of particles to sample from y, default None
#     """
#     n, n_sample, _ = x.shape
#     m, m_sample, _ = y.shape
# 
#     if n_sample_batch is None:
#         n_sample_batch = m_sample
# 
#     master_key, key_subsampling = jax.random.split(rng)
#     keys = jax.random.split(key_subsampling, num=m+1)
#     y_tgt = jax.vmap(lambda z, key: jax.random.choice(key, z, (n_sample_batch,), replace=False))(y, keys[1:m+1])
# 
#     out, grad = jax.value_and_grad(lambda z: mmd_dataset_distillation(master_key, z, y_tgt, kernel, x_weights, y_weights, c, w_img, h_img))(x)
#     return out, n * n_sample * grad # multiply also by number of inner sample
# 
# 
# def target_value_and_grad_riesz_dataset_distillation(x, y, rng, n_batch_real, r=1, n_projs=500, c=1, w_img=28, h_img=28):
#     master_key, key = jax.random.split(rng, num=2)
#     kernel = lambda k, l: riesz_kernel_sw(k, l, key, r=r, n_projs=n_projs)
#     l, grad = target_grad_mmd_dataset_distillation(x, y, kernel, master_key, n_sample_batch=n_batch_real, c=c, w_img=w_img, h_img=h_img)
#     return l, grad

def mmd_dataset_distillation(
        rng, x, y, kernel, get_list_augmentation=lambda rng: [lambda z: z],
        x_weights=None, y_weights=None, c=1, w_img=28, h_img=28):
    """
        Parameters
        ----------
        rng: key
        x: array of size (n_distr, n_samples_by_distr, d)
        y: array of size (m_distr, m_samples_by_distr, d)
        kernel: function taking x,y as input
        get_list_augmentation: function taking as input a PRNGKey, and returns a list of augmentations
        x_weights: array of size (n_distr,)
        y_weights: array of size (m_distr,)
    """
    vmapped_kernel = jax.vmap(kernel, in_axes=(0, None))
    pairwise_kernel = jax.vmap(vmapped_kernel, in_axes=(None, 0))

    master_key, key_nn, key_choice_aug, key_aug = jax.random.split(rng, 4)

    # Sample an embedding
    model = ConvNet(key_nn, channel=c, im_size=(w_img, h_img))

    # Sample an augmentation
    list_augmentations = get_list_augmentation(key_aug)
    ind_aug = jax.random.randint(key_choice_aug, (), 0,
                                 len(list_augmentations))

    def apply_aug(z):
        return jax.lax.switch(ind_aug, list_augmentations, z)

    n_classes, n_samples, d = x.shape
    m_classes, m_samples, d = y.shape

    # Wrapper embedding + augmentation
    def wrap_embedding(x):
        x = jnp.reshape(x, (-1, w_img, h_img))
        x_aug = apply_aug(x)
        x_emb = model.embed(x_aug)
        return jnp.ravel(x_emb)

    wrap_embedding_vmap = jax.jit(jax.vmap(wrap_embedding))

    # Apply embedding
    x_emb = wrap_embedding_vmap(x.reshape((-1, d))).reshape((n_classes, n_samples, -1))
    y_emb = wrap_embedding_vmap(y.reshape((-1, d))).reshape((m_classes, m_samples, -1))

    Kxx = pairwise_kernel(x_emb, x_emb)
    Kyy = pairwise_kernel(y_emb, y_emb)
    Kxy = pairwise_kernel(x_emb, y_emb)

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


def target_grad_mmd_dataset_distillation(
        x, y, kernel, rng, get_list_augmentation=lambda rng: [lambda z: z],
        x_weights=None, y_weights=None, n_sample_batch=None, c=1, w_img=28,
        h_img=28):
    """
        Use autodifferentiation.

        Parameters
        ----------
        x: array of size (n_distr, n_samples_by_distr, d)
        y: array of size (m_distr, m_samples_by_distr, d)
        kernel: function taking x,y as input
        rng: key
        get_list_augmentation: function taking as input a PRNGKey, and returns a list of augmentations
        x_weights: array of size (n_distr,)
        y_weights: array of size (m_distr,)
        n_sample_batch: number of particles to sample from y, default None
    """
    n, n_sample, _ = x.shape
    m, m_sample, _ = y.shape

    if n_sample_batch is None:
        n_sample_batch = m_sample

    master_key, key_subsampling = jax.random.split(rng)
    keys = jax.random.split(key_subsampling, num=m+1)

    y_tgt = jax.vmap(
        lambda z, key: jax.random.choice(key, z, (n_sample_batch,),
                                         replace=False)
        )(y, keys[1:m+1])

    def func_mmd(z):
        return mmd_dataset_distillation(master_key, z, y_tgt, kernel,
                                        get_list_augmentation,
                                        x_weights, y_weights, c, w_img, h_img)

    out, grad = jax.value_and_grad(func_mmd)(x)
    return out, n * n_sample * grad  # multiply also by number of inner sample


def target_value_and_grad_riesz_dataset_distillation(x, y, rng, n_batch_real,
                                                     r=1, n_projs=500, c=1,
                                                     w_img=28, h_img=28):
    master_key, key = jax.random.split(rng, num=2)

    def kernel(u, v):
        return riesz_kernel_sw(u, v, key, r=r, n_projs=n_projs)

    l, grad = target_grad_mmd_dataset_distillation(
        x, y, kernel, master_key, n_sample_batch=n_batch_real, c=c,
        w_img=w_img, h_img=h_img)

    return l, grad


def target_value_and_grad_riesz_dataset_distillation_aug_mnist(
        x, y, rng, n_batch_real, r=1, n_projs=500, c=1, w_img=28, h_img=28):

    master_key, key = jax.random.split(rng, num=2)

    def kernel(u, v):
        return riesz_kernel_sw(u, v, key, r=r, n_projs=n_projs)

    def get_list_augmentation(rng):
        list_augmentations = [lambda z: rand_color(rng, z),
                              lambda z: rand_crop(rng, z),
                              lambda z: rand_cutout(rng, z),
                              lambda z: rand_scale(rng, z),
                              lambda z: rand_rotate(rng, z)]
        return list_augmentations

    l, grad = target_grad_mmd_dataset_distillation(x, y, kernel, master_key,
                                                   get_list_augmentation,
                                                   n_sample_batch=n_batch_real,
                                                   c=c, w_img=w_img,
                                                   h_img=h_img)
    return l, grad


def target_value_and_grad_riesz_dataset_distillation_aug_full(
        x, y, rng, n_batch_real, r=1, n_projs=500, c=1, w_img=28, h_img=28):

    master_key, key = jax.random.split(rng, num=2)

    def kernel(u, v):
        return riesz_kernel_sw(u, v, key, r=r, n_projs=n_projs)

    def get_list_augmentation(rng):
        list_augmentations = [lambda z: rand_color(rng, z),
                              lambda z: rand_crop(rng, z),
                              lambda z: rand_cutout(rng, z),
                              lambda z: rand_scale(rng, z),
                              lambda z: rand_rotate(rng, z),
                              lambda z: rand_flip(rng, z)]
        return list_augmentations

    l, grad = target_grad_mmd_dataset_distillation(x, y, kernel, master_key,
                                                   get_list_augmentation,
                                                   n_sample_batch=n_batch_real,
                                                   c=c, w_img=w_img,
                                                   h_img=h_img)
    return l, grad

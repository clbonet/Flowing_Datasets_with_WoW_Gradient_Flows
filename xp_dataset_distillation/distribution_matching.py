import sys
import jax
import jax.numpy as jnp

from utils_augmentation import rand_flip, rand_rotate, rand_scale, \
    rand_crop, rand_cutout, rand_color

sys.path.append("../lib")
from lib.classif_nn import ConvNet


def target_value_and_grad_dm_ambient(
        x, X_tgt, rng, n_batch_real):
    """
        Distribution Matching for one class (without any augmentation)

        Parameters
        ----------
        x: ndarray of size (n, d)
        X_tgt: ndarray of size (m, d)
        rng:
        n_batch_real: int, choose n_batch_real subsample from X_tgt
    """
    master_key, key_inds = jax.random.split(rng)
    inds = jax.random.randint(key_inds, (n_batch_real,), 0, len(X_tgt))
    X_tgt_subsample = X_tgt[inds]

    X_tgt_emb = jnp.mean(X_tgt_subsample, axis=0)

    @jax.jit
    def loss(z):
        x_emb = jnp.mean(z, axis=0)
        return jnp.sum(jnp.square(x_emb - X_tgt_emb))

    return jax.value_and_grad(loss)(x)


def target_value_and_grad_dm_full_class_ambient(
        x, X_tgt, rng, n_batch_real):
    """
        Do not use embedding with random neural network or augmentation

        Parameters
        ----------
        x: ndarray of size (c, n, d)
        X_tgt: ndarray of size (c, m, d)
        rng
        n_batch_real: int, choose n_batch_real subsample from X_tgt
    """
    target_value_and_grad = jax.vmap(
        lambda x, y, rng: target_value_and_grad_dm_ambient(
            x, y, rng, n_batch_real))

    keys = jax.random.split(rng, len(x))

    losses, grads = target_value_and_grad(x, X_tgt, keys)
    return jnp.mean(losses), grads


def target_value_and_grad_dm_emb(
        x, X_tgt, rng, n_batch_real, get_list_augmentation,
        c=1, w_img=28, h_img=28):
    """
        Distribution matching using an embedding (random NN) and
        possible an augmentation

        Parameters
        ----------
        x: ndarray of size (n, d)
        X_tgt: ndarray of size (m, d)
        rng: PRNGKey
        n_batch_real: int, choose n_batch_real subsample from X_tgt
        get_list_augmentation: function taking as input a PRNGKey, and returns a list of augmentations
    """
    _, key_inds, key_nn, key_choice_aug, key_aug = jax.random.split(rng, 5)
    inds = jax.random.randint(key_inds, (n_batch_real,), 0, len(X_tgt))
    X_tgt_subsample = X_tgt[inds]

    # Sample an embedding
    model = ConvNet(key_nn,  channel=c, im_size=(w_img, h_img))

    # Sample an augmentation
    list_augmentations = get_list_augmentation(key_aug)
    ind_aug = jax.random.randint(key_choice_aug, (), 0,
                                 len(list_augmentations))

    def apply_aug(z):
        return jax.lax.switch(ind_aug, list_augmentations, z)

    # Wrapper embedding + augmentation
    def wrap_embedding(x):
        x = jnp.reshape(x, (-1, w_img, h_img))
        x_aug = apply_aug(x)
        x_emb = model.embed(x_aug)
        return jnp.ravel(x_emb)

    wrap_embedding_vmap = jax.jit(jax.vmap(wrap_embedding))

    # Apply embedding to target
    X_tgt_emb = jnp.mean(wrap_embedding_vmap(X_tgt_subsample), axis=0)

    @jax.jit
    def loss(z):
        x_emb = jnp.mean(wrap_embedding_vmap(z), axis=0)
        return jnp.sum(jnp.square(x_emb - X_tgt_emb))

    return jax.value_and_grad(loss)(x)


def target_value_and_grad_dm_full_class_emb_mnist(
        x, X_tgt, rng, n_batch_real, c=1, w_img=28, h_img=28):
    """
        Distribution Matching with embedding and augmentation suitable
        for MNIST

        Parameters
        ----------
        x: ndarray of size (c, n, d)
        X_tgt: ndarray of size (c, m, d)
        rng
        n_batch_real: int, choose n_batch_real subsample from X_tgt
    """

    def get_list_augmentation(rng):
        list_augmentations = [lambda z: rand_color(rng, z),
                              lambda z: rand_crop(rng, z),
                              lambda z: rand_cutout(rng, z),
                              lambda z: rand_scale(rng, z),
                              lambda z: rand_rotate(rng, z)]
        return list_augmentations

    target_value_and_grad = jax.vmap(
        lambda x, y, rng: target_value_and_grad_dm_emb(
            x, y, rng, n_batch_real, get_list_augmentation, c, w_img, h_img))

    keys = jax.random.split(rng, len(x))

    losses, grads = target_value_and_grad(x, X_tgt, keys)
    return jnp.mean(losses), grads


def target_value_and_grad_dm_full_class_emb(
        x, X_tgt, rng, n_batch_real, c=1, w_img=28, h_img=28):
    """
        Distribution Matching with embedding and augmentation suitable
        for any other datasets

        Parameters
        ----------
        x: ndarray of size (c, n, d)
        X_tgt: ndarray of size (c, m, d)
        rng
        n_batch_real: int, choose n_batch_real subsample from X_tgt
    """

    def get_list_augmentation(rng):
        list_augmentations = [lambda z: rand_color(rng, z),
                              lambda z: rand_crop(rng, z),
                              lambda z: rand_cutout(rng, z),
                              lambda z: rand_scale(rng, z),
                              lambda z: rand_rotate(rng, z),
                              lambda z: rand_flip(rng, z)]
        return list_augmentations

    target_value_and_grad = jax.vmap(
        lambda x, y, rng: target_value_and_grad_dm_emb(
            x, y, rng, n_batch_real, get_list_augmentation, c, w_img, h_img))

    keys = jax.random.split(rng, len(x))

    losses, grads = target_value_and_grad(x, X_tgt, keys)
    return jnp.mean(losses), grads


def target_value_and_grad_dm_full_class_only_emb(
        x, X_tgt, rng, n_batch_real, c=1, w_img=28, h_img=28):
    """
        Distribution Matching with embedding but no augmentation

        Parameters
        ----------
        x: ndarray of size (c, n, d)
        X_tgt: ndarray of size (c, m, d)
        rng
        n_batch_real: int, choose n_batch_real subsample from X_tgt
    """

    def get_list_augmentation(rng):
        list_augmentations = [lambda z: z]
        return list_augmentations

    target_value_and_grad = jax.vmap(
        lambda x, y, rng: target_value_and_grad_dm_emb(
            x, y, rng, n_batch_real, get_list_augmentation, c, w_img, h_img))

    keys = jax.random.split(rng, len(x))

    losses, grads = target_value_and_grad(x, X_tgt, keys)
    return jnp.mean(losses), grads


def target_value_and_grad_dm_only_aug(
        x, X_tgt, rng, n_batch_real, get_list_augmentation,
        c=1, w_img=28, h_img=28):
    """
        Distribution Matching with augmentation but no embedding

        Parameters
        ----------
        x: ndarray of size (n, d)
        X_tgt: ndarray of size (m, d)
        rng: PRNGKey
        n_batch_real: int, choose n_batch_real subsample from X_tgt
        get_list_augmentation: function taking as input a PRNGKey, and returns a list of augmentations
    """
    _, key_inds, key_choice_aug, key_aug = jax.random.split(rng, 4)
    inds = jax.random.randint(key_inds, (n_batch_real,), 0, len(X_tgt))
    X_tgt_subsample = X_tgt[inds]

    # Sample an augmentation
    list_augmentations = get_list_augmentation(key_aug)
    ind_aug = jax.random.randint(key_choice_aug, (), 0,
                                 len(list_augmentations))

    def apply_aug(z):
        return jax.lax.switch(ind_aug, list_augmentations, z)

    # Wrapper embedding + augmentation
    def wrap_embedding(x):
        x = jnp.reshape(x, (-1, w_img, h_img))
        x_aug = apply_aug(x)
        return jnp.ravel(x_aug)

    wrap_embedding_vmap = jax.jit(jax.vmap(wrap_embedding))

    # Apply embedding to target
    X_tgt_emb = jnp.mean(wrap_embedding_vmap(X_tgt_subsample), axis=0)

    @jax.jit
    def loss(z):
        x_emb = jnp.mean(wrap_embedding_vmap(z), axis=0)
        return jnp.sum(jnp.square(x_emb - X_tgt_emb))

    return jax.value_and_grad(loss)(x)


def target_value_and_grad_dm_full_class_only_aug_mnist(
        x, X_tgt, rng, n_batch_real, c=1, w_img=28, h_img=28):
    """
        Distribution Matching with augmentation (for MNIST) but no embedding

        Parameters
        ----------
        x: ndarray of size (c, n, d)
        X_tgt: ndarray of size (c, m, d)
        rng
        n_batch_real: int, choose n_batch_real subsample from X_tgt
    """

    def get_list_augmentation(rng):
        list_augmentations = [lambda z: rand_color(rng, z),
                              lambda z: rand_crop(rng, z),
                              lambda z: rand_cutout(rng, z),
                              lambda z: rand_scale(rng, z),
                              lambda z: rand_rotate(rng, z)]
        return list_augmentations

    target_value_and_grad = jax.vmap(
        lambda x, y, rng: target_value_and_grad_dm_only_aug(
            x, y, rng, n_batch_real, get_list_augmentation, c, w_img, h_img))

    keys = jax.random.split(rng, len(x))

    losses, grads = target_value_and_grad(x, X_tgt, keys)
    return jnp.mean(losses), grads


def target_value_and_grad_dm_full_class_only_aug(
        x, X_tgt, rng, n_batch_real, c=1, w_img=28, h_img=28):
    """
        Distribution Matching with augmentation (for MNIST) but no embedding

        Parameters
        ----------
        x: ndarray of size (c, n, d)
        X_tgt: ndarray of size (c, m, d)
        rng
        n_batch_real: int, choose n_batch_real subsample from X_tgt
    """

    def get_list_augmentation(rng):
        list_augmentations = [lambda z: rand_color(rng, z),
                              lambda z: rand_crop(rng, z),
                              lambda z: rand_cutout(rng, z),
                              lambda z: rand_scale(rng, z),
                              lambda z: rand_rotate(rng, z),
                              lambda z: rand_flip(rng, z)]
        return list_augmentations

    target_value_and_grad = jax.vmap(
        lambda x, y, rng: target_value_and_grad_dm_only_aug(
            x, y, rng, n_batch_real, get_list_augmentation, c, w_img, h_img))
    keys = jax.random.split(rng, len(x))

    losses, grads = target_value_and_grad(x, X_tgt, keys)
    return jnp.mean(losses), grads

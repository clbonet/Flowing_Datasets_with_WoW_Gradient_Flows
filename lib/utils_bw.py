import jax.numpy as jnp


def exp_bw(Sigma, S):
    """
        Exponential map Bures-Wasserstein space at Sigma: exp_Sigma(S)
    """
    d = S.shape[-1]
    Id = jnp.eye(d)
    C = Id + S

    return jnp.einsum("bnm, bmk, bkl -> bnl", C, Sigma, C)


def sqrtm(a):
    """
        From POT https://github.com/PythonOT/POT/blob/master/ot/backend.py
    """

    L, V = jnp.linalg.eigh(a)
    L = jnp.sqrt(L)
    # Q[...] = V[...] @ diag(L[...])
    Q = jnp.einsum('...jk,...k->...jk', V, L)
    # R[...] = Q[...] @ V[...].T
    return jnp.einsum('...jk,...kl->...jl', Q, jnp.swapaxes(V, -1, -2))


def trace(C):
    """
        batchable trace
    """
    return C.diagonal(offset=0, axis1=-1, axis2=-2).sum(-1)


def bures_wasserstein_batch(m0, m1, s0, s1):
    """
        Parameters
        ----------
        - m0: shape (n, d)
        - m1: shape (m, d)
        - s0: shape (n, d, d)
        - s1: shape (m, d, d)

        Output
        ------
        - BW distance: shape (n, m)
    """
    n, d = m0.shape
    m = m1.shape[0]

    dist_m = jnp.linalg.norm(m0[:, None]-m1[None], axis=-1)**2

    s12 = sqrtm(s0)
    C12 = sqrtm(jnp.einsum("nij, mjk, nkl -> nmil", s12, s1, s12).reshape(-1, d, d)).reshape(n, m, d, d)
    dist_b = trace(s0)[:, None] + trace(s1)[None] - 2 * trace(C12)

    output = jnp.sqrt(dist_m + dist_b)
    return jnp.nan_to_num(output, 0)


# def bures_wasserstein_batch(m0, m1, s0, s1):
#     """
#         Parameters
#         ----------
#         - m0: shape (n, d)
#         - m1: shape (m, d)
#         - s0: shape (n, d, d)
#         - s1: shape (m, d, d)

#         Output
#         ------
#         - BW distance: shape (n, m)
#     """
#     n, d = m0.shape
#     m = m1.shape[0]

#     dist_m = jnp.linalg.norm(m0[:, None]-m1[None], axis=-1)**2

#     batch_n_id = jnp.stack([jnp.eye(d) for _ in range(n)])
#     batch_nm_id = jnp.stack([jnp.eye(d) for _ in range(n*m)])

#     s12 = sqrtm(s0 + 1e-5 * batch_n_id)
#     C12 = sqrtm(jnp.einsum("nij, mjk, nkl -> nmil", s12, s1, s12).reshape(-1, d, d) + 1e-5 * batch_nm_id).reshape(n, m, d, d)
#     dist_b = trace(s0)[:, None] + trace(s1)[None] - 2 * trace(C12)

#     output = jnp.sqrt(dist_m + dist_b)
#     return jnp.nan_to_num(output, 0)


def bures_wasserstein(m0, m1, s0, s1):
    """
        Parameters
        ----------
        - m0: shape (d,)
        - m1: shape (d,)
        - s0: shape (d, d)
        - s1: shape (d, d)

        Output
        ------
        - BW distance
    """
    d = len(m0)

    dist_m = jnp.linalg.norm(m0-m1, axis=-1)**2

    s12 = sqrtm(s0)
    C12 = sqrtm(jnp.einsum("ij, jk, kl -> il", s12, s1, s12))
    dist_b = trace(s0) + trace(s1) - 2 * trace(C12)

    output = jnp.sqrt(dist_m + dist_b)
    return jnp.nan_to_num(output, 0)

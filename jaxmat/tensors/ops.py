from __future__ import annotations

import jax
import jax.numpy as jnp

from .spaces import fourth_order, second_order, symmetric_fourth_order, symmetric_second_order
from .tensor import Tensor


def asarray(x) -> jax.Array:
    if isinstance(x, Tensor):
        return x.as_full()
    return jnp.asarray(x)


def trace(A) -> jax.Array:
    return jnp.trace(asarray(A), axis1=-2, axis2=-1)


def sym(A):
    if isinstance(A, Tensor):
        return A.sym
    arr = asarray(A)
    return 0.5 * (arr + jnp.swapaxes(arr, -1, -2))


def skew(A):
    if isinstance(A, Tensor):
        return A.skew
    arr = asarray(A)
    return 0.5 * (arr - jnp.swapaxes(arr, -1, -2))


def spherical(A):
    arr = asarray(A)
    dim = arr.shape[-1]
    mean = trace(arr) / dim
    eye = jnp.eye(dim, dtype=arr.dtype)
    return mean[..., None, None] * eye


def dev(A):
    arr = asarray(A)
    full = arr - spherical(arr)
    if isinstance(A, Tensor) and A.rank == 2:
        return A._wrap_full(symmetric_second_order(A.dim), full)
    return full


def inner(A, B) -> jax.Array:
    return jnp.sum(asarray(A) * asarray(B), axis=(-2, -1))


def norm(A) -> jax.Array:
    return jnp.linalg.norm(asarray(A), axis=(-2, -1))


def outer(A, B):
    arr = jnp.einsum("...ij,...kl->...ijkl", asarray(A), asarray(B))
    if isinstance(A, Tensor) and isinstance(B, Tensor):
        if A.space == symmetric_second_order(A.dim) and B.space == symmetric_second_order(B.dim):
            return Tensor.from_full(symmetric_fourth_order(A.dim), arr, project=True)
    return Tensor.from_full(fourth_order(asarray(A).shape[-1]), arr)


def contract(A, B, axes=2):
    return jnp.tensordot(asarray(A), asarray(B), axes=axes)


def apply(A, B):
    if isinstance(A, Tensor):
        return A @ B
    arr = jnp.einsum("...ijkl,...kl->...ij", asarray(A), asarray(B))
    return arr


def identity2(dim: int, dtype=jnp.float64) -> Tensor:
    return Tensor.from_full(second_order(dim), jnp.eye(dim, dtype=dtype))


def identity4s(dim: int, dtype=jnp.float64) -> Tensor:
    n = dim * (dim + 1) // 2
    return Tensor.from_compact(symmetric_fourth_order(dim), jnp.eye(n, dtype=dtype))


def spherical_projector(dim: int, dtype=jnp.float64) -> Tensor:
    Identity = identity2(dim, dtype=dtype).as_full()
    J = jnp.einsum("ij,kl->ijkl", Identity, Identity) / dim
    return Tensor.from_full(symmetric_fourth_order(dim), J, project=True)


def deviatoric_projector(dim: int, dtype=jnp.float64) -> Tensor:
    return identity4s(dim, dtype=dtype) - spherical_projector(dim, dtype=dtype)


def isotropic_stiffness(kappa: float, mu: float, dim: int = 3, dtype=jnp.float64) -> Tensor:
    return spherical_projector(dim, dtype=dtype) * (3.0 * kappa) + deviatoric_projector(
        dim, dtype=dtype
    ) * (2.0 * mu)

from __future__ import annotations

import jax.numpy as jnp

from .ops import asarray
from .spaces import symmetric_second_order
from .tensor import Tensor


def spectral_decomposition_sym(A):
    arr = asarray(A)
    eigvals, eigvecs = jnp.linalg.eigh(arr)
    projectors = jnp.einsum("...ia,...ja->...aij", eigvecs, eigvecs)
    return eigvals, projectors


def eigenvalues(A):
    return jnp.linalg.eigvalsh(asarray(A))


def eigenprojectors(A):
    return spectral_decomposition_sym(A)[1]


def eig33(A, rtol=1e-16):
    eigvals, projectors = spectral_decomposition_sym(A)
    return eigvals, projectors


def matrix_function_sym(A, fun):
    eigvals, eigvecs = jnp.linalg.eigh(asarray(A))
    transformed = fun(eigvals)
    return jnp.einsum("...ia,...a,...ja->...ij", eigvecs, transformed, eigvecs)


def sqrtm(A):
    full = matrix_function_sym(A, jnp.sqrt)
    if isinstance(A, Tensor):
        return A._wrap_full(symmetric_second_order(A.dim), full)
    return full


def inv_sqrtm(A):
    full = matrix_function_sym(A, lambda x: 1.0 / jnp.sqrt(x))
    if isinstance(A, Tensor):
        return A._wrap_full(symmetric_second_order(A.dim), full)
    return full


def expm(A):
    full = matrix_function_sym(A, jnp.exp)
    if isinstance(A, Tensor):
        return A._wrap_full(symmetric_second_order(A.dim), full)
    return full


def logm(A):
    full = matrix_function_sym(A, jnp.log)
    if isinstance(A, Tensor):
        return A._wrap_full(symmetric_second_order(A.dim), full)
    return full


def powm(A, m):
    full = matrix_function_sym(A, lambda x: jnp.power(x, m))
    if isinstance(A, Tensor):
        return A._wrap_full(symmetric_second_order(A.dim), full)
    return full

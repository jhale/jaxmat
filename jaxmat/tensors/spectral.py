from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from .ops import asarray
from .spaces import symmetric_second_order
from .tensor import Tensor
from .utils import safe_norm, safe_sqrt


@partial(jax.jit, static_argnums=1)
def _eig22_single(A, rtol=1e-16):
    def compute_eigvals(A):
        A = jnp.asarray(A)
        trA = jnp.trace(A)
        mean = trA / 2.0
        dev = A - mean * jnp.eye(2, dtype=A.dtype)
        radius = safe_sqrt(0.5 * jnp.sum(dev * dev))
        normA = safe_norm(A)

        def branch_near_iso(_):
            eigvals = jnp.ones((2,), dtype=A.dtype) * mean
            return eigvals, eigvals

        def branch_general(_):
            eigvals = jnp.asarray([mean - radius, mean + radius], dtype=A.dtype)
            return eigvals, eigvals

        return lax.cond(radius < rtol * normA, branch_near_iso, branch_general, operand=None)

    eigendyads, eigvals = jax.jacfwd(compute_eigvals, has_aux=True)(A)
    return eigvals, eigendyads


@partial(jax.jit, static_argnums=1)
def _eig33_single(A, rtol=1e-16):
    def dim(A):
        return A.shape[0]

    def tr(A):
        return jnp.trace(A)

    def dev(A):
        d = dim(A)
        Id = jnp.eye(d, dtype=A.dtype)
        return A - tr(A) / d * Id

    def compute_eigvals_harari_albocher(A):
        A = jnp.asarray(A)
        norm = safe_norm(A)
        Id = jnp.eye(dim(A), dtype=A.dtype)
        I1 = jnp.trace(A)
        S = dev(A)
        J2 = tr(S.T @ S) / 2
        s = safe_sqrt(J2 / 3)

        def branch_near_iso(_):
            eigvals = jnp.ones((3,), dtype=A.dtype) * I1 / 3
            return eigvals, eigvals

        def branch_general(_):
            T = S @ S - 2 * J2 / 3 * Id
            d = safe_norm(T - s * S) / safe_norm(T + s * S)
            sj = jnp.sign(1 - d)
            cond = sj * (1 - d) < rtol * norm

            def branch_two_eigvals(_):
                lamb_max = jnp.sqrt(3.0) * s
                eigvals_dev = jnp.asarray([lamb_max, 0.0, -lamb_max], dtype=A.dtype)
                eigvals = eigvals_dev + I1 / 3
                return eigvals, eigvals

            def branch_three_eigvals(_):
                alpha = 2 / 3 * jnp.arctan(d**sj)
                lambda_d = 2 * sj * s * jnp.cos(alpha)
                sd = jnp.sqrt(3.0) * s * jnp.sin(alpha)

                eigvals_dev = lax.cond(
                    lambda_d > 0,
                    lambda _: jnp.asarray(
                        [-lambda_d / 2 - sd, -lambda_d / 2 + sd, lambda_d], dtype=A.dtype
                    ),
                    lambda _: jnp.asarray(
                        [lambda_d, -lambda_d / 2 - sd, -lambda_d / 2 + sd], dtype=A.dtype
                    ),
                    operand=None,
                )
                eigvals = eigvals_dev + I1 / 3
                return eigvals, eigvals

            return lax.cond(cond, branch_two_eigvals, branch_three_eigvals, operand=None)

        return lax.cond(s < rtol * norm, branch_near_iso, branch_general, operand=None)

    eigendyads, eigvals = jax.jacfwd(compute_eigvals_harari_albocher, has_aux=True)(A)
    return eigvals, eigendyads


def _batched_spectral(single_fun, arr, *args):
    dim = arr.shape[-1]
    if arr.ndim == 2:
        return single_fun(arr, *args)
    batch_shape = arr.shape[:-2]
    flat = arr.reshape((-1, dim, dim))
    eigvals, projectors = jax.vmap(lambda x: single_fun(x, *args))(flat)
    eigvals = eigvals.reshape((*batch_shape, dim))
    projectors = projectors.reshape((*batch_shape, dim, dim, dim))
    return eigvals, projectors


def spectral_decomposition_sym(A):
    arr = asarray(A)
    dim = arr.shape[-1]
    if dim == 2:
        return _batched_spectral(_eig22_single, arr)
    if dim == 3:
        return _batched_spectral(_eig33_single, arr)
    eigvals, eigvecs = jnp.linalg.eigh(arr)
    projectors = jnp.einsum("...ia,...ja->...aij", eigvecs, eigvecs)
    return eigvals, projectors


def eigenvalues(A):
    return spectral_decomposition_sym(A)[0]


def eigenprojectors(A):
    return spectral_decomposition_sym(A)[1]


def eig33(A, rtol=1e-16):
    return _batched_spectral(_eig33_single, asarray(A), rtol)


def matrix_function_sym(A, fun):
    eigvals, projectors = spectral_decomposition_sym(A)
    transformed = fun(eigvals)
    return jnp.einsum("...a,...aij->...ij", transformed, projectors)


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

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


@partial(jax.jit, static_argnums=1)
def _eig33_single_invariants(A, rtol=1e-16):
    def J2s(A):
        d0 = A[0, 0] - A[1, 1]
        d1 = A[0, 0] - A[2, 2]
        d2 = A[1, 1] - A[2, 2]
        offdiag = A[0, 1] ** 2 + A[0, 2] ** 2 + A[1, 2] ** 2
        diag = (d0**2 + d1**2 + d2**2) / 6.0
        return offdiag + diag

    def J3s(A):
        d0 = A[0, 0] - A[1, 1]
        d1 = A[0, 0] - A[2, 2]
        d2 = A[1, 1] - A[2, 2]
        t1 = d1 + d2
        t2 = d0 - d2
        t3 = -d0 - d1
        offdiag = 2.0 * A[0, 1] * A[1, 2] * A[0, 2]
        mixed = (A[0, 1] ** 2 * t1 + A[0, 2] ** 2 * t2 + A[1, 2] ** 2 * t3) / 3.0
        diag = (t1 * t2 * t3) / 27.0
        return offdiag + mixed - diag

    def dxs(A):
        d0 = A[0, 0] - A[1, 1]
        d1 = A[0, 0] - A[2, 2]
        d2 = A[1, 1] - A[2, 2]

        w = A[0, 1]
        v = A[0, 2]
        u = A[1, 2]

        alpha = d2
        beta = -d1
        gamma = d0

        return jnp.asarray(
            [
                3.0 * jnp.sqrt(3.0) * (v * w * alpha + u * (v * v - w * w)),
                alpha * beta * gamma + alpha * u * u + beta * v * v + gamma * w * w,
                2.0 * u * beta * gamma - v * w * (beta - gamma) + u * (2.0 * u * u - v * v - w * w),
                2.0
                * (v * alpha * gamma + u * w * (beta - gamma) + v * (v * v + w * w - 2.0 * u * u)),
                2.0
                * (w * alpha * beta + u * v * (beta - gamma) + w * (v * v + w * w - 2.0 * u * u)),
            ],
            dtype=A.dtype,
        )

    def discs(A):
        terms = dxs(A)
        return jnp.sum(terms * terms)

    def compute_eigvals(A):
        A = jnp.asarray(A)
        I1 = jnp.trace(A)
        j2 = J2s(A)
        j3 = J3s(A)
        discriminant = discs(A)
        normA = safe_norm(A)

        def branch_near_iso(_):
            eigvals = jnp.ones((3,), dtype=A.dtype) * I1 / 3.0
            return eigvals, eigvals

        def branch_general(_):
            phi = jnp.arctan2(safe_sqrt(27.0 * discriminant), 27.0 * j3)
            amplitude = 2.0 * safe_sqrt(3.0 * j2)
            shifts = 2.0 * jnp.pi * jnp.asarray([1.0, 2.0, 3.0], dtype=A.dtype)
            eigvals = (amplitude * jnp.cos((phi + shifts) / 3.0) + I1) / 3.0
            return eigvals, eigvals

        return lax.cond(j2 < rtol * normA, branch_near_iso, branch_general, operand=None)

    eigendyads, eigvals = jax.jacfwd(compute_eigvals, has_aux=True)(A)
    order = jnp.argsort(eigvals)
    eigvals = eigvals[order]
    eigendyads = eigendyads[order]
    eigendyads = 0.5 * (eigendyads + jnp.swapaxes(eigendyads, -1, -2))
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


def eig33(A, rtol=1e-16, method: str = "harari_albocher"):
    arr = asarray(A)
    if method == "harari_albocher":
        return _batched_spectral(_eig33_single, arr, rtol)
    if method == "habera_zilian":
        return _batched_spectral(_eig33_single_invariants, arr, rtol)
    raise ValueError("method must be 'harari_albocher' or 'habera_zilian'")


def eig33_invariants(A, rtol=1e-16):
    return eig33(A, rtol=rtol, method="habera_zilian")


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

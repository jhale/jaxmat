from __future__ import annotations

import jax
import jax.numpy as jnp

from .ops import asarray, dev, trace


def det(A) -> jax.Array:
    return jnp.linalg.det(asarray(A))


def principal_invariants(A):
    arr = asarray(A)
    i1 = trace(arr)
    i2 = 0.5 * (i1**2 - trace(arr @ arr))
    i3 = jnp.linalg.det(arr)
    return i1, i2, i3


def main_invariants(A):
    arr = asarray(A)
    j1 = trace(arr)
    j2 = trace(arr @ arr)
    j3 = trace(arr @ arr @ arr)
    return j1, j2, j3


def J2(A):
    s = asarray(dev(A))
    return 0.5 * jnp.sum(s * s, axis=(-2, -1))


def J3(A):
    s = asarray(dev(A))
    return jnp.linalg.det(s)


def pq_invariants(sig):
    arr = asarray(sig)
    p = -trace(arr) / arr.shape[-1]
    q = jnp.sqrt(3.0 * J2(arr))
    return p, q


def von_mises(sig):
    return jnp.sqrt(3.0 * J2(sig))


def lode_angle(sig, eps: float = 1e-16):
    j2 = J2(sig)
    j3 = J3(sig)
    xi = (3.0 * jnp.sqrt(3.0) / 2.0) * j3 / jnp.maximum(j2, eps) ** 1.5
    return jnp.arcsin(jnp.clip(xi, -1.0, 1.0)) / 3.0

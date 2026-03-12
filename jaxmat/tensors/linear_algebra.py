from __future__ import annotations

import jax.numpy as jnp

from .invariants import main_invariants, pq_invariants, principal_invariants
from .ops import dev, trace
from .spectral import eig33, expm, inv_sqrtm, logm, matrix_function_sym, powm, sqrtm


def dim(A):
    return jnp.asarray(A).shape[-1]


def tr(A):
    return trace(A)


def det33(A):
    return jnp.linalg.det(jnp.asarray(A))


def inv33(A):
    return jnp.linalg.inv(jnp.asarray(A))


def isotropic_function(fun, A):
    return matrix_function_sym(A, fun)

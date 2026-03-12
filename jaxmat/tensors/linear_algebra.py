from __future__ import annotations

import jax.numpy as jnp

from .ops import trace
from .spectral import matrix_function_sym


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

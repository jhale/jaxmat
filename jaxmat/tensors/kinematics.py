from __future__ import annotations

import jax
import jax.numpy as jnp

from .ops import identity2
from .spaces import second_order, symmetric_second_order
from .spectral import inv_sqrtm, sqrtm
from .tensor import Tensor


def _as_tensor(F) -> Tensor:
    if isinstance(F, Tensor):
        return F
    arr = jnp.asarray(F)
    return Tensor.from_full(second_order(arr.shape[-1]), arr)


def jacobian(F) -> jax.Array:
    return jnp.linalg.det(_as_tensor(F).as_full())


def inverse(F):
    tensor = _as_tensor(F)
    return tensor._wrap_full(second_order(tensor.dim), jnp.linalg.inv(tensor.as_full()))


def cofactor(F):
    tensor = _as_tensor(F)
    full = jacobian(tensor)[..., None, None] * jnp.swapaxes(
        jnp.linalg.inv(tensor.as_full()), -1, -2
    )
    return tensor._wrap_full(second_order(tensor.dim), full)


def right_cauchy_green(F):
    tensor = _as_tensor(F)
    full = jnp.swapaxes(tensor.as_full(), -1, -2) @ tensor.as_full()
    return tensor._wrap_full(symmetric_second_order(tensor.dim), full)


def left_cauchy_green(F):
    tensor = _as_tensor(F)
    full = tensor.as_full() @ jnp.swapaxes(tensor.as_full(), -1, -2)
    return tensor._wrap_full(symmetric_second_order(tensor.dim), full)


def green_lagrange(F):
    tensor = _as_tensor(F)
    C = right_cauchy_green(tensor).as_full()
    Identity = identity2(tensor.dim, dtype=C.dtype).as_full()
    return tensor._wrap_full(symmetric_second_order(tensor.dim), 0.5 * (C - Identity))


def almansi(F):
    tensor = _as_tensor(F)
    b = left_cauchy_green(tensor).as_full()
    Identity = identity2(tensor.dim, dtype=b.dtype).as_full()
    return tensor._wrap_full(
        symmetric_second_order(tensor.dim), 0.5 * (Identity - jnp.linalg.inv(b))
    )


def polar(F, mode: str = "RU"):
    tensor = _as_tensor(F)
    C = right_cauchy_green(tensor)
    U = sqrtm(C)
    U_inv = inv_sqrtm(C)
    R = tensor._wrap_full(second_order(tensor.dim), tensor.as_full() @ U_inv.as_full())
    if mode == "RU":
        return R, U
    if mode == "VR":
        V = tensor._wrap_full(
            symmetric_second_order(tensor.dim),
            R.as_full() @ U.as_full() @ jnp.swapaxes(R.as_full(), -1, -2),
        )
        return V, R
    raise ValueError("mode must be 'RU' or 'VR'")


def stretch(F, kind: str = "right"):
    if kind == "right":
        return polar(F, mode="RU")[1]
    if kind == "left":
        return polar(F, mode="VR")[0]
    raise ValueError("kind must be 'right' or 'left'")

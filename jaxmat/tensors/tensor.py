from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from .conversions import full_to_mandel2, full_to_mandel4, mandel2_to_full, mandel4_to_full
from .spaces import (
    TensorSpace,
    fourth_order,
    second_order,
    symmetric_fourth_order,
    symmetric_second_order,
    tensor_space,
)


def _space_from_other(other: Any, fallback_rank: int, fallback_dim: int) -> TensorSpace:
    if isinstance(other, Tensor):
        return other.space
    return tensor_space(fallback_dim, fallback_rank)


def _as_full(x: Any) -> jax.Array:
    if isinstance(x, Tensor):
        return x.as_full()
    return jnp.asarray(x)


class Tensor(eqx.Module):
    data: jax.Array
    space: TensorSpace = eqx.field(static=True)
    is_compact: bool = eqx.field(static=True, default=False)

    __array_priority__ = 1000

    @classmethod
    def from_full(
        cls,
        space: TensorSpace,
        x: jax.Array,
        *,
        validate: bool = False,
        project: bool = False,
    ) -> Tensor:
        full = jnp.asarray(x)
        if full.shape[-space.rank :] != space.full_shape:
            raise ValueError(
                f"Wrong shape {full.shape}; expected trailing shape {space.full_shape}"
            )
        if project:
            full = space.project_full(full)
        if validate and not space.validate_full(full):
            raise ValueError(f"Input data does not satisfy symmetry {space.symmetry.name}")
        return cls(data=full, space=space, is_compact=False)

    @classmethod
    def from_compact(cls, space: TensorSpace, a: jax.Array) -> Tensor:
        compact = jnp.asarray(a)
        if compact.shape[-len(space.compact_shape) :] != space.compact_shape:
            raise ValueError(
                f"Wrong compact shape {compact.shape}; "
                "expected trailing shape {space.compact_shape}"
            )
        return cls(data=compact, space=space, is_compact=True)

    @property
    def dim(self) -> int:
        return self.space.dim

    @property
    def rank(self) -> int:
        return self.space.rank

    @property
    def tensor(self) -> jax.Array:
        return self.as_full()

    @property
    def compact(self) -> jax.Array:
        return self.as_compact()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.space.full_shape

    def as_full(self) -> jax.Array:
        if not self.is_compact:
            return self.data
        if self.space.symmetry.compact_kind == "mandel2":
            return mandel2_to_full(self.data, self.dim)
        if self.space.symmetry.compact_kind == "mandel4":
            return mandel4_to_full(self.data, self.dim)
        raise ValueError(f"Unsupported compact kind {self.space.symmetry.compact_kind}")

    def as_compact(self) -> jax.Array:
        if self.is_compact:
            return self.data
        full = self.data
        if self.space.symmetry.compact_kind == "full":
            return full
        if self.space.symmetry.compact_kind == "mandel2":
            return full_to_mandel2(full)
        if self.space.symmetry.compact_kind == "mandel4":
            return full_to_mandel4(full)
        raise ValueError(f"Unsupported compact kind {self.space.symmetry.compact_kind}")

    def project(self) -> Tensor:
        return self._wrap_full(self.space, self.space.project_full(self.as_full()))

    def validate(self, atol: float = 1e-8) -> bool:
        return self.space.validate_full(self.as_full(), atol=atol)

    def __getitem__(self, idx):
        return self.as_full()[idx]

    def __jax_array__(self):
        return self.as_full()

    def __array__(self, dtype=None):
        return jnp.asarray(self.as_full(), dtype=dtype)

    def _wrap_full(self, space: TensorSpace, full: jax.Array) -> Tensor:
        return Tensor.from_full(space, full, project=False)

    def _binary_result_space(self, other: Any) -> TensorSpace:
        if isinstance(other, Tensor) and other.space == self.space:
            return self.space
        return tensor_space(self.dim, self.rank)

    def __add__(self, other):
        other_full = _as_full(other)
        return self._wrap_full(self._binary_result_space(other), self.as_full() + other_full)

    def __sub__(self, other):
        other_full = _as_full(other)
        return self._wrap_full(self._binary_result_space(other), self.as_full() - other_full)

    def __mul__(self, other):
        return self._wrap_full(self.space, self.as_full() * jnp.asarray(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._wrap_full(self.space, self.as_full() / jnp.asarray(other))

    def __neg__(self):
        return self._wrap_full(self.space, -self.as_full())

    def __matmul__(self, other):
        lhs = self.as_full()
        rhs = _as_full(other)
        other_rank = other.rank if isinstance(other, Tensor) else rhs.ndim
        if self.rank == 2 and rhs.shape[-2:] == (self.dim, self.dim):
            return self._wrap_full(second_order(self.dim), lhs @ rhs)
        if self.rank == 4 and isinstance(other, Tensor) and other.rank == 4:
            if self.space == symmetric_fourth_order(
                self.dim
            ) and other.space == symmetric_fourth_order(self.dim):
                result = self.as_compact() @ other.as_compact()
                return self._wrap_full(
                    symmetric_fourth_order(self.dim), mandel4_to_full(result, self.dim)
                )
            result = jnp.einsum("...ijmn,...mnkl->...ijkl", lhs, rhs)
            return self._wrap_full(fourth_order(self.dim), result)
        if self.rank == 4 and (other_rank == 2 or rhs.shape[-2:] == (self.dim, self.dim)):
            result = jnp.einsum("...ijkl,...kl->...ij", lhs, rhs)
            space = second_order(self.dim)
            if self.space == symmetric_fourth_order(self.dim):
                if isinstance(other, Tensor) and other.space == symmetric_second_order(self.dim):
                    space = symmetric_second_order(self.dim)
            return self._wrap_full(space, result)
        raise TypeError("Unsupported matmul operands")

    def __rmatmul__(self, other):
        other_full = _as_full(other)
        if other_full.shape[-2:] == (self.dim, self.dim) and self.rank == 2:
            return self._wrap_full(second_order(self.dim), other_full @ self.as_full())
        raise TypeError("Unsupported matmul operands")

    @property
    def T(self) -> Tensor:
        if self.rank != 2:
            raise AttributeError("Transpose is only defined for rank-2 tensors")
        full = jnp.swapaxes(self.as_full(), -1, -2)
        return self._wrap_full(self.space, full)

    @property
    def sym(self) -> Tensor:
        if self.rank != 2:
            raise AttributeError("sym is only defined for rank-2 tensors")
        full = 0.5 * (self.as_full() + jnp.swapaxes(self.as_full(), -1, -2))
        return self._wrap_full(symmetric_second_order(self.dim), full)

    @property
    def skew(self) -> Tensor:
        if self.rank != 2:
            raise AttributeError("skew is only defined for rank-2 tensors")
        full = 0.5 * (self.as_full() - jnp.swapaxes(self.as_full(), -1, -2))
        return self._wrap_full(second_order(self.dim), full)

    @property
    def inv(self) -> Tensor:
        if self.rank != 2:
            raise AttributeError("Inverse is only defined for rank-2 tensors")
        space = second_order(self.dim)
        if self.space == symmetric_second_order(self.dim):
            space = symmetric_second_order(self.dim)
        return self._wrap_full(space, jnp.linalg.inv(self.as_full()))

    def rotate(self, R: jax.Array) -> Tensor:
        assert self.rank <= 13
        pairs = [(chr(97 + 2 * i), chr(97 + 2 * i + 1)) for i in range(self.rank)]
        rotation_pairs = [first + second for first, second in pairs]
        output_indices = "".join(first for first, _ in pairs)
        tensor_indices = "".join(second for _, second in pairs)
        einsum_str = ",".join(rotation_pairs) + "," + tensor_indices + "->" + output_indices
        full = jnp.einsum(einsum_str, *([R] * self.rank), self.as_full())
        return self._wrap_full(self.space, full)

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

CompactKind = Literal["full", "mandel2", "mandel4"]
Permutation = tuple[int, ...]


def _permute_trailing_axes(x: jax.Array, perm: Permutation) -> jax.Array:
    rank = len(perm)
    prefix = tuple(range(x.ndim - rank))
    suffix = tuple(x.ndim - rank + p for p in perm)
    return jnp.transpose(x, prefix + suffix)


@dataclass(frozen=True)
class SymmetryClass:
    name: str
    rank: int
    generators: tuple[Permutation, ...]
    signs: tuple[int, ...]
    compact_kind: CompactKind = "full"

    def __post_init__(self):
        if len(self.generators) != len(self.signs):
            raise ValueError("generators and signs must have the same length")
        for perm in self.generators:
            if tuple(sorted(perm)) != tuple(range(self.rank)):
                raise ValueError(f"Invalid generator {perm} for rank {self.rank}")

    def project_full(self, x: jax.Array) -> jax.Array:
        projected = x
        for perm, sign in zip(self.generators, self.signs, strict=True):
            projected = 0.5 * (projected + sign * _permute_trailing_axes(projected, perm))
        return projected

    def validate_full(self, x: jax.Array, atol: float = 1e-8) -> bool:
        return all(
            jnp.allclose(x, sign * _permute_trailing_axes(x, perm), atol=atol)
            for perm, sign in zip(self.generators, self.signs, strict=True)
        )

    def n_independent(self, dim: int) -> int:
        if self.compact_kind == "full":
            return dim**self.rank
        if self.compact_kind == "mandel2":
            return dim * (dim + 1) // 2
        if self.compact_kind == "mandel4":
            n = dim * (dim + 1) // 2
            return n * (n + 1) // 2
        raise ValueError(f"Unsupported compact kind {self.compact_kind}")


@dataclass(frozen=True)
class TensorSpace:
    dim: int
    rank: int
    symmetry: SymmetryClass

    def __post_init__(self):
        if self.dim < 2:
            raise ValueError("dim must be >= 2")
        if self.rank != self.symmetry.rank:
            raise ValueError("space rank must match symmetry rank")

    @property
    def full_shape(self) -> tuple[int, ...]:
        return (self.dim,) * self.rank

    @property
    def compact_shape(self) -> tuple[int, ...]:
        if self.symmetry.compact_kind == "full":
            return self.full_shape
        if self.symmetry.compact_kind == "mandel2":
            return (self.symmetry.n_independent(self.dim),)
        if self.symmetry.compact_kind == "mandel4":
            n = self.dim * (self.dim + 1) // 2
            return (n, n)
        raise ValueError(f"Unsupported compact kind {self.symmetry.compact_kind}")

    def project_full(self, x: jax.Array) -> jax.Array:
        return self.symmetry.project_full(x)

    def validate_full(self, x: jax.Array, atol: float = 1e-8) -> bool:
        return self.symmetry.validate_full(x, atol=atol)


NONE_2 = SymmetryClass(
    name="none_2",
    rank=2,
    generators=(),
    signs=(),
    compact_kind="full",
)

SYMMETRIC_2 = SymmetryClass(
    name="symmetric_2",
    rank=2,
    generators=((1, 0),),
    signs=(+1,),
    compact_kind="mandel2",
)

SKEW_2 = SymmetryClass(
    name="skew_2",
    rank=2,
    generators=((1, 0),),
    signs=(-1,),
    compact_kind="full",
)

NONE_4 = SymmetryClass(
    name="none_4",
    rank=4,
    generators=(),
    signs=(),
    compact_kind="full",
)

MINOR_SYMMETRIC_4 = SymmetryClass(
    name="minor_symmetric_4",
    rank=4,
    generators=((1, 0, 2, 3), (0, 1, 3, 2)),
    signs=(+1, +1),
    compact_kind="full",
)

MAJOR_MINOR_SYMMETRIC_4 = SymmetryClass(
    name="major_minor_symmetric_4",
    rank=4,
    generators=((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)),
    signs=(+1, +1, +1),
    compact_kind="mandel4",
)


def tensor_space(dim: int, rank: int) -> TensorSpace:
    if rank == 2:
        return second_order(dim)
    if rank == 4:
        return fourth_order(dim)
    return TensorSpace(dim=dim, rank=rank, symmetry=SymmetryClass(f"none_{rank}", rank, (), ()))


def second_order(dim: int) -> TensorSpace:
    return TensorSpace(dim=dim, rank=2, symmetry=NONE_2)


def symmetric_second_order(dim: int) -> TensorSpace:
    return TensorSpace(dim=dim, rank=2, symmetry=SYMMETRIC_2)


def skew_second_order(dim: int) -> TensorSpace:
    return TensorSpace(dim=dim, rank=2, symmetry=SKEW_2)


def fourth_order(dim: int) -> TensorSpace:
    return TensorSpace(dim=dim, rank=4, symmetry=NONE_4)


def minor_symmetric_fourth_order(dim: int) -> TensorSpace:
    return TensorSpace(dim=dim, rank=4, symmetry=MINOR_SYMMETRIC_4)


def symmetric_fourth_order(dim: int) -> TensorSpace:
    return TensorSpace(dim=dim, rank=4, symmetry=MAJOR_MINOR_SYMMETRIC_4)

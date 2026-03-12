from __future__ import annotations

import jax
import jax.numpy as jnp

from .conversions import (
    full_to_legacy_tensor2_array,
    full_to_mandel2,
    full_to_mandel4,
    legacy_tensor2_array_to_full,
    mandel2_to_full,
    mandel4_to_full,
)
from .ops import deviatoric_projector, identity2, isotropic_stiffness, spherical_projector
from .spaces import second_order, symmetric_fourth_order, symmetric_second_order
from .spectral import eig33
from .tensor import Tensor


class Tensor2(Tensor):
    def __init__(
        self, tensor: jax.Array | None = None, array: jax.Array | None = None, dim: int = 3
    ):
        space = second_order(dim)
        if tensor is not None:
            obj = Tensor.from_full(space, tensor)
        elif array is not None:
            if array.shape[-1] != dim**2:
                raise ValueError(f"Wrong shape {array.shape}; expected trailing shape {(dim**2,)}")
            obj = Tensor.from_full(space, legacy_tensor2_array_to_full(jnp.asarray(array), dim))
        else:
            obj = Tensor.from_full(space, jnp.zeros(space.full_shape))
        super().__init__(data=obj.data, space=obj.space, is_compact=obj.is_compact)

    @classmethod
    def identity(cls, dim: int = 3):
        return cls(tensor=identity2(dim).as_full(), dim=dim)

    @property
    def array(self):
        return full_to_legacy_tensor2_array(self.as_full())

    @property
    def array_shape(self):
        return (self.dim**2,)

    def _wrap_full(self, space, full):
        if space == second_order(3):
            return Tensor2(tensor=full)
        if space == symmetric_second_order(3):
            return SymmetricTensor2(tensor=full)
        return Tensor.from_full(space, full)

    @property
    def eigenvalues(self):
        eigvals, eigendyads = eig33(self.as_full())
        return eigvals, jnp.asarray([SymmetricTensor2(tensor=N) for N in eigendyads])


class SymmetricTensor2(Tensor):
    def __init__(
        self,
        tensor: jax.Array | None = None,
        array: jax.Array | None = None,
        dim: int = 3,
        validate: bool = False,
    ):
        space = symmetric_second_order(dim)
        if tensor is not None:
            obj = Tensor.from_full(space, tensor, validate=validate)
        elif array is not None:
            if array.shape[-1] != dim * (dim + 1) // 2:
                raise ValueError(
                    f"Wrong shape {array.shape}; expected trailing shape {(dim * (dim + 1) // 2,)}"
                )
            obj = Tensor.from_full(space, mandel2_to_full(jnp.asarray(array), dim))
        else:
            obj = Tensor.from_full(space, jnp.zeros(space.full_shape))
        super().__init__(data=obj.data, space=obj.space, is_compact=obj.is_compact)

    @classmethod
    def identity(cls, dim: int = 3):
        return cls(tensor=identity2(dim).as_full(), dim=dim)

    @property
    def array(self):
        return full_to_mandel2(self.as_full())

    @property
    def array_shape(self):
        return (self.dim * (self.dim + 1) // 2,)

    def is_symmetric(self):
        return self.validate()

    def _wrap_full(self, space, full):
        if space == symmetric_second_order(3):
            return SymmetricTensor2(tensor=full)
        if space == second_order(3):
            return Tensor2(tensor=full)
        if space == symmetric_fourth_order(3):
            return SymmetricTensor4(tensor=full)
        return Tensor.from_full(space, full)


class SymmetricTensor4(Tensor):
    def __init__(
        self, tensor: jax.Array | None = None, array: jax.Array | None = None, dim: int = 3
    ):
        space = symmetric_fourth_order(dim)
        if tensor is not None:
            obj = Tensor.from_full(space, tensor, project=True)
        elif array is not None:
            n = dim * (dim + 1) // 2
            if array.shape[-2:] != (n, n):
                raise ValueError(f"Wrong shape {array.shape}; expected trailing shape {(n, n)}")
            obj = Tensor.from_full(space, mandel4_to_full(jnp.asarray(array), dim))
        else:
            n = dim * (dim + 1) // 2
            obj = Tensor.from_compact(space, jnp.zeros((n, n)))
        super().__init__(data=obj.data, space=obj.space, is_compact=obj.is_compact)

    @classmethod
    def identity(cls, dim: int = 3):
        return cls(array=jnp.eye(dim * (dim + 1) // 2), dim=dim)

    @classmethod
    def J(cls, dim: int = 3):
        return cls(tensor=spherical_projector(dim).as_full(), dim=dim)

    @classmethod
    def K(cls, dim: int = 3):
        return cls(tensor=deviatoric_projector(dim).as_full(), dim=dim)

    @property
    def array(self):
        return full_to_mandel4(self.as_full())

    @property
    def array_shape(self):
        n = self.dim * (self.dim + 1) // 2
        return (n, n)

    def is_symmetric(self):
        return self.validate()

    @property
    def inv(self):
        return SymmetricTensor4(array=jnp.linalg.inv(self.array), dim=self.dim)

    def _wrap_full(self, space, full):
        if space == symmetric_fourth_order(3):
            return SymmetricTensor4(tensor=full)
        if space == symmetric_second_order(3):
            return SymmetricTensor2(tensor=full)
        if space == second_order(3):
            return Tensor2(tensor=full)
        return Tensor.from_full(space, full)


class IsotropicTensor4(SymmetricTensor4):
    kappa: float
    mu: float

    def __init__(self, kappa, mu, dim: int = 3):
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "mu", mu)
        super().__init__(tensor=isotropic_stiffness(kappa, mu, dim=dim).as_full(), dim=dim)

    @property
    def coeffs(self):
        return jnp.asarray([3.0 * self.kappa, 2.0 * self.mu])

    @property
    def inv(self):
        return IsotropicTensor4(1.0 / (9.0 * self.kappa), 1.0 / (4.0 * self.mu), dim=self.dim)

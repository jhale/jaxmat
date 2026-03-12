import jax
import jax.numpy as jnp

import jaxmat.materials as jm
from jaxmat.tensors import Tensor, second_order


def test_compressible_ogden():
    F = jnp.eye(3)
    material = jm.CompressibleOgden(mu=4.0, alpha=jnp.array([2.0, -2.0]), kappa=1e3)

    lamb = jnp.linspace(1, 2.5, 10)
    N = len(lamb)

    F = jnp.broadcast_to(F, (N, 3, 3))
    F = F.at[:, 0, 0].set(lamb)
    F = F.at[:, 1, 1].set(lamb)
    F = F.at[:, 2, 2].set(1 / lamb**2)

    F = Tensor.from_full(second_order(3), F)

    jax.vmap(material.PK1)(F)
    jax.vmap(material.Cauchy)(F)

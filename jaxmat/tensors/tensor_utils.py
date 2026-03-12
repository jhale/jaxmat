from __future__ import annotations

import jax
import jax.numpy as jnp

from .ops import skew
from .spectral import eigenvalues as spectral_eigenvalues


def axl(A):
    As = skew(A)
    arr = As.as_full() if hasattr(As, "as_full") else jnp.asarray(As)
    return jnp.stack([-arr[..., 1, 2], arr[..., 0, 2], -arr[..., 0, 1]], axis=-1)


def dev_vect(A):
    arr = jnp.asarray(A)
    dim = 2 if arr.shape[-1] == 3 else 3
    from .ops import deviatoric_projector

    return deviatoric_projector(dim).as_compact() @ arr


@jax.custom_jvp
def eigenvalues_jvp_ready(sig):
    return spectral_eigenvalues(sig)


@eigenvalues_jvp_ready.defjvp
def _eigenvalues_jvp(primals, tangents):
    (sig,) = primals
    (dsig,) = tangents
    vals = spectral_eigenvalues(sig)
    deig = jax.jvp(spectral_eigenvalues, (sig,), (dsig,))[1]
    return vals, deig


eigenvalues = eigenvalues_jvp_ready

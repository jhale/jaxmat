from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def mandel_pairs(dim: int) -> list[tuple[tuple[int, int], float]]:
    pairs = [((i, i), 1.0) for i in range(dim)]
    sqrt2 = math.sqrt(2.0)
    for i in range(dim):
        for j in range(i + 1, dim):
            pairs.append(((i, j), sqrt2))
    return pairs


def full_to_mandel2(x: jax.Array) -> jax.Array:
    dim = x.shape[-1]
    comps = [x[..., i, i] for i in range(dim)]
    sqrt2 = jnp.sqrt(2.0)
    for i in range(dim):
        for j in range(i + 1, dim):
            comps.append(sqrt2 * x[..., i, j])
    return jnp.stack(comps, axis=-1)


def mandel2_to_full(a: jax.Array, dim: int) -> jax.Array:
    out = jnp.zeros((*a.shape[:-1], dim, dim), dtype=a.dtype)
    for i in range(dim):
        out = out.at[..., i, i].set(a[..., i])
    offset = dim
    sqrt2 = jnp.sqrt(2.0)
    for i in range(dim):
        for j in range(i + 1, dim):
            value = a[..., offset] / sqrt2
            out = out.at[..., i, j].set(value)
            out = out.at[..., j, i].set(value)
            offset += 1
    return out


def full_to_mandel4(x: jax.Array) -> jax.Array:
    dim = x.shape[-1]
    pairs = mandel_pairs(dim)
    rows = []
    for ij, ni in pairs:
        row = []
        for kl, nk in pairs:
            row.append(ni * nk * x[..., ij[0], ij[1], kl[0], kl[1]])
        rows.append(jnp.stack(row, axis=-1))
    return jnp.stack(rows, axis=-2)


def mandel4_to_full(a: jax.Array, dim: int) -> jax.Array:
    out = jnp.zeros((*a.shape[:-2], dim, dim, dim, dim), dtype=a.dtype)
    pairs = mandel_pairs(dim)
    for alpha, (ij, ni) in enumerate(pairs):
        ij_swapped = tuple(reversed(ij))
        ij_perms = {ij, ij_swapped}
        for beta, (kl, nk) in enumerate(pairs):
            kl_swapped = tuple(reversed(kl))
            kl_perms = {kl, kl_swapped}
            value = a[..., alpha, beta] / (ni * nk)
            for ii, jj in ij_perms:
                for kk, ll in kl_perms:
                    out = out.at[..., ii, jj, kk, ll].set(value)
                    out = out.at[..., kk, ll, ii, jj].set(value)
    return out


def full_to_legacy_tensor2_array(x: jax.Array) -> jax.Array:
    dim = x.shape[-1]
    comps = [x[..., i, i] for i in range(dim)]
    for i in range(dim):
        for j in range(i + 1, dim):
            comps.append(x[..., i, j])
            comps.append(x[..., j, i])
    return jnp.stack(comps, axis=-1)


def legacy_tensor2_array_to_full(a: jax.Array, dim: int) -> jax.Array:
    out = jnp.zeros((*a.shape[:-1], dim, dim), dtype=a.dtype)
    for i in range(dim):
        out = out.at[..., i, i].set(a[..., i])
    offset = dim
    for i in range(dim):
        for j in range(i + 1, dim):
            out = out.at[..., i, j].set(a[..., offset])
            out = out.at[..., j, i].set(a[..., offset + 1])
            offset += 2
    return out

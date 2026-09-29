"""Self-intersection test for triangle meshes (used by cloth_bench.py).

intersecting_pairs(V, F) returns the pairs of non-adjacent triangles (no shared vertex) whose
interiors cross, via bounding-box culling on a uniform grid and exact segment-triangle tests.
"""
from __future__ import annotations

import numpy as np


def _seg_tri(p0, p1, a, b, c, eps=1e-12):
    """Vectorised: does segment p0-p1 cross triangle abc (strictly, Moller-Trumbore)?"""
    d = p1 - p0
    e1, e2 = b - a, c - a
    h = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(det) > eps
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = p0 - a
    u = inv * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = inv * np.einsum("ij,ij->i", d, q)
    t = inv * np.einsum("ij,ij->i", e2, q)
    return ok & (u > 0) & (v > 0) & (u + v < 1) & (t > 0) & (t < 1)


def candidate_pairs(V, F, cell: float | None = None):
    """Non-adjacent triangle pairs with overlapping bounding boxes."""
    T = V[F]                                   # (n, 3, 3)
    lo, hi = T.min(axis=1), T.max(axis=1)
    if cell is None:
        cell = float(np.median(np.linalg.norm(T[:, 1] - T[:, 0], axis=1))) * 1.5
    # bin every triangle into all grid cells its box touches, then pair within cells
    c0 = np.floor(lo / cell).astype(np.int64)
    c1 = np.floor(hi / cell).astype(np.int64)
    ids, keys = [], []
    span = c1 - c0 + 1
    for dx in range(span[:, 0].max()):
        for dy in range(span[:, 1].max()):
            for dz in range(span[:, 2].max()):
                sel = (dx < span[:, 0]) & (dy < span[:, 1]) & (dz < span[:, 2])
                c = c0[sel] + [dx, dy, dz]
                ids.append(np.where(sel)[0])
                keys.append((c[:, 0] * 73856093) ^ (c[:, 1] * 19349663) ^ (c[:, 2] * 83492791))
    ids, keys = np.concatenate(ids), np.concatenate(keys)
    order = np.argsort(keys, kind="stable")
    ids, keys = ids[order], keys[order]
    starts = np.r_[0, np.where(np.diff(keys))[0] + 1, len(keys)]
    pairs = []
    for s, e in zip(starts[:-1], starts[1:]):
        if e - s > 1:
            g = ids[s:e]
            i, j = np.triu_indices(len(g), 1)
            pairs.append(np.stack([g[i], g[j]], axis=1))
    if not pairs:
        return np.zeros((0, 2), int)
    P = np.unique(np.sort(np.concatenate(pairs), axis=1), axis=0)
    P = P[P[:, 0] != P[:, 1]]
    a, b = P[:, 0], P[:, 1]
    overlap = np.all((lo[a] <= hi[b]) & (lo[b] <= hi[a]), axis=1)
    shared = (F[a][:, :, None] == F[b][:, None, :]).any(axis=(1, 2))
    return P[overlap & ~shared]


def intersecting_pairs(V, F):
    P = candidate_pairs(V, F)
    if len(P) == 0:
        return P
    hit = np.zeros(len(P), bool)
    for tri, other in ((0, 1), (1, 0)):
        A, B = F[P[:, tri]], F[P[:, other]]
        for e0, e1 in ((0, 1), (1, 2), (2, 0)):
            hit |= _seg_tri(V[A[:, e0]], V[A[:, e1]], V[B[:, 0]], V[B[:, 1]], V[B[:, 2]])
    return P[hit]

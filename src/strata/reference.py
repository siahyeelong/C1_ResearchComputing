"""Supplied NumPy reference implementations (float64 by default).

Shares no code with the JAX kernels. Used by the tests (float64) and as the
host-CPU GRM baseline in Question 3(a) (float32, the same formula, one BLAS call).
"""
from __future__ import annotations

import numpy as np


def standardise(X, dtype=np.float64):
    """Peak host memory: X plus one n x p `dtype` buffer when there are no
    missing calls (the simulated cohorts); a mask is kept only if a 255 occurs."""
    X = np.asarray(X)
    miss = X == 255
    if miss.any():
        Xf = np.where(miss, 0, X).astype(dtype)
        n_obs = X.shape[0] - miss.sum(0)
    else:
        del miss
        miss = None
        Xf = X.astype(dtype)
        n_obs = X.shape[0]
    f = Xf.sum(0, dtype=np.float64) / (2.0 * n_obs)
    Xf -= (2.0 * f).astype(dtype)                    # in place: one n x p buffer
    Xf /= np.sqrt(2.0 * f * (1.0 - f)).astype(dtype)
    if miss is not None:
        Xf[miss] = 0.0
    return Xf


def grm(Xt):
    """G = Xt Xt^T / p. NumPy sends Xt @ Xt.T to BLAS syrk, which exploits
    symmetry and performs about n^2 p FLOP, half the counted 2 n^2 p. The
    division is in place, so only one n x n buffer is allocated."""
    G = Xt @ Xt.T
    G /= Xt.shape[1]
    return G


def eigh_pcs(G, k):
    w, U = np.linalg.eigh(G)
    order = np.argsort(w)[::-1][:k]
    return U[:, order], w[order]


def principal_angles(Q1, Q2):
    """Degrees, ascending. arccos of sigma(Q1^T Q2) where cos^2 < 1/2, arcsin
    of sigma(Q2 - Q1 Q1^T Q2) elsewhere (small angles), in float64. Inputs are
    re-orthonormalised by QR, so float32 inputs do not set an arccos floor."""
    Q1, _ = np.linalg.qr(np.asarray(Q1, np.float64))
    Q2, _ = np.linalg.qr(np.asarray(Q2, np.float64))
    M = Q1.T @ Q2
    c = np.linalg.svd(M, compute_uv=False)
    s = np.linalg.svd(Q2 - Q1 @ M, compute_uv=False)[::-1][:c.size]
    th = np.where(c * c < 0.5, np.arccos(np.minimum(1.0, c)),
                  np.arcsin(np.minimum(1.0, s)))
    return np.degrees(th)


def normalised_traits(y, perm_idx):
    """perm_idx: (B, n) integer permutations (row 0 the identity)."""
    Y = np.asarray(y, np.float64)[np.asarray(perm_idx)].T
    Y = Y - Y.mean(0)
    return Y / np.linalg.norm(Y, axis=0)


def scan(Xt, Yt):
    n = Xt.shape[0]
    S = Xt.T @ Yt
    s = (Xt * Xt).sum(0)
    return n * S ** 2 / s[:, None]

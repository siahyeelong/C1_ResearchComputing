"""Simulated cohorts with known ancestry (the generative model).

K ancestral populations with Balding-Nichols allele frequencies at divergence
F_ST, admixture proportions q_i ~ Dirichlet(alpha), and genotypes drawn
binomially from the individual frequency f_ij = q_i . F_j. Every marker has
sample MAF >= 0.05 and there are no missing calls, by construction.

The genotype matrix is generated on the default device in blocks of
SIM_CHUNK columns, so a 16384 x 131072 cohort never needs the n x p float32
intermediates at once (that alone would be 8.6 GB per array). The block size is
fixed, so the cohort is a deterministic function of (seed, n, p, K) and the
first n rows / first p columns of a larger cohort with the same seed and the
same K are NOT guaranteed identical to a smaller cohort: always slice the
larger cohort, as the brief does in Question 3.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

SIM_CHUNK = 4096          # columns per generation block (fixed: part of the model)
FST = 0.1                 # Balding-Nichols divergence
ALPHA = 0.5               # symmetric Dirichlet concentration of q_i
MAF_MIN = 0.05            # guaranteed minimum sample MAF
N_CAUSAL = 10
H2 = 0.3                  # variance explained by the causal markers
STRUCT_VAR = 0.2          # variance of the ancestry-dependent trait term


@functools.partial(jax.jit, static_argnums=(2,))
def _block(key, q, c):
    """One block of c markers: frequencies -> genotypes (uint8)."""
    n, K = q.shape
    ka, kb, k1, k2 = jax.random.split(key, 4)
    anc = jax.random.uniform(ka, (c,), jnp.float32, 0.1, 0.9)
    a = anc * (1 - FST) / FST
    b = (1 - anc) * (1 - FST) / FST
    F = jax.random.beta(kb, a[None, :], b[None, :], (K, c), jnp.float32)
    F = jnp.clip(F, 0.05, 0.95)
    f = q @ F                                            # (n, c), float32
    g = (jax.random.uniform(k1, (n, c)) < f).astype(jnp.uint8) + \
        (jax.random.uniform(k2, (n, c)) < f).astype(jnp.uint8)
    frq = g.astype(jnp.float32).sum(0) / (2 * n)
    ok = jnp.minimum(frq, 1 - frq) >= MAF_MIN
    return g, ok


def cohort(seed: int, n: int, p: int, K: int = 3) -> dict:
    """Return a dict of device arrays: geno (n,p) uint8, pheno (n,) float32,
    truth_q (n,K) float32, truth_causal_idx, truth_beta, and meta."""
    base = jax.random.key(seed)
    q = jax.random.dirichlet(jax.random.fold_in(base, 0),
                             jnp.full((K,), ALPHA, jnp.float32), (n,))
    q = q.astype(jnp.float32)
    blocks = []
    for ib, j0 in enumerate(range(0, p, SIM_CHUNK)):
        c = min(SIM_CHUNK, p - j0)
        kblk = jax.random.fold_in(jax.random.fold_in(base, 2), ib)
        g, ok = _block(kblk, q, c)
        attempt = 0
        while not bool(ok.all()):           # rare: redraw failing columns only
            attempt += 1
            if attempt > 50:
                raise RuntimeError("could not reach MAF >= 0.05; n too small")
            g2, ok2 = _block(jax.random.fold_in(kblk, attempt), q, c)
            g = jnp.where(ok[None, :], g, g2)
            ok = ok | ok2
        blocks.append(g)
    X = blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks, axis=1)
    del blocks

    kt = jax.random.fold_in(base, 1)
    kc, kbeta, ks, ke = jax.random.split(kt, 4)
    idx = jnp.sort(jax.random.choice(kc, p, (min(N_CAUSAL, p),), replace=False))
    Xc = X[:, idx].astype(jnp.float32)
    fc = Xc.mean(0) / 2
    Zc = (Xc - 2 * fc) / jnp.sqrt(2 * fc * (1 - fc))
    beta = jax.random.normal(kbeta, (idx.shape[0],), jnp.float32)
    gval = Zc @ beta
    scale_g = jnp.sqrt(H2 / jnp.var(gval))
    beta = beta * scale_g
    gval = gval * scale_g
    sdir = q @ jax.random.normal(ks, (K,), jnp.float32)
    sval = sdir * jnp.sqrt(STRUCT_VAR / jnp.maximum(jnp.var(sdir), 1e-12))
    e = jax.random.normal(ke, (n,), jnp.float32) * np.sqrt(1 - H2 - STRUCT_VAR)
    y = (gval + sval + e).astype(jnp.float32)
    meta = dict(seed=seed, n=n, p=p, K=K, fst=FST, alpha=ALPHA, maf_min=MAF_MIN,
                n_causal=int(idx.shape[0]), h2=H2, struct_var=STRUCT_VAR,
                sim_chunk=SIM_CHUNK)
    return dict(geno=X, pheno=y, truth_q=q, truth_causal_idx=idx.astype(jnp.int32),
                truth_beta=beta, meta=meta)

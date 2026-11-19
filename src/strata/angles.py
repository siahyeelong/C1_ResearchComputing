"""Principal angles between subspaces (supplied by the course)."""
import jax
import jax.numpy as jnp

HI = jax.lax.Precision.HIGHEST


@jax.jit
def principal_angles(Q1, Q2):
    """Principal angles (degrees, ascending) between span(Q1) and span(Q2),
    both n x k with orthonormal columns.

    cos theta_i = sigma_i(Q1^T Q2) (descending) and
    sin theta_i = sigma_{k+1-i}(Q2 - Q1 Q1^T Q2) (the residual of Q2 off
    span(Q1)). arccos is used where cos^2 < 1/2 and arcsin elsewhere
    (Knyazev and Argentati 2002). arccos alone has a float32 floor: inputs
    orthonormal to ~1e-6 give arccos(1 - 1e-6) ~ 0.08 deg however close the
    subspaces are; arcsin of the residual resolves angles down to ~1e-4 deg.
    """
    if Q1.shape != Q2.shape:
        raise ValueError(f"principal_angles needs equal shapes, got {Q1.shape}, {Q2.shape}")
    Q1 = Q1.astype(jnp.float32)
    Q2 = Q2.astype(jnp.float32)
    M = jnp.dot(Q1.T, Q2, precision=HI)
    c = jnp.linalg.svd(M, compute_uv=False)                              # descending
    s = jnp.linalg.svd(Q2 - jnp.dot(Q1, M, precision=HI), compute_uv=False)[::-1]
    theta = jnp.where(c * c < 0.5, jnp.arccos(jnp.minimum(1.0, c)),
                      jnp.arcsin(jnp.minimum(1.0, s)))
    return jnp.degrees(theta)

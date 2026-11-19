"""Timing helper (supplied by the course)."""
import statistics
import time

import jax


def timeit(f, *args, reps: int = 7):
    t = time.perf_counter()
    jax.block_until_ready(f(*args))
    t0 = time.perf_counter() - t
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(f(*args))
        ts.append(time.perf_counter() - t)
    med = statistics.median(ts)
    return dict(t0=t0, times=ts, t=med, t_compile=t0 - med)

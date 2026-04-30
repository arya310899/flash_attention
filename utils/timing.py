from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Tuple

import numpy as np


def benchmark_torch(
    fn: Callable[..., Any], *args: Any, warmup: int = 10, iters: int = 50
) -> Tuple[float, Any]:
    """Warm up then return (avg_seconds, last_output)."""
    import torch

    for _ in range(warmup):
        out = fn(*args)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        out = fn(*args)
    torch.cuda.synchronize()
    return (time.time() - t0) / iters, out


def peak_memory_mb_torch(fn: Callable[..., Any], *args: Any) -> Tuple[float, Any]:
    """Return (peak_MB, output) for a single torch CUDA call."""
    import torch

    torch.cuda.reset_peak_memory_stats()
    out = fn(*args)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2), out


def cuda_event_time_ms(
    fn: Callable[[], Any], warmup: int = 10, iters: int = 50
) -> float:
    """Average per-call CUDA event timing in milliseconds."""
    import torch

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_evt.record()
    for _ in range(iters):
        fn()
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) / iters


def time_jax(
    fn: Callable[..., Any], args: Iterable[Any], warmup: int = 3, iters: int = 10
) -> float:
    """Median per-call time in seconds using jax.block_until_ready."""
    import jax

    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def safe_time_jax(
    fn: Callable[..., Any],
    args: Iterable[Any],
    warmup: int = 3,
    iters: int = 10,
    verbose: bool = True,
) -> float:
    """Run JAX timer; return NaN on OOM/compile/runtime failure."""
    try:
        return time_jax(fn, args, warmup=warmup, iters=iters)
    except Exception as exc:  # noqa: BLE001 - notebook utility
        if verbose:
            msg = str(exc).splitlines()[0][:80]
            print(f"  [skip: {type(exc).__name__}: {msg}]", end="")
        return float("nan")

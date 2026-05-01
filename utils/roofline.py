from __future__ import annotations


DTYPE_BYTES = 2
LSE_BYTES = 4

# ── NVIDIA A100 (SXM, 80GB class) hardware ceilings ──
# Peak FP16/BF16 tensor-core throughput is the DENSE (no 2:4 structured sparsity)
# ceiling — the same regime as FlashAttention / standard matmul. NVIDIA also cites
# ~624 TFLOP/s with sparsity; that is intentionally NOT used here.
# Peak HBM bandwidth per NVIDIA spec sheet for this SKU class.
A100_PEAK_TFLOPS = 312.0
A100_PEAK_BW_GBS = 2039.0

# Default roofline targets for notebooks (alias for clarity at callsites).
PEAK_TFLOPS = A100_PEAK_TFLOPS
PEAK_BW_GBS = A100_PEAK_BW_GBS


def ridge_point(peak_tflops: float, peak_bw_gbs: float) -> float:
    """Return ridge point in FLOPs/byte from peak TFLOP/s and GB/s."""
    return (peak_tflops * 1e12) / (peak_bw_gbs * 1e9)


def throughput(elapsed_s: float, ops: float) -> float:
    """Return ops/sec, guarding invalid timings."""
    if elapsed_s != elapsed_s or elapsed_s == 0:
        return float("nan")
    return ops / elapsed_s


def attn_fwd_flops(T: int, C: int, causal: bool = True) -> float:
    """Algorithmic forward FLOPs per (B*H) attention instance."""
    flops = 4 * T * T * C
    return flops * (0.5 if causal else 1.0)


def attn_bwd_flops(T: int, C: int, causal: bool = True) -> float:
    """Algorithmic backward FLOPs per (B*H) attention instance."""
    flops = 10 * T * T * C
    return flops * (0.5 if causal else 1.0)


def naive_fwd_bytes(T: int, C: int, dtype_bytes: int = DTYPE_BYTES) -> int:
    """Naive attention HBM traffic in bytes for forward pass."""
    return (3 * T * C + 4 * T * T + T * C) * dtype_bytes


def naive_bwd_bytes(T: int, C: int, dtype_bytes: int = DTYPE_BYTES) -> int:
    """Naive attention HBM traffic in bytes for backward pass (approx)."""
    return (8 * T * C + 4 * T * T) * dtype_bytes


def flash_fwd_bytes(
    T: int, C: int, dtype_bytes: int = DTYPE_BYTES, lse_bytes: int = LSE_BYTES
) -> int:
    """FlashAttention HBM traffic in bytes for forward pass."""
    return 4 * T * C * dtype_bytes + T * lse_bytes


def flash_bwd_bytes(
    T: int, C: int, dtype_bytes: int = DTYPE_BYTES, lse_bytes: int = LSE_BYTES
) -> int:
    """FlashAttention HBM traffic in bytes for backward pass (idealized)."""
    return 8 * T * C * dtype_bytes + T * lse_bytes


def theoretical_flops(N: int, D: int) -> int:
    """Dominant forward FLOPs for one-head attention: ~= 4*N*N*D."""
    return 4 * N * N * D


def theoretical_bytes_naive(N: int, D: int, bytes_per_elem: int = DTYPE_BYTES) -> int:
    """Naive forward bytes with N^2 intermediates materialized in HBM."""
    return (4 * N * N + 4 * N * D) * bytes_per_elem


def theoretical_bytes_flash(
    N: int,
    D: int,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    bytes_per_elem: int = DTYPE_BYTES,
) -> int:
    """Flash forward bytes without N^2 HBM spill."""
    _ = BLOCK_M, BLOCK_N  # kept for API compatibility with notebook callsites
    return 4 * N * D * bytes_per_elem + N * LSE_BYTES

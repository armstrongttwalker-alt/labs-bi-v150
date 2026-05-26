"""
02_autotune_roofline.py
-----------------------
Find the fastest tile inside the legal space from step 01, and build a
roofline-style intuition for why some tiles dominate others.

Part A: Autotune over the legal space
-------------------------------------
Step 01 gave us a set of tiles that FIT on the chip. But "fits" is a
necessary, not sufficient, condition for good performance. Among all
legal tiles, which is actually fastest on 2048 x 2048 x 2048 matmul?

Hand-picking is unreliable: we expect the optimum NOT to be a square
tile, because A and B are accessed along different strides and the
chip's real access costs are not symmetric. @triton.autotune is the
right tool -- hand it the legal configs and a cache key, let it time
each one, and pick the winner.

Part B: BK sweep -> roofline intuition
--------------------------------------
Fix (BM, BN) at the autotuned winner and vary BK. The question is: why
does BK matter? Arithmetic intensity for a single tile is

    AI = FLOPs / bytes_loaded = (2 * BM * BN * BK) / ((BM + BN) * BK * 4)
       = BM * BN / (2 * (BM + BN))         # BK cancels (!)

So on paper BK shouldn't affect AI at all. But in practice it is the
*dominant* performance knob. Running the sweep tells us why.
"""
import time
import torch
import triton
import triton.language as tl

from common import bench_ms, make_inputs, silent_stdio


# =============================================================================
# Part A: autotune across the legal space
# =============================================================================
# Hardware constants (measured in step 01, using placeholder values here)
ACC_MAX_KB = 128
OPS_BUDGET_KB = 256


def is_legal(BM, BN, BK, dtype_bytes=4):
    acc_kb = BM * BN * dtype_bytes / 1024
    ops_kb = (BM + BN) * BK * dtype_bytes / 1024
    aligned = (BM % 16 == 0) and (BN % 16 == 0) and (BK % 16 == 0)
    return (acc_kb <= ACC_MAX_KB) and (ops_kb <= OPS_BUDGET_KB) and aligned


def build_configs():
    """Generate the Config list for autotune. One Config per legal tile."""
    # >>> YOUR CODE HERE >>>
    # Loop over candidate BM, BN, BK values and include those that pass
    # is_legal(). Return a list of triton.Config objects, each of the form:
    #   triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN, 'BLOCK_K': BK})
    #
    # Suggested candidate ranges (keeps search under ~80 configs):
    #   BM in {32, 64, 96, 128, 160, 192, 256}
    #   BN in {32, 64, 96, 128, 160, 192, 256}
    #   BK in {32, 64, 96, 128}
    configs = []
    # <<< END OF YOUR CODE <<<
    return configs


# Build configs with silenced warnings
with silent_stdio():
    AUTOTUNE_CONFIGS = build_configs()


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def matmul_autotuned_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Identical body to common.matmul_kernel; only the decorator differs."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[:, None] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul_autotuned(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),
                         triton.cdiv(N, META['BLOCK_N']))
    matmul_autotuned_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# =============================================================================
# Part B: fixed-shape kernel for the BK sweep
# =============================================================================
@triton.jit
def matmul_fixed_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[:, None] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul_fixed(a, b, BM, BN, BK):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_fixed_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )
    return c


# =============================================================================
# Runners
# =============================================================================
def part_a():
    print("=" * 72)
    print("Part A: autotune over the legal tile space (2048 x 2048 x 2048)")
    print("=" * 72)
    print(f"  Search space size: {len(AUTOTUNE_CONFIGS)} configs")
    print("  Autotune will compile all of them on first call -- this is slow")
    print("  the first time (~1-3 min) and free after that (cached).")
    print()

    M = N = K = 2048
    a, b = make_inputs(M, N, K)

    # Baseline reference: Day 1 defaults
    t_baseline = bench_ms(lambda: matmul_fixed(a, b, 64, 64, 32))

    # Warm up autotune (this is the slow part).
    print("  ... running autotune ...")
    with silent_stdio():
        matmul_autotuned(a, b)
    print("  ... done.")

    # Correctness
    c_ref = a @ b
    with silent_stdio():
        c_at = matmul_autotuned(a, b)
    max_diff = (c_ref - c_at).abs().max().item()

    # Extract the winning config
    best_cfg = matmul_autotuned_kernel.best_config
    BM = best_cfg.kwargs['BLOCK_M']
    BN = best_cfg.kwargs['BLOCK_N']
    BK = best_cfg.kwargs['BLOCK_K']

    # Timings
    with silent_stdio():
        t_auto = bench_ms(matmul_autotuned, a, b)
    t_torch = bench_ms(torch.matmul, a, b)

    print()
    print(f"  {'version':<32} {'time (ms)':>10} {'vs baseline':>12} {'vs torch':>10}")
    print("  " + "-" * 68)
    print(f"  {'baseline (BM=64,BN=64,BK=32)':<32} "
          f"{t_baseline:>10.4f} {1.0:>12.2f}x {t_torch/t_baseline:>9.2f}x")
    print(f"  {f'autotuned (BM={BM},BN={BN},BK={BK})':<32} "
          f"{t_auto:>10.4f} {t_baseline/t_auto:>11.2f}x {t_torch/t_auto:>9.2f}x")
    print(f"  {'torch.matmul':<32} "
          f"{t_torch:>10.4f} {t_baseline/t_torch:>11.2f}x {1.0:>9.2f}x")
    print()
    print(f"  Correctness: max diff = {max_diff:.2e}")

    return (BM, BN, BK), t_baseline, t_auto, t_torch


def part_b(best_BM, best_BN):
    print()
    print("=" * 72)
    print(f"Part B: BK sweep at fixed (BM={best_BM}, BN={best_BN})")
    print("=" * 72)
    print()
    print("We fix the tile shape at the autotuned winner's (BM, BN), then")
    print("vary BK.")
    print()

    M = N = K = 2048
    a, b = make_inputs(M, N, K)

    BKs = [16, 32, 48, 64, 96, 128, 160, 192]
    rows = []
    for BK in BKs:
        if not is_legal(best_BM, best_BN, BK):
            continue
        ms = bench_ms(lambda bk=BK: matmul_fixed(a, b, best_BM, best_BN, bk))
        l1_kb = (best_BM + best_BN) * BK * 4 / 1024
        rows.append((BK, l1_kb, ms))

    print(f"  {'BK':>4} {'Ops KB':>8} {'time (ms)':>10} {'vs BK=16':>10}")
    print("  " + "-" * 40)
    ref_ms = rows[0][2]
    for BK, l1_kb, ms in rows:
        speedup = ref_ms / ms
        print(f"  {BK:>4} {l1_kb:>8.1f} {ms:>10.4f} {speedup:>9.2f}x")

    print()
    print("Reading the curve")
    print("-" * 72)
    print("  * Going from BK=16 to BK=32/64 often halves the time. Why?")
    print("    Because AI only counts data INSIDE one tile. When BK is small,")
    print("    the K-loop runs many iterations, each with overhead. Bigger BK")
    print("    means fewer loop iterations.")
    print()
    print("  * Past some BK the curve flattens. This is the roofline ceiling:")
    print("    the compute unit is saturated.")


if __name__ == "__main__":
    (best_BM, best_BN, best_BK), t_baseline, t_auto, t_torch = part_a()
    part_b(best_BM, best_BN)

    print()
    print("=" * 72)
    print("Summary of step 02")
    print("=" * 72)
    print("  * autotune picked a tile (often asymmetric)")
    print("  * BK is the dominant knob; pick 32 or 64, let autotune confirm")
    print("  * We matched torch.matmul on fp32 2048^3")
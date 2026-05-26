"""
03_summary.py
-------------
Put every version of the matmul we wrote today on one table, at the
same shape, measured with the same benchmark, so the progress is
visible in one glance.

There is no new code here -- it is just a readout.
"""
import torch

from common import bench_ms, make_inputs, silent_stdio
# Re-use the two kernels from step 02.
from importlib import import_module
with silent_stdio():
    step2 = import_module("02_autotune_roofline")


def main():
    M = N = K = 2048
    a, b = make_inputs(M, N, K)

    print(f"Final comparison at M=N=K={M}, fp32")
    print("=" * 78)

    # 1. Baseline: Day 1's default (64, 64, 32)
    t_baseline = bench_ms(lambda: step2.matmul_fixed(a, b, 64, 64, 32))

    # 2. Hand-picked legal tile: just (128, 128, 64) -- a "reasonable guess"
    t_legal = bench_ms(lambda: step2.matmul_fixed(a, b, 128, 128, 64))

    # 3. Autotuned: step 02 picks the best legal tile
    with silent_stdio():
        step2.matmul_autotuned(a, b)  # trigger autotune warmup
    best_cfg = step2.matmul_autotuned_kernel.best_config
    bm, bn, bk = best_cfg.kwargs['BLOCK_M'], best_cfg.kwargs['BLOCK_N'], best_cfg.kwargs['BLOCK_K']
    with silent_stdio():
        t_auto = bench_ms(step2.matmul_autotuned, a, b)

    # 4. Vendor: torch.matmul
    t_torch = bench_ms(torch.matmul, a, b)

    print()
    print(f"  {'version':<45} {'ms':>10} {'vs baseline':>13} {'vs vendor':>11}")
    print("  " + "-" * 78)

    def row(name, t):
        print(f"  {name:<45} {t:>10.4f} {t_baseline/t:>12.2f}x {t_torch/t:>10.2f}x")

    row("baseline  (Day 1 default 64x64x32)", t_baseline)
    row("buffer-model tile  (128x128x64, hand)", t_legal)
    row(f"autotuned          ({bm}x{bn}x{bk})", t_auto)
    row("torch.matmul  (library optimized)", t_torch)

    # The gap that remains
    print()
    print("=" * 78)
    print("What we did NOT do (hooks for further exploration)")
    print("=" * 78)
    print()
    print("  1. MIXED PRECISION. We kept fp32 throughout. Many GPUs have")
    print("     fp16 / bf16 throughput SEVERAL TIMES their fp32 throughput.")
    print("     A fp16 (x fp16 -> fp32 accumulate) matmul routinely beats a")
    print("     fp32 matmul by 3-5x.")
    print()
    print("  2. GROUPED ORDERING (GROUP_SIZE_M). The Triton GPU tutorial")
    print("     uses a trick where nearby program ids compute a GROUP_SIZE_M")
    print("     x N block of the output, improving L2 cache reuse.")
    print()
    print("  3. OPERATOR FUSION. matmul-then-bias-then-gelu is three kernel")
    print("     launches and three full round-trips to global memory. A")
    print("     fused kernel keeps the output on-chip for the epilogue.")
    print("     Triton makes this easy.")
    print()
    print("  The point of today was not to beat the library -- it was to")
    print("  show that the library's gap is EXPLAINABLE and CLOSABLE with")
    print("  measurement-driven reasoning. Every remaining optimization has")
    print("  the same structure: form a hypothesis about the hardware,")
    print("  measure, adjust the code, verify.")


if __name__ == "__main__":
    main()
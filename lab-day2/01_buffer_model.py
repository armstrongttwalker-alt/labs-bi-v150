"""
01_buffer_model.py
------------------
Build a hardware model of the GPU compute path by experiment.

Where we start
--------------
A common mental model is:
    "the chip has some on-chip memory; as long as my tile fits, I'm fine."

In practice, a single tl.dot is not a monolithic operation -- it is a
pipeline that moves data through several *physically distinct* buffers:

    Global Memory  --DMA-->  Shared Memory / L1  -->  Registers
                                                      |
                                                   Compute Unit
                                                      |
                                                   Output Buffer

Each level has its own capacity. We test the limits and quantify the capacities.

What you will do
----------------
Part A: Run a broad tile sweep. Observe that some configs compile and
        run, others fail. Read the error patterns.

Part B: Targeted probes designed to stress *one* buffer at a time.
        Fill in the legality predicate from what you observed.

Part C: Use the model to produce the legal tile space we will hand to
        autotune in step 02.
"""
import torch
import triton
import triton.language as tl

from common import matmul_kernel, make_inputs


# =============================================================================
# Run one (BM, BN, BK) and report the outcome in a compact form
# =============================================================================
def try_config(M, N, K, BM, BN, BK, verbose_error=False):
    """Return (status, error_msg).

    status is one of 'OK' or 'FAIL'.
    """
    a, b = make_inputs(M, N, K)
    c = torch.empty((M, N), dtype=torch.float32, device="cuda")
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    try:
        matmul_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        )
        torch.cuda.synchronize()
        return "OK", None
    except Exception as e:
        msg = str(e)
        return "FAIL", msg[:200] if verbose_error else "tile too large or misaligned"


def print_row(BM, BN, BK, status, error_msg):
    acc_kb = BM * BN * 4 / 1024
    ab_kb = (BM + BN) * BK * 4 / 1024
    if status == "OK":
        print(f"  {BM:>4} {BN:>4} {BK:>4}  "
              f"BM*BN*4={acc_kb:>6.1f}KB  (BM+BN)*BK*4={ab_kb:>7.1f}KB  "
              f"-> OK")
    else:
        print(f"  {BM:>4} {BN:>4} {BK:>4}  "
              f"BM*BN*4={acc_kb:>6.1f}KB  (BM+BN)*BK*4={ab_kb:>7.1f}KB  "
              f"-> FAIL: {error_msg}")


# =============================================================================
# Part A: broad sweep
# =============================================================================
def part_a():
    print("=" * 72)
    print("Part A: which tiles compile?")
    print("=" * 72)
    print()
    print("We try 10 tile configurations at (M=N=K=1024). Some will work,")
    print("some will fail. We want to see the failure pattern.")
    print()

    configs = [
        ( 64,  64,  32),   # Day 1 default
        (128, 128,  32),
        (128, 128, 128),
        (256, 256,  16),
        (256, 256,  32),
        (256, 256, 128),
        ( 64,  64, 512),
        ( 64,  64, 1024),
        (128, 128, 512),
        (512, 512, 128),
    ]

    print(f"  {'BM':>4} {'BN':>4} {'BK':>4}  "
          f"{'(acc proxy)':<16}  {'(ops proxy)':<18}  result")
    for BM, BN, BK in configs:
        status, error_msg = try_config(1024, 1024, 1024, BM, BN, BK)
        print_row(BM, BN, BK, status, error_msg)

    print()
    print("Discussion")
    print("-" * 72)
    print("  1. The failures are NOT explained by 'total tile size' alone.")
    print("  2. Different hardware may have different constraints.")
    print("  3. Alignment requirements (multiples of 16 or 32) also matter.")


# =============================================================================
# Part B: targeted probes
# =============================================================================
def part_b():
    print()
    print("=" * 72)
    print("Part B: isolating constraints")
    print("=" * 72)
    print()
    print("Hypothesis:")
    print("  Accumulator (output tile) is bounded by BM * BN * 4")
    print("  Operands are bounded by (BM + BN) * BK * 4")
    print()

    probes = [
        ("Acc stress 1",  256, 128,  16),   # acc=128KB, operands=24KB
        ("Acc stress 2",  256, 256,  16),   # acc=256KB, operands=32KB
        ("Acc stress 3",  128, 256,  16),   # acc=128KB, operands=24KB
        ("Ops stress 1",   64,  64, 512),   # acc=16KB,  operands=256KB
        ("Ops stress 2",   64,  64, 1024),  # acc=16KB,  operands=512KB
        ("Ops stress 3",  128, 128, 512),   # acc=64KB,  operands=512KB
    ]

    print(f"  {'label':<14}  {'BM':>4} {'BN':>4} {'BK':>4}  result")
    for label, BM, BN, BK in probes:
        status, error_msg = try_config(1024, 1024, 1024, BM, BN, BK)
        acc_kb = BM * BN * 4 / 1024
        ab_kb = (BM + BN) * BK * 4 / 1024
        if status == "OK":
            print(f"  {label:<14}  {BM:>4} {BN:>4} {BK:>4}  "
                  f"acc={acc_kb:>5.0f}KB  ops={ab_kb:>5.0f}KB  -> OK")
        else:
            print(f"  {label:<14}  {BM:>4} {BN:>4} {BK:>4}  "
                  f"acc={acc_kb:>5.0f}KB  ops={ab_kb:>5.0f}KB  "
                  f"-> FAIL")

    print()
    print("What the probes tell us")
    print("-" * 72)
    print("  * Observe which configurations pass and which fail.")
    print("  * Write down the thresholds you observe.")


# =============================================================================
# Part C: turn the model into a legality predicate
# =============================================================================
# TODO (student) ------------------------------------------------------------
# Based on Parts A and B, fill in the two thresholds (in KB) used below.
#
# Hardware facts you should have just measured:
#   Accumulator capacity = ?     KB   (bounds BM * BN * 4)
#   Operand capacity = ?     KB   (bounds (BM+BN) * BK * 4)
#
# Set ACC_MAX_KB and OPS_BUDGET_KB to the values you measured.
# ---------------------------------------------------------------------------
# >>> YOUR CODE HERE >>>
ACC_MAX_KB = ...      # hint: one of {64, 128, 192, 256}
OPS_BUDGET_KB = ...   # hint: depends on shared memory size
# <<< END OF YOUR CODE <<<


def is_legal(BM, BN, BK, dtype_bytes=4):
    """Return True if (BM, BN, BK) fits in both constraints."""
    acc_kb = BM * BN * dtype_bytes / 1024
    ops_kb = (BM + BN) * BK * dtype_bytes / 1024
    aligned = (BM % 16 == 0) and (BN % 16 == 0) and (BK % 16 == 0)
    return (acc_kb <= ACC_MAX_KB) and (ops_kb <= OPS_BUDGET_KB) and aligned


def part_c():
    print()
    print("=" * 72)
    print("Part C: the legal tile space")
    print("=" * 72)
    print()
    if ACC_MAX_KB is ... or OPS_BUDGET_KB is ...:
        print("  (Fill in ACC_MAX_KB and OPS_BUDGET_KB first, then re-run.)")
        return

    print(f"  Using ACC_MAX_KB = {ACC_MAX_KB}, OPS_BUDGET_KB = {OPS_BUDGET_KB}")
    print(f"  Plus 16-alignment on every block dimension.")
    print()

    BMs = [16, 32, 48, 64, 96, 128, 160, 192, 256]
    BNs = [16, 32, 48, 64, 96, 128, 160, 192, 256]
    BKs = [16, 32, 48, 64, 96, 128, 160, 192]

    legal = []
    for BM in BMs:
        for BN in BNs:
            for BK in BKs:
                if is_legal(BM, BN, BK):
                    legal.append((BM, BN, BK))

    total = len(BMs) * len(BNs) * len(BKs)
    print(f"  Searched {total} combinations in the grid, "
          f"{len(legal)} are legal.")
    print()
    print("  First 12 legal tiles (as a sanity check):")
    for BM, BN, BK in legal[:12]:
        print(f"    (BM={BM:>3}, BN={BN:>3}, BK={BK:>3})")
    if len(legal) > 12:
        print(f"    ... and {len(legal) - 12} more.")
    print()
    print("  This legal space is exactly what we will hand to @triton.autotune")
    print("  in step 02.")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
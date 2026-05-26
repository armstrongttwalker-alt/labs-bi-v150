# Day 2 Handout: Tuning a Triton Matmul

## Framing

Yesterday we wrote a tiled matrix multiplication kernel in Triton and
saw it run correctly. It was also slower than `torch.matmul`. A reasonable
question: *is that gap fundamental, or is it the result of specific
hardware details we haven't engaged with?*

Today's answer is the second. We are going to build a **model of the
GPU compute path from measurement** -- not from the documentation, not
from analogy -- and then tune our kernel against that measured model.

The lab has four steps, but the logic is a single loop repeated:

> **Hypothesis → Experiment → Data → Refined understanding → Code change**

If you walk away with one thing today, let it be this loop. The
specific hardware facts (buffer capacities, alignment preferences) are
useful, but the loop is the transferable skill.

---

## Step 00: Calibrate

### What we assume coming in

Day 1's benchmark used 256, 512, 1024 square matmuls. Kernel times at
those sizes are a few tens of microseconds -- close enough to kernel
launch overhead and synchronization granularity that any
"optimization" we do could be lost in noise.

### Why this step exists

Before we claim a speedup, we need a stable baseline. Moving the test
shape up to 1024^3 and 2048^3 pushes kernel time into the
hundreds-of-microseconds to milliseconds range, where the signal is
much larger than the noise floor.

---

## Step 01: Discover Buffer Structure

### The hypothesis

A common mental model is:

> "The chip has some on-chip memory. As long as my tile fits in it,
> the hardware figures out the rest."

In practice, a single `tl.dot(A_tile, B_tile)` is not a
monolithic operation -- it's a short pipeline that moves data through
several physically distinct buffers:

```
 Global Memory
      │  DMA
      ▼
  Shared Memory / L1    ← A, B tiles buffered here
      │
      ▼
  Registers             ← accumulator lives here
      │
  Compute Unit
      │
 Global Memory
```

Each stage has its own capacity. If this model is correct, we should see
**independent failure modes** when we make tiles too large.

### What to write down

- **Accumulator capacity** = constrains BM * BN
- **Operand capacity** = constrains (BM + BN) * BK
- **Alignment requirements** = typically multiples of 16 or 32

### Legality predicate

Combining the constraints with alignment:

```python
def is_legal(BM, BN, BK):
    acc_ok = (BM * BN * 4) <= ACC_MAX_KB * 1024
    ops_ok = ((BM + BN) * BK * 4) <= OPS_BUDGET_KB * 1024
    aligned = (BM % 16 == 0) and (BN % 16 == 0) and (BK % 16 == 0)
    return acc_ok and ops_ok and aligned
```

This predicate is the output of Step 01. Step 02 uses it as the
search space for autotune.

---

## Step 02: Find the Best Legal Tile, and Understand Why

### The hypothesis (Part A)

Step 01 told us *which* tiles are legal. The next question: which legal
tile is fastest?

Naive expectation: the biggest legal square tile wins, because it
maximizes arithmetic intensity.

Correction that the experiment will force: "biggest square" is
probably wrong. A and B are loaded with different stride patterns
(row of A vs column of B), so the real cost of growing BM is not the
same as the real cost of growing BN. The best tile is likely
**asymmetric**.

### Why autotune, not manual search

With a few dozen legal configs, hand-timing each is feasible but
miserable. `@triton.autotune` compiles and times each once, caches the
result keyed on `(M, N, K)`, and uses the winner from then on.

### Part B: why BK matters even though the formula says it shouldn't

Arithmetic intensity for a single tile is

    AI = BM·BN / (2·(BM+BN))    (FLOPs per byte, BK cancels)

BK does not appear. So why does BK sweep produce any change at all?

Because AI only counts data *inside one tile*. The real kernel
has a loop over K that runs `K/BK` iterations. Each iteration
has per-iteration overhead. Small BK means many iterations and a
lot of that overhead; large BK amortizes it.

Past some BK, the compute unit is already saturated. That's the
**roofline ceiling** you're watching the curve flatten against.

### What to write down

- **Tile shape is often asymmetric**. Trust autotune.
- **BK is the dominant knob**. Once `BM·BN` approaches the capacity cap,
  shifting BK is what moves you along the roofline.
- **Prefer BK as a multiple of 32** for best alignment.

---

## Step 03: Summary

The final table collects everything on one line.

The "buffer-model tile" row is interesting: even without running
autotune, just picking *any* tile the model says is legal and big
gets you most of the way to the library. **Most of the win is in the
model**; autotune gives you the last few percent.

### The transferable pattern

The thing we did today is applicable beyond matmul. For any kernel
you write:

1. **What is the hardware pipeline actually doing?** Draw the path
   the data takes from global memory to result. Name each on-chip
   buffer it touches.
2. **What capacity limits those stages?** Don't trust the docs
   alone. Construct targeted experiments that stress one stage at a
   time.
3. **What is the legal tile space?** Express the capacity limits as
   a predicate. Alignment requirements go in here too.
4. **What is optimal inside the legal space?** Use autotune rather
   than intuition. The real costs are usually asymmetric.
5. **Sweep the one knob that didn't show up in your model**, and
   look for the mismatch between predicted and measured. That's
   where the next optimization lives.

This is what "understanding the chip" means in practice.
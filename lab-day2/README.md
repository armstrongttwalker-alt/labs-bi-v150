# FlagGems-Triton Lab Day 2

Hands-on lab for the international training course
**"Software and Hardware Foundations of Intelligent Computing Systems"**.

Day 2 picks up where Day 1 left off: the tiled Triton matmul in
`04_matmul_compare.py` runs correctly but is slower than
`torch.matmul`. Today we close most of that gap -- **not** by learning
more Triton syntax, but by building an empirical model of the GPU
compute path and tuning our kernel against measured hardware
facts.

---

## Quick Start

Each new terminal:

```bash
source setup.sh
```

Then run the four steps in order:

```bash
python 00_baseline.py              # calibrate measurement
python 01_buffer_model.py          # main event: measure buffer constraints
python 02_autotune_roofline.py     # autotune + BK sweep
python 03_summary.py               # final comparison table
```

---

## What You'll Do

| File                        | What you do                                        | Time    |
|-----------------------------|-----------------------------------------------------|---------|
| `00_baseline.py`            | Run only. Record starting numbers.                  | ~10 min |
| `01_buffer_model.py`        | Observe failures, fill in two capacity values.      | ~35 min |
| `02_autotune_roofline.py`   | Fill in the legal-tile config generator.            | ~40 min |
| `03_summary.py`             | Run only. Read the summary table.                   | ~15 min |

Total ~100 minutes of hands-on, leaving time for discussion.

---

## The Story

**Day 1 left us asking**: why is our straightforward tiled matmul so
much slower than `torch.matmul`? Is the gap a fact of life ("vendors
always win") or can we reason about it?

**Day 2 answers**: it is reasonable, and it is closable. We'll make
and measure three claims about the GPU compute path:

1. The compute path is a **multi-stage pipeline** with several
   independent on-chip buffers, not a single shared scratchpad.
2. Each stage has a **concrete, measurable capacity** that bounds what
   tile shapes we can legally pick.
3. Inside the legal space, **which tile is best is not obvious** --
   the optimum is asymmetric. We let autotune find it, and we build
   roofline intuition for why.

The detailed narrative, with the "hypothesis → experiment → data"
structure for each step, is in [`handout.md`](./handout.md).

---

## Environment

Identical to Day 1:

| Component       | Version           |
|-----------------|-------------------|
| Hardware        | Iluvatar BI-V150 (32 GB HBM) |
| Driver          | IX-ML 4.4.0       |
| CUDA Version    | 10.2              |
| PyTorch         | 2.2.0+            |
| Triton          | 2.2.0+            |
| FlagGems        | Latest            |
| Python          | 3.9-3.11         |

---

## Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: CUDA out of memory` | Tile too large | Use smaller BLOCK sizes |
| Step 02 takes 1-3 min on first run | autotune compiles every config | Wait. Subsequent runs use cache |
| Numbers fluctuate run-to-run | Small shape + few iters = noise | Stick to the script's shapes |
| Triton compilation errors | Tile exceeds hardware limits | Check is_legal() function |

To clear the Triton compilation cache:

```bash
rm -rf ~/.triton/cache ~/.triton/dump
```

---

## Resources

- Day 1 lab: Getting Started with BI-V150 and Triton
- Triton documentation: https://triton-lang.org/main/
- OpenAI Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/

---

## License

CC BY-NC 4.0 — Creative Commons Attribution-NonCommercial 4.0 International.
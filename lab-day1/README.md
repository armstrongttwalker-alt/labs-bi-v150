# FlagGems-Triton Lab Day 1

Hands-on lab for the international training course
**"Software and Hardware Foundations of Intelligent Computing Systems"**.

This lab introduces the BI-V150 (TianShu) AI accelerator software stack through PyTorch and Triton with FlagGems.
**You will write code yourself** -- each script has small TODO blocks for you
to fill in. Reference solutions are at the end of `handout.md` if you get stuck.

---

## Quick Start

In your environment, every time you open a new terminal:

```bash
source setup.sh
```

Then:

```bash
python 00_check_env.py # verify everything works
```

If `00_check_env.py` prints "All checks passed", you are ready.

---

## How This Lab Works

Each numbered script has a small section marked with `TODO` and
`>>> YOUR CODE HERE >>>` / `<<< END OF YOUR CODE <<<` markers.

1. Read the comments above the TODO -- they describe what each line should do.
2. Write the missing code.
3. Run the script. The script prints PASS / FAIL or a results table.
4. If you get stuck, check the **Reference Solutions** section in `handout.md`.
5. Then move to the **Try-it-yourself** prompts at the bottom of each script.

---

## Lab Order

| File                       | What you'll do                                          |
|----------------------------|---------------------------------------------------------|
| `00_check_env.py`          | (No coding.) Verify the environment.                    |
| `01_torch_hello.py`        | Run PyTorch on GPU (4 lines).                           |
| `02_vector_add_triton.py`  | Write a Triton vector-add kernel (5-6 lines).           |
| `03_softmax_compare.py`    | Implement numerically stable softmax in Triton.         |
| `04_matmul_compare.py`     | Fill in the K-loop accumulator of a tiled matmul.       |

For the full handout (lecture notes, exercises, reference solutions), see
[`handout.md`](./handout.md).

---

## Environment

This lab assumes an environment with:

| Component       | Version           |
|-----------------|-------------------|
| Hardware        | Iluvatar BI-V150 (32 GB HBM) |
| Driver          | IX-ML 4.4.0       |
| CUDA Version    | 10.2              |
| PyTorch         | 2.2.0+            |
| Triton          | 2.2.0+            |
| FlagGems        | Latest (from source) |
| Python          | 3.9-3.11         |

FlagGems provides a Triton-based operator library that enables running
PyTorch models on BI-V150 with competitive performance.

---

## Setup History (How This Environment Was Configured)

This section records the actual commands used to set up the lab
environment, in case you need to reproduce it elsewhere or troubleshoot.

### Step 0: Install dependencies

```bash
pip install -U scikit-build-core pybind11 ninja cmake
```

### Step 1: Install FlagGems

```bash
git clone https://github.com/flagos-ai/FlagGems.git
cd FlagGems
pip install --no-build-isolation -e .
cd ..
```

### Step 2: Enable FlagGems

```python
import flag_gems
flag_gems.enable()  # Enables Triton-based operators
```

### Step 3: End-to-end verification

```bash
python -c "import torch, triton, flag_gems; flag_gems.enable(); print('all ok')"
```

If this prints `all ok`, the environment is ready. The first Triton
kernel launch will JIT-compile and take some time; subsequent launches use the cache in
`~/.triton/cache/` and are fast.

---

## Common Issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: CUDA out of memory` | Insufficient device memory | Reduce tensor sizes or batch size |
| Triton's first run takes ~1 minute | JIT compilation | Wait. Subsequent runs use `~/.triton/cache/` |
| `Device do not support double dtype` warning | Hardware does not support fp64 | Use `dtype=torch.float32` explicitly |
| `ImportError: cannot import name 'flag_gems'` | FlagGems not installed | Run `pip install --no-build-isolation -e .` in FlagGems directory |

---

## Resources

- FlagGems (GitHub): https://github.com/flagos-ai/FlagGems
- Triton documentation: https://triton-lang.org/main/
- OpenAI Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/

---

## License

CC BY-NC 4.0 — Creative Commons Attribution-NonCommercial 4.0 International.

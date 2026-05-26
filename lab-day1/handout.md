# Lab Day 1: Getting Started with BI-V150 and Triton

**Course:** Software and Hardware Foundations of Intelligent Computing Systems
**Duration:** 2 hours
**Audience:** Faculty members (instructors-in-training)

---

## Learning Goals

By the end of this lab, you will be able to:

1. Explain the BI-V150 software stack and how it relates to the NVIDIA CUDA stack you may already know.
2. Run a basic PyTorch program on a BI-V150 GPU using standard CUDA API.
3. Write and execute a Triton kernel on BI-V150 hardware.
4. Compare the performance of a Triton kernel against optimized library operators on the same GPU.
5. Know where to find documentation and examples to continue exploring on your own.

---

## Two Narratives We Will Demonstrate

This lab is built around **two independent claims** about Triton:

**Narrative 1 -- Triton has real value.**
Inside the same GPU, a small Python Triton kernel can be a serious
alternative to optimized library kernels. Sometimes it even wins. We show this by benchmarking your Triton kernel against
`torch.softmax` / `torch.matmul`, both running on the GPU.

**Narrative 2 -- Triton source code is portable.**
The same `@triton.jit` kernel runs on multiple backends (NVIDIA, AMD, BI-V150), byte-for-byte. We
show this by comparing the source you write today with the OpenAI Triton
tutorial source. The portability claim is *source-level*.

> **About today's hardware:** Your environment has a BI-V150 GPU.
> FlagGems provides a Triton-based operator library that enables
> running PyTorch models with competitive performance.

---

## 0. Today's Hardware and Software (5 min)

| Component   | Version          | Notes |
|-------------|------------------|-------|
| Hardware    | Iluvatar BI-V150 | 32 GB HBM AI accelerator |
| Driver      | IX-ML 4.4.0      | Iluvatar driver stack |
| CUDA API    | 10.2 compatible  | Standard PyTorch device interface |
| PyTorch     | 2.2.0+           | Standard PyTorch |
| Triton      | 2.2.0+           | OpenAI Triton |
| FlagGems    | Latest           | Triton-based operator library |

**Key idea:** FlagGems provides optimized Triton implementations of
PyTorch operators. You can use standard PyTorch CUDA code and FlagGems
will handle the backend for Iluvatar BI-V150.

**Check your GPU:**
```bash
ixsmi  # Iluvatar GPU status
```

---

## 1. The Software Stack (15 min, lecture)

```
+--------------------------------------------------+
|   User code: PyTorch / Triton                    |   <-- you write here
+--------------------------------------------------+
|   FlagGems: Triton-based operator library        |   <-- optimized ops
+--------------------------------------------------+
|   Triton compiler                                |
+--------------------------------------------------+
|   Driver + Hardware (BI-V150)                    |
+--------------------------------------------------+
```

| Term               | Description                        |
|--------------------|------------------------------------|
| GPU                | AI accelerator device              |
| CUDA API           | Standard PyTorch device interface  |
| FlagGems           | Triton-based operator library      |
| Triton             | Python DSL for GPU kernels         |

**Why Triton matters here.** Writing GPU kernels directly is powerful
but has a steep learning curve. Triton lets you write a
Python DSL and run it on multiple hardware backends with **the same source code**.
That is the central promise we will demonstrate today.

---

## 2. Environment Check (10 min, hands-on)

Open a terminal in your environment.

**Important:** every time you open a new terminal, run:

```bash
source setup.sh
```

Then run the environment check:

```bash
python 00_check_env.py
```

If everything prints OK, you are ready.

---

## 3. PyTorch on GPU: Hello World (15 min, hands-on)

Open `01_torch_hello.py` in your editor. Find the TODO block in
`matmul_on_device`. Fill in the device selection and tensor creation.

After filling it in, run:

```bash
python 01_torch_hello.py
```

You should see a matmul timing and "Output shape and dtype correct."

> **What to notice:**
> - Standard PyTorch CUDA code works on BI-V150.
> - `device="cuda"` is the standard way to use GPU in PyTorch.

> **Try it yourself:** modify the script to compute `D = (A @ B) + bias`
> where `bias` is a length-N random vector, and verify against CPU.

---

## 4. Your First Triton Kernel (30 min, hands-on)

Open `02_vector_add_triton.py`. The kernel `add_kernel` has a TODO with
five sub-tasks. Each task corresponds to one line you need to write.
The comments above the TODO list the exact API for each step.

Run:

```bash
python 02_vector_add_triton.py
```

If your kernel is correct you will see `PASS` and `Max absolute difference: 0.00e+00`.

> **What to notice:**
> - The `@triton.jit` decorator marks a Triton kernel.
> - The first run is slow (JIT compilation). Triton
>   compiles your Python kernel and caches it in `~/.triton/cache/`.
>   The second run is fast.

> **Discussion questions:**
> - What does `tl.program_id(axis=0)` correspond to in CUDA terminology?
>   *(blockIdx.x.)*
> - What happens if you change `BLOCK_SIZE` from 1024 to 512?
>   *(Recompilation, because BLOCK_SIZE is a `tl.constexpr`.)*

---

## 5. Triton vs PyTorch: Performance Comparison (40 min, hands-on)

This is the centerpiece. We compare your hand-written Triton kernels
against optimized library operators. **Both run on the same GPU.**

### 5a. Softmax

Open `03_softmax_compare.py`. Two TODOs in the kernel: load the row, and
implement numerically stable softmax (`max -> subtract -> exp -> sum -> divide`).

Run:

```bash
python 03_softmax_compare.py
```

You will see a table of timings.

> **Discussion:**
> - Why might Triton beat the library on softmax for some shapes?
>   *(One pass over memory; no intermediate tensors.)*
> - Why is the comparison fair?
>   *(Same hardware, same dtype, same problem size.)*

### 5b. Matrix Multiplication

Open `04_matmul_compare.py`. Two TODOs: initialize the accumulator tile,
and write the K-loop body that loads tiles of A and B and accumulates
their product.

Run:

```bash
python 04_matmul_compare.py
```

> **Discussion:**
> - Where does Triton's performance come from? *(Tile-based memory
>   access, automatic pipelining inserted by the compiler.)*
> - Takeaway: **Triton's value is highest for custom ops** that have no
>   library version.

> **This file is also the baseline for tomorrow's lab,** where we will
> progressively optimize this matmul.

---

## 6. Wrap-up and Resources (10 min)

**What we have demonstrated:**
- Inside the GPU: a 30-line Python Triton kernel can compete with
  optimized library kernels on real workloads.
- Across hardware: the same Triton source runs on multiple backends.
- The ecosystem is mature enough to support real research and teaching.

**Resources:**
- FlagGems (GitHub): https://github.com/flagos-ai/FlagGems
- Triton documentation: https://triton-lang.org/main/
- OpenAI Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/

**Tomorrow:** progressively tune the matmul kernel.

---

## Quick Reference: Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `RuntimeError: CUDA out of memory` | Insufficient memory | Reduce tensor sizes |
| `ImportError: flag_gems` | Not installed | Run install in FlagGems directory |
| First Triton run takes ~1 minute | JIT compilation | Wait 30s-2min |
| `Device do not support double dtype` | Hardware has no fp64 | Use `dtype=torch.float32` |

---

# Reference Solutions

> If you got everything working without looking at this section,
> congratulations -- skip to the Try-it-yourself prompts in each script.

## Solution: `01_torch_hello.py`

```python
def matmul_on_device(M, K, N):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.randn(M, K, dtype=torch.float32, device=device)
    B = torch.randn(K, N, dtype=torch.float32, device=device)
    _ = A @ B
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        C = A @ B
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / 50 * 1000
    return C, elapsed_ms
```

## Solution: `02_vector_add_triton.py`

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

Notes:
- `tl.arange(0, BLOCK_SIZE)` produces a **vector** of offsets, not a single
  scalar. The kernel is *block-vectorized* by construction.
- The `mask` is essential. Without it, the last block (when `n_elements`
  is not divisible by `BLOCK_SIZE`) would read/write past the buffer.

## Solution: `03_softmax_compare.py`

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # (1) load
    row_start_ptr = input_ptr + row_idx * input_row_stride
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

    # (2) numerically stable softmax
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)
```

Notes:
- `other=-float("inf")` matters: it makes masked-out positions invisible
  to `tl.max`. With `other=0` or `other=-1e30` you would get subtly wrong
  results when the row contains negative numbers near the chosen sentinel.
- The whole softmax is fused into one kernel: one read of the row, one
  write of the output.

## Solution: `04_matmul_compare.py`

```python
# (1) accumulator
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

# (2) K loop body
for k in range(0, tl.cdiv(K, BLOCK_K)):
    k_remaining = K - k * BLOCK_K
    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
    b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
    acc += tl.dot(a, b)
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
```

Notes:
- The accumulator stays in registers across the K loop -- it is never
  written back to global memory until the final `tl.store`. This is the key
  performance trick of tile-based matmul.
- `other=0.0` for the masked load lets us pretend the partial K tile is
  full of zeros; zeros do not affect the dot product.
- `tl.dot(a, b)` performs a tile-level matmul.
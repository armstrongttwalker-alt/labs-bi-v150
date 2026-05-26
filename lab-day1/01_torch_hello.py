"""
01_torch_hello.py
-----------------
PyTorch "hello world" on the BI-V150 GPU.

If you have written PyTorch code for an NVIDIA GPU before, the changes
for BI-V150 with FlagGems are minimal. FlagGems provides CUDA-compatible
API, so standard PyTorch CUDA code works:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = x.cuda()                    # standard CUDA API
    x = torch.randn(.., device='cuda')  # standard CUDA device

Everything else -- model definition, training loop, autograd -- is identical.

In this exercise you will run a basic PyTorch computation on GPU. There is
exactly ONE small thing to fill in.
"""
import time
import torch


def matmul_on_device(M: int, K: int, N: int):
    """
    Run a (M, K) @ (K, N) float32 matmul on the GPU and time it.

    TODO ------------------------------------------------------------------
    Fill in the device selection and tensor creation. Use CUDA if available,
    otherwise fall back to CPU.

    Your task: set device and create tensors A and B on the device.
    -----------------------------------------------------------------------
    """
    # >>> YOUR CODE HERE >>>
    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.randn(M, K, dtype=torch.float32, device=device)
    B = torch.randn(K, N, dtype=torch.float32, device=device)
    # <<< END OF YOUR CODE <<<

    _ = A @ B                       # warm-up
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        C = A @ B
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / 50 * 1000
    return C, elapsed_ms


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPU devices: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    print()

    M = K = N = 1024
    C_gpu, elapsed_ms = matmul_on_device(M, K, N)
    print(f"Matmul {M}x{K} @ {K}x{N}: {elapsed_ms:.3f} ms / iter on GPU")

    # Cross-check against CPU
    torch.manual_seed(0)
    A_cpu = torch.randn(M, K, dtype=torch.float32)
    B_cpu = torch.randn(K, N, dtype=torch.float32)
    C_cpu = A_cpu @ B_cpu
    # Note: A and B inside matmul_on_device are different random tensors,
    # so we only sanity-check shape and dtype here, not values.
    assert C_gpu.shape == (M, N), f"expected ({M},{N}), got {tuple(C_gpu.shape)}"
    assert C_gpu.dtype == torch.float32
    print(f"Output shape and dtype correct.")

    # ----- Try-it-yourself -------------------------------------------------
    # 1. Add a bias vector of length N and compute  D = (A @ B) + bias.
    #    Verify the result against a CPU computation.
    # 2. Switch to dtype=torch.float16 and re-time. How much faster is it?
    # -----------------------------------------------------------------------


if __name__ == "__main__":
    main()
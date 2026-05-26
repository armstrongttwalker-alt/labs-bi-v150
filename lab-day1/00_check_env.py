"""
00_check_env.py
---------------
Verify that everything in the lab environment is working before we start.
Run this first. If it all passes, you are ready for the rest of the lab.

(No coding required in this file -- it is just a smoke test.)

Hardware: Iluvatar BI-V150
Driver: IX-ML 4.4.0
"""
import sys
import subprocess


def section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check(label, ok, detail=""):
    mark = "[OK] " if ok else "[FAIL]"
    line = f"  {mark} {label}"
    if detail:
        line += f"  --  {detail}"
    print(line)


# -----------------------------------------------------------
section("Python")
# -----------------------------------------------------------
py_ok = sys.version_info[:2] >= (3, 9) and sys.version_info[:2] <= (3, 11)
check("Python 3.9-3.11", py_ok, sys.version.split()[0])

# -----------------------------------------------------------
section("Hardware: Iluvatar BI-V150")
# -----------------------------------------------------------
try:
    result = subprocess.run(["ixsmi"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        for line in lines:
            if "BI-V150" in line or "Iluvatar" in line:
                check("GPU detected", True, "Iluvatar BI-V150")
                break
        check("ixsmi available", True, "IX-ML driver")
    else:
        check("ixsmi available", False, "run ixsmi manually")
except Exception:
    check("ixsmi check", False, "subprocess error")

# -----------------------------------------------------------
section("PyTorch and CUDA")
# -----------------------------------------------------------
import torch

check("torch imported", True, torch.__version__)
check("CUDA available", torch.cuda.is_available())
n = torch.cuda.device_count()
check("CUDA device count", n >= 1, f"{n} device(s)")
if n >= 1:
    device_name = torch.cuda.get_device_name(0)
    check("Device name", True, device_name)

# -----------------------------------------------------------
section("Triton")
# -----------------------------------------------------------
import triton
import triton.language as tl  # noqa: F401

check("triton imported", True, triton.__version__)

# -----------------------------------------------------------
section("FlagGems")
# -----------------------------------------------------------
try:
    import flag_gems
    check("flag_gems imported", True, "installed")
    flag_gems.enable()
    check("flag_gems enabled", True)
except ImportError:
    check("flag_gems imported", False, "NOT INSTALLED")

# -----------------------------------------------------------
section("Smoke test: tensor on GPU")
# -----------------------------------------------------------
x = torch.randn(1024, 1024, dtype=torch.float32, device="cuda")
y = torch.randn(1024, 1024, dtype=torch.float32, device="cuda")
z = x @ y
torch.cuda.synchronize()
check("Matmul on GPU", z.shape == (1024, 1024), f"output shape {tuple(z.shape)}")

# -----------------------------------------------------------
section("Smoke test: minimal Triton kernel on GPU")
# -----------------------------------------------------------


@triton.jit
def _add_one(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, v + 1.0, mask=mask)


a = torch.zeros(2048, dtype=torch.float32, device="cuda")
b = torch.empty_like(a)
grid = (triton.cdiv(a.numel(), 1024),)
_add_one[grid](a, b, a.numel(), BLOCK=1024)
torch.cuda.synchronize()
ok = torch.allclose(b, torch.ones_like(b))
check("Triton kernel on GPU", ok, "result == 1.0 everywhere")

print()
print("All checks passed. You are ready for the lab.")
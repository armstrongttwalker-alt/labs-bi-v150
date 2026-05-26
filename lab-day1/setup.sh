#!/bin/bash
# Source this file at the start of every new terminal session.
# Usage:  source setup.sh

echo "======================================================"
echo "  FlagGems-Triton Lab Day 1 - Environment"
echo "  Hardware: Iluvatar BI-V150"
echo "======================================================"

# Check GPU status
ixsmi 2>/dev/null || nvidia-smi 2>/dev/null || echo "GPU info not available"

python - <<'PY'
import torch
import triton

print(f"PyTorch:         {torch.__version__}")
print(f"Triton:          {triton.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count:    {torch.cuda.device_count()}")
    print(f"Device name:     {torch.cuda.get_device_name(0)}")

try:
    import flag_gems
    print(f"FlagGems:        installed")
except ImportError:
    print("FlagGems:        NOT INSTALLED - run: pip install --no-build-isolation -e . in FlagGems directory")
PY

echo "======================================================"
echo "  Ready. If you see all versions above and 'CUDA"
echo "  available: True', your environment is OK."
echo "======================================================"

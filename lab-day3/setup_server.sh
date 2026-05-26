#!/bin/bash
# setup_server.sh - Setup script for vLLM inference service on Iluvatar BI-V150

echo "======================================================"
echo "  LLM Deployment Lab - Server Setup"
echo "  Hardware: Iluvatar BI-V150"
echo "======================================================"

# Step 0: Check GPU
echo ""
echo "Step 0: Checking GPU..."
ixsmi 2>/dev/null || echo "ixsmi not found, trying nvidia-smi..."
nvidia-smi 2>/dev/null || echo "GPU status tools not available"

# Step 1: Check if model path is provided
MODEL_PATH="${1:-/mnt/share/user_homes/models/Qwen3-4B}"
echo ""
echo "Model path: $MODEL_PATH"
echo ""

# Step 2: Install dependencies if needed
echo "Step 1: Checking dependencies..."
pip install -U scikit-build-core pybind11 ninja cmake 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -U scikit-build-core pybind11 ninja cmake
}

# Step 3: Check FlagGems
echo ""
echo "Step 2: Checking FlagGems..."
python -c "import flag_gems; print('FlagGems installed')" 2>/dev/null || {
    echo "FlagGems not found. Install with:"
    echo "  git clone https://github.com/flagos-ai/FlagGems.git"
    echo "  cd FlagGems && pip install --no-build-isolation -e ."
}

# Step 4: Check vllm-plugin-FL
echo ""
echo "Step 3: Checking vllm-plugin-FL..."
python -c "import vllm_fl; print('vllm-plugin-FL installed')" 2>/dev/null || {
    echo "vllm-plugin-FL not found. Install with:"
    echo "  git clone https://github.com/flagos-ai/vllm-plugin-FL.git"
    echo "  cd vllm-plugin-FL && pip install --no-build-isolation -e . --no-deps"
}

echo ""
echo "======================================================"
echo "  To start the server, run:"
echo ""
echo "  export VLLM_ENGINE_ITERATION_TIMEOUT_S=36000"
echo "  export VLLM_RPC_TIMEOUT=36000000"
echo "  vllm serve $MODEL_PATH --served-model-name qwen --enforce-eager"
echo ""
echo "  Note: First startup takes ~15 minutes"
echo "======================================================"
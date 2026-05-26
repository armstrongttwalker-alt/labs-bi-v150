# Lab Day 3: LLM Deployment on BI-V150

**Course:** Software and Hardware Foundations of Intelligent Computing Systems
**Duration:** 2 hours
**Audience:** Faculty members (instructors-in-training)

---

## Learning Goals

By the end of this lab, you will be able to:

1. Set up an LLM inference service using vLLM with FlagGems backend
2. Understand the software stack for LLM deployment on BI-V150
3. Use the OpenAI-compatible API to interact with local models
4. Implement multi-turn conversation with context management

---

## Overview

This lab integrates the server-side setup (from the background materials) with client-side interaction. You will:

1. **Server Side**: Set up and run a vLLM inference service for Qwen3-4B
2. **Client Side**: Use Python clients to interact with the model

---

## Part 1: Server Setup (60 min)

### Step 1: Download the Model

Download Qwen3-4B from ModelScope:

```bash
modelscope download --model Qwen/Qwen3-4B
```

This will download the model to your local cache. Note the path for later use.

### Step 2: Install Dependencies

Install the required software stack:

```bash
# Install build dependencies
pip install -U scikit-build-core pybind11 ninja cmake

# Clone and install FlagGems
git clone https://github.com/flagos-ai/FlagGems.git
cd FlagGems
pip install --no-build-isolation -e .
cd ..

# Clone and install vllm-plugin-FL
git clone https://github.com/flagos-ai/vllm-plugin-FL.git
cd vllm-plugin-FL
pip install --no-build-isolation -e . --no-deps
cd ..
```

### Step 3: Run the Inference Server

Start the vLLM server:

```bash
export VLLM_ENGINE_ITERATION_TIMEOUT_S=36000
export VLLM_RPC_TIMEOUT=36000000
vllm serve /path/to/Qwen3-4B/ --served-model-name qwen --enforce-eager
```

**Important timing notes:**
- **First server startup**: ~15 minutes (model loading and compilation)
- **Subsequent startups**: <2 minutes (cached)
- **First inference**: ~8 minutes (kernel compilation)
- **Subsequent inferences**: Several seconds

---

## Part 2: Client Interaction (45 min)

### Understanding the API

The vLLM service exposes an OpenAI-compatible REST API. This means:

1. You can use the standard `openai` Python SDK
2. Just change `base_url` to point to your local service
3. The API key can be any string (vLLM doesn't validate it)

### Single-Turn Chat

Run the simple client:

```bash
python client.py
```

This sends a single message and prints the streaming response.

**Key code pattern:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Your question here"}],
    max_tokens=1024,
    temperature=0.7,
    stream=True,  # Enable streaming
)

for chunk in response:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Multi-Turn Conversation

Run the interactive client:

```bash
python test.py
```

**Features:**
- Maintains conversation history
- Token estimation and context limit handling
- Interactive commands: `/quit`, `/clear`, `/tokens`

**Command-line options:**

```bash
python test.py --base-url http://localhost:8000/v1 \
               --model qwen \
               --max-tokens 512 \
               --max-context 4096 \
               --temperature 0.7
```

---

## Part 3: Exercises (15 min)

### Exercise 1: Try Different Parameters

Experiment with different generation parameters:

```python
response = client.chat.completions.create(
    model="qwen",
    messages=[...],
    temperature=0.9,  # Higher = more creative
    top_p=0.95,       # Nucleus sampling
    max_tokens=2048,  # Longer responses
)
```

Observe how different settings affect the output.

### Exercise 2: System Prompts

Add a custom system prompt:

```bash
python test.py --system "You are a helpful coding assistant. Always provide code examples."
```

### Exercise 3: Context Management

Test the context limit handling:

1. Have a long conversation until you see the warning
2. Use `/tokens` to check usage
3. Use `/clear` to reset

---

## Software Stack Summary

```
+--------------------------------------------------+
|   Client: OpenAI Python SDK                      |
+--------------------------------------------------+
|   vLLM: LLM inference engine                     |
+--------------------------------------------------+
|   vllm-plugin-FL: FlagOS integration             |
+--------------------------------------------------+
|   FlagGems: Triton-based operator library        |
+--------------------------------------------------+
|   Triton: GPU kernel language                    |
+--------------------------------------------------+
|   Hardware: BI-V150 (TianShu)                    |
+--------------------------------------------------+
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Connection refused | Server not running | Start vLLM server |
| Slow first response | Kernel compilation | Wait; subsequent calls are fast |
| Out of memory | Model too large | Use smaller model or quantization |
| Import errors | Missing dependencies | Reinstall FlagGems/vllm-plugin-FL |

---

## Resources

- FlagGems: https://github.com/flagos-ai/FlagGems
- vLLM: https://docs.vllm.ai/
- OpenAI API: https://platform.openai.com/docs/api-reference
- Qwen models: https://modelscope.cn/models/Qwen

---

## Summary

Today you learned:

1. **Server setup**: How to deploy an LLM inference service on BI-V150
2. **OpenAI-compatible API**: How to use standard SDKs with local models
3. **Multi-turn conversation**: How to maintain context and handle limits
4. **Software stack**: How FlagGems, vLLM, and Triton work together

This completes the three-day hands-on series on GPU programming, optimization, and LLM deployment!
# FlagGems-Triton Lab Day 3: LLM Deployment

Hands-on lab for the China-Africa AI Compute Faculty Development Program.

Day 3 focuses on deploying and serving large language models (LLMs) locally. Building on Days 1-2, which covered GPU programming and performance optimization, today you will learn to deploy an LLM inference service and interact with it.

---

## Quick Start

### Prerequisites

- A running vLLM inference service at `http://localhost:8000/v1`
- Python 3.9+ with `openai` package installed

```bash
pip install openai
```

### Basic Usage

Simple single-turn chat:

```bash
python client.py
```

Continuous multi-turn conversation:

```bash
python test.py
```

---

## Files

| File | Description |
|------|-------------|
| `handout.md` | Detailed lab handbook with step-by-step instructions |
| `setup_server.sh` | Server setup script for vLLM with FlagGems |
| `client.py` | Simple OpenAI-compatible client for single-turn chat |
| `test.py` | Multi-turn continuous chat client with context management |

---

## What You'll Learn

### 1. Server-Side Setup

- Download model from ModelScope
- Install FlagGems and vllm-plugin-FL
- Configure and run vLLM inference service

### 2. OpenAI-Compatible API

The local inference service exposes an OpenAI-compatible REST API. This means:

- You can use the official `openai` Python SDK
- Simple `base_url` change points to your local service
- Code written for OpenAI works with local models

### 3. Single-Turn vs Multi-Turn Chat

- `client.py` sends one message and receives one response
- `test.py` maintains conversation history for multi-turn dialogue

### 4. Context Management

The multi-turn client (`test.py`) includes:

- **Token estimation** - Rough count of CJK and English tokens
- **Context limit handling** - Automatic trimming when approaching limits
- **Interactive commands** - `/quit`, `/clear`, `/tokens`

---

## Server Setup

### Step 1: Download Model

```bash
modelscope download --model Qwen/Qwen3-4B
```

### Step 2: Install Dependencies

```bash
pip install -U scikit-build-core pybind11 ninja cmake
git clone https://github.com/flagos-ai/FlagGems.git
cd FlagGems
pip install --no-build-isolation -e .
cd ..
git clone https://github.com/flagos-ai/vllm-plugin-FL.git
pip install --no-build-isolation -e . --no-deps
cd ..
```

### Step 3: Run Server

```bash
export VLLM_ENGINE_ITERATION_TIMEOUT_S=36000
export VLLM_RPC_TIMEOUT=36000000
vllm serve /path/to/Qwen3-4B/ --served-model-name qwen --enforce-eager
```

**Note:** First-time server startup takes ~15 minutes. Subsequent startups are faster (~2 minutes).

---

## Client Usage

### test.py — Interactive Commands

```
You: Hello, who are you?
Qwen: I am a helpful AI assistant...
[~120/4096 tokens]

You: /tokens
[Used ~120 / 4096 tokens]

You: /clear
[History cleared]

You: /quit
Goodbye!
```

| Command | Description |
|---------|-------------|
| `/quit` | Exit the chat |
| `/clear` | Clear conversation history |
| `/tokens` | Show current token usage |

### Command-line Options

```bash
python test.py --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--base-url` | `http://localhost:8000/v1` | vLLM service URL |
| `--model` | `qwen` | Model name |
| `--max-tokens` | `512` | Max tokens per response |
| `--max-context` | `4096` | Total context window limit |
| `--temperature` | `0.7` | Generation temperature |
| `--system` | `You are a helpful assistant.` | System prompt |

---

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Connection refused` | Service not running | Start vLLM service on port 8000 |
| Empty response | Model not loaded | Wait for model initialization |
| `Context limit exceeded` | Long conversation | Use `/clear` or wait for auto-trim |
| Slow first response | Model loading | Subsequent responses are faster |
| First inference takes ~8 min | Model warming | Second inference is much faster |

---

## Resources

- Day 1 lab: GPU & Triton basics
- Day 2 lab: Performance tuning and autotuning
- vLLM documentation: https://docs.vllm.ai/
- OpenAI API reference: https://platform.openai.com/docs/api-reference
- FlagGems: https://github.com/flagos-ai/FlagGems

---

## License

CC BY-NC 4.0 — Creative Commons Attribution-NonCommercial 4.0 International.
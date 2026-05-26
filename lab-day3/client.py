#!/usr/bin/env python3
"""
client.py - Simple single-turn chat client for LLM inference service.

Usage: python client.py

This client connects to a local vLLM inference service and sends a
single message to the model.
"""
from openai import OpenAI

# Configure the client to use local vLLM service
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # vLLM doesn't require a real API key
)

# Send a message and print the response
response = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Give me a short introduction to large language models."}],
    max_tokens=1024,
    temperature=0.7,
    top_p=0.8,
    stream=True,  # Stream the response for better UX
)

print("Response:")
print("-" * 60)
for chunk in response:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
print("-" * 60)
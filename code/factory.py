"""factory.py

Small helpers to build model wrappers by name.

This file contains no credentials. Configure API keys via environment variables.
"""

from __future__ import annotations

from typing import List

from model import AnthropicModel, GoogleModel, OllamaModel, OpenAIModel, OpenRouterModel


DEFAULT_MODELS: List[str] = [
    "gemma2:9b",
    "qwen2.5-coder:7b-instruct",
    "deepseek-r1:8b",
    "llama3.1:8b",
    "mistral:v0.3",
]


def build_model(model_name: str):
    """Return the appropriate wrapper for the given model name."""
    if model_name == "gpt-4o" or model_name.startswith("gpt-"):
        return OpenAIModel(model_name=model_name)
    if model_name.startswith("gemini"):
        return GoogleModel(model_name=model_name)
    if model_name.startswith("claude"):
        return AnthropicModel(model_name=model_name)
    if "/" in model_name:
        return OpenRouterModel(model_name=model_name)
    return OllamaModel(model_name=model_name)

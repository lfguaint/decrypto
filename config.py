from __future__ import annotations
import os
import socket
from dataclasses import dataclass, field
from typing import Optional


def _dgx_base_url(
    usp_host: str = "200.144.205.82",
    port: int = 8000,
    timeout: float = 1.0,
) -> str:
    """Return the DGX base URL, auto-detecting USP network vs. SSH tunnel.

    Priority:
      1. DGX_BASE_URL env var (manual override always wins)
      2. USP IP — if reachable within `timeout` seconds
      3. localhost — assumes SSH tunnel is active
    """
    if url := os.environ.get("DGX_BASE_URL"):
        return url
    try:
        with socket.create_connection((usp_host, port), timeout=timeout):
            return f"http://{usp_host}:{port}/v1"
    except OSError:
        return f"http://localhost:{port}/v1"


# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------

class Backend:
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    CLUSTER = "cluster"   # any OpenAI-compatible local cluster (vLLM, etc.)


BACKEND_SETTINGS = {
    Backend.OLLAMA: {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "default_headers": {},
    },
    Backend.OPENROUTER: {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_headers": {
            "HTTP-Referer": "https://github.com/lfguaint/decrypto",
            "X-Title": "Decrypto LLM Benchmark",
        },
    },
    Backend.CLUSTER: {
        # DGX Spark cluster (OpenAI-compatible API).
        # URL auto-detected: USP network → 200.144.205.82:8000, fora → localhost:8000 (túnel SSH).
        # Override: export DGX_BASE_URL=<url>  |  Key: export DGX_API_KEY=<key>
        "base_url": _dgx_base_url(),
        "api_key_env": "DGX_API_KEY",
        "default_headers": {},
    },
}


# ---------------------------------------------------------------------------
# Ollama models (name:tag format)
# ---------------------------------------------------------------------------

OLLAMA_MODELS = {
    "gpt-oss-20b": "gpt-oss:20b",
    "qwen3-8b":    "qwen3:8b",
}


# ---------------------------------------------------------------------------
# OpenRouter models (shortcut → full model ID)
# Verify current IDs and pricing at https://openrouter.ai/models
# ---------------------------------------------------------------------------

OPENROUTER_MODELS = {
    # ── Experimental models ───────────────────────────────────────────────
    # T1 — lightweight
    "gemma":        "google/gemma-4-31b-it",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen3-32b":    "qwen/qwen3-32b",
    # T2 — capable
    "deepseek-flash":   "deepseek/deepseek-v4-flash",  # reference model
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gpt-6-luna":       "openai/gpt-6-luna",
    "claude-haiku":     "anthropic/claude-haiku-4.5",
    # T3 — frontier
    "deepseek-pro": "deepseek/deepseek-v4-pro",
    "kimi-k3":      "moonshotai/kimi-k3",
    "gpt-6.1-sol":  "openai/gpt-6.1-sol",
}

# Each model is served by one pinned OpenRouter provider, never routed elsewhere,
# and the pin stays fixed for the whole benchmark even if prices change. Rule: the
# developer's own provider when OpenRouter offers it, otherwise a provider at fp8 or
# higher precision, chosen per model (the reference DeepSeek V4 Flash is on DeepInfra
# for its stable pricing, 2026-10-27). Quantization is pinned where reported.
# Providers of the other models are still to be reviewed one by one; service tiers
# (flex/standard/priority) of OpenAI, Google AI Studio and DeepInfra-Gemma are not pinned yet.
OPENROUTER_PROVIDERS = {
    "google/gemma-4-31b-it":      {"provider": "deepinfra", "quantization": "fp8"},
    "openai/gpt-oss-120b":        {"provider": "deepinfra", "quantization": "bf16"},
    "qwen/qwen3-32b":             {"provider": "deepinfra", "quantization": "fp8"},
    "deepseek/deepseek-v4-flash": {"provider": "deepinfra", "quantization": "fp8"},
    "deepseek/deepseek-v4-pro":   {"provider": "streamlake", "quantization": "fp8"},
    "moonshotai/kimi-k3":         {"provider": "moonshotai"},
    "google/gemini-2.5-flash":    {"provider": "google-ai-studio"},
    "openai/gpt-6-luna":          {"provider": "openai"},
    "openai/gpt-6.1-sol":         {"provider": "openai"},
    "anthropic/claude-haiku-4.5": {"provider": "anthropic"},
}


def openrouter_routing(model_id: str) -> dict:
    """OpenRouter `provider` request field pinning `model_id` to its provider."""
    pin = OPENROUTER_PROVIDERS.get(model_id)
    if pin is None:
        raise ValueError(f"No pinned OpenRouter provider for {model_id!r}; add it to OPENROUTER_PROVIDERS.")
    routing = {"order": [pin["provider"]], "allow_fallbacks": False}
    if "quantization" in pin:
        routing["quantizations"] = [pin["quantization"]]
    return routing

# ---------------------------------------------------------------------------
# DGX Spark cluster models
# nemotron-3-super: 256K ctx, complex reasoning — use max_tokens ≥ 1024
# gemma-4-31b:      32K ctx, fast, accepts images
# ---------------------------------------------------------------------------

DGX_MODELS = {
    "nemotron-super": "nemotron-3-super",
    "dgx-gemma":      "gemma-4-31b",
}

# Combined shortcut registry
ALL_MODELS = {**OPENROUTER_MODELS, **OLLAMA_MODELS, **DGX_MODELS}


# ---------------------------------------------------------------------------
# Agent and experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    encryptor_model: str = "gpt-oss-20b"
    decoder_model: str = "gpt-oss-20b"
    interceptor_model: str = "gpt-oss-20b"
    encryptor_backend: str | None = None   # None = inherit ExperimentConfig.backend
    decoder_backend: str | None = None
    interceptor_backend: str | None = None
    encryptor_temperature: float | None = None    # None = provider default
    decoder_temperature: float | None = None
    interceptor_temperature: float | None = None
    max_tokens: int | None = None   # None = no limit (model default)
    prompt_version: int = 3         # label stored per experiment; prompts.py implements only v3

@dataclass
class ExperimentConfig:
    backend: str = Backend.OLLAMA      # switch to Backend.OPENROUTER for paid models
    num_keywords: int = 4              # number of keyword cards (digits range from 1 to num_keywords)
    code_length: int = 3               # number of digits per code
    num_keyword_sets: int = 3
    games_per_keyword_set: int = 2
    rounds_per_game: int = 5
    seed: Optional[int] = None
    keyword_seed: Optional[int] = None  # keyword-set stream; None = same as seed
    keyword_sets_file: Optional[str] = None  # read sets 1..num_keyword_sets from this JSON instead of drawing
    agents: AgentConfig = field(default_factory=AgentConfig)

    @property
    def num_possible_codes(self) -> int:
        """P(num_keywords, code_length) — total distinct codes."""
        n, k = self.num_keywords, self.code_length
        result = 1
        for i in range(k):
            result *= (n - i)
        return result

    def __post_init__(self) -> None:
        if self.keyword_seed is None:
            self.keyword_seed = self.seed
        if self.code_length >= self.num_keywords:
            raise ValueError(
                f"code_length ({self.code_length}) must be less than "
                f"num_keywords ({self.num_keywords})."
            )

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------

class Backend:
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


BACKEND_SETTINGS = {
    Backend.OLLAMA: {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",          # required by the SDK but ignored by Ollama
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
    "claude-opus":  "anthropic/claude-opus-4.6",
    "claude-haiku": "anthropic/claude-haiku-4.5",
    "gpt-5.2":      "openai/gpt-5.2",
    "gpt-5-mini":   "openai/gpt-5-mini",
    "gemini-pro":   "google/gemini-2.5-pro",
    "grok":         "x-ai/grok-4",
    "deepseek":     "deepseek/deepseek-v3.2",
    "llama":        "meta-llama/llama-4-maverick",
    "qwen":         "qwen/qwen3-235b-a22b",
    "kimi":         "moonshotai/kimi-k2.5",
}

# Combined shortcut registry (Ollama shortcuts take precedence when using Ollama)
ALL_MODELS = {**OPENROUTER_MODELS, **OLLAMA_MODELS}


# ---------------------------------------------------------------------------
# Agent and experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    encryptor_model: str = "gpt-oss-20b"
    decoder_model: str = "gpt-oss-20b"
    interceptor_model: str = "gpt-oss-20b"
    encryptor_temperature: float = 0.7
    decoder_temperature: float = 0.7
    interceptor_temperature: float = 0.7
    max_tokens: int | None = None   # None = no limit (model default)
    prompt_version: int = 1         # 1 = baseline; 2 = strategic encryptor


@dataclass
class ExperimentConfig:
    backend: str = Backend.OLLAMA      # switch to Backend.OPENROUTER for paid models
    num_keywords: int = 4              # number of keyword cards (digits range from 1 to num_keywords)
    code_length: int = 3               # number of digits per code
    num_keyword_sets: int = 3
    games_per_keyword_set: int = 2
    rounds_per_game: int = 5
    seed: Optional[int] = None
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
        if self.code_length >= self.num_keywords:
            raise ValueError(
                f"code_length ({self.code_length}) must be less than "
                f"num_keywords ({self.num_keywords})."
            )
        if self.rounds_per_game >= self.num_possible_codes:
            raise ValueError(
                f"rounds_per_game ({self.rounds_per_game}) must be less than "
                f"num_possible_codes ({self.num_possible_codes}) "
                f"[P({self.num_keywords},{self.code_length})]."
            )

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
    "gpt-5.4":      "openai/gpt-5.4",        # $2.50/$15.00 — Mar 2026
    "gpt-5.4-mini": "openai/gpt-5.4-mini",   # $0.75/$4.50  — Mar 2026
    "gemini":       "google/gemini-3.5-flash",     # $1.50/$9.00 — May 2026
    "grok":         "x-ai/grok-4.3",         # $1.25/$2.50 — Apr 2026
    "deepseek":       "deepseek/deepseek-v4-pro",    # $0.44/$0.87 — Apr 2026
    "deepseek-flash": "deepseek/deepseek-v4-flash",  # $0.10/$0.20 — Apr 2026
    "llama":        "meta-llama/llama-4-maverick",
    "qwen":         "qwen/qwen3.5-397b-a17b",      # $0.60/$3.60 — open-weights
    "kimi":         "moonshotai/kimi-k2.6",        # $0.68/$3.41 — Apr 2026
    "glm":          "z-ai/glm-5",                  # $0.60/$1.92 — Feb 2026, open-weights
    "gpt-oss":      "openai/gpt-oss-120b",         # ~$0.18/$0.80 — OpenAI open-weights
    "gemma":        "google/gemma-4-31b-it",       # $0.12/$0.35 — Google open-weights
    "minimax":      "minimax/minimax-m3",          # $0.30/$1.20 — MiniMax
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
    encryptor_temperature: float | None = None    # None = provider default
    decoder_temperature: float | None = None
    interceptor_temperature: float | None = None
    max_tokens: int | None = None   # None = no limit (model default)
    prompt_version: int = 1         # 1 = baseline; 2 = strategic encryptor
    encryptor_system_template: str | None = None  # custom template (GEPA); overrides prompt_version


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

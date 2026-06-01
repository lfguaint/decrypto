from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# The 10 models selected for the Decrypto benchmark.
# Shortcuts → exact OpenRouter model IDs.
# Verify current IDs and pricing at https://openrouter.ai/models
# ---------------------------------------------------------------------------
MODELS = {
    # Shortcut            OpenRouter ID                          Input    Output
    "claude-opus":    "anthropic/claude-opus-4.6",        #   $5.00   $25.00 /MTok
    "claude-haiku":   "anthropic/claude-haiku-4.5",       #   $1.00    $5.00 /MTok
    "gpt-5.2":        "openai/gpt-5.2",                   #   $1.75   $14.00 /MTok
    "gpt-5-mini":     "openai/gpt-5-mini",                #   $0.25    $2.00 /MTok
    "gemini-pro":     "google/gemini-2.5-pro",            #   $1.25   $10.00 /MTok
    "grok":           "x-ai/grok-4",                      #   $3.00   $15.00 /MTok
    "deepseek":       "deepseek/deepseek-v3.2",           #   $0.28    $0.40 /MTok
    "llama":          "meta-llama/llama-4-maverick",      #   $0.15    $0.60 /MTok
    "qwen":           "qwen/qwen3-235b-a22b",             #   $0.46    $1.82 /MTok
    "kimi":           "moonshotai/kimi-k2.5",             #   $0.60    $3.00 /MTok
}

# Default model for quick local tests — cheapest option in the list
_DEFAULT = "deepseek"


@dataclass
class AgentConfig:
    encryptor_model: str = _DEFAULT
    decoder_model: str = _DEFAULT
    interceptor_model: str = _DEFAULT
    max_tokens: int = 256
    temperature: float = 0.7   # same as Mini-Mafia paper (Appendix A.3)


@dataclass
class ExperimentConfig:
    num_keyword_sets: int = 3
    games_per_keyword_set: int = 2
    rounds_per_game: int = 8
    seed: Optional[int] = None
    agents: AgentConfig = field(default_factory=AgentConfig)

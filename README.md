# Decrypto LLM Benchmark

A multi-agent simulation of the board game **Decrypto** to evaluate and compare LLM capabilities across three distinct roles: strategic communication, informed decoding, and pattern inference under information asymmetry.

Inspired by the analytical framework introduced in [*Deceive, Detect, and Disclose: Large Language Models Play Mini-Mafia*](https://arxiv.org/abs/2509.23023) (Costa & Vicente, 2026).

---

## Game Overview

Each game uses **4 keyword cards** (numbered 1–4) known only to the Encryptor and Decoder. Over **8 rounds**, a 3-digit code is drawn each round (no repetition, from the 24 possible permutations of {1,2,3,4} taken 3 at a time). Three LLM agents play:

| Role | Information | Objective |
|------|-------------|-----------|
| **Encryptor** | Knows the 4 keywords | Give 3 clues (one per digit) that the Decoder can decode but the Interceptor cannot |
| **Decoder** | Knows the 4 keywords | Reconstruct the 3-digit code from the clues |
| **Interceptor** | Does **not** know the keywords | Infer the code from the clues and accumulated history |

All clues, guesses, and actual codes are **public** and accumulate across rounds — the Interceptor can learn the keyword mapping over time. Only the keyword cards themselves are private.

### Round order

```
Encryptor gives 3 clues  →  Interceptor guesses  →  Decoder guesses
```

### Outcome categories per round

| Outcome | Condition |
|---------|-----------|
| ✅ Successful transmission | Decoder correct **and** Interceptor wrong |
| ❌ Decoder failure | Decoder guesses wrong |
| 🔓 Interceptor success | Interceptor guesses correctly |

---

## Setup

```bash
# 1. Clone
git clone https://github.com/<your-username>/decrypto.git
cd decrypto

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenRouter API key
export OPENROUTER_API_KEY="sk-or-..."
```

Get an API key at [openrouter.ai/keys](https://openrouter.ai/keys). OpenRouter provides access to all 10 benchmark models through a single OpenAI-compatible API.

---

## Usage

```bash
# Quick test — all roles use DeepSeek V3.2 (cheapest, ~$0.003/game)
python main.py --seed 42

# Use model shortcuts (see table below)
python main.py \
  --encryptor-model claude-opus \
  --decoder-model gpt-5-mini \
  --interceptor-model grok \
  --seed 42

# Full OpenRouter model IDs also accepted
python main.py --interceptor-model x-ai/grok-4

# More games for statistical stability
python main.py --keyword-sets 5 --games-per-set 4 --seed 42
```

Results are saved to `results.json` by default. Use `--output` to change the path.

### All options

```
--keyword-sets     Number of keyword sets to generate     (default: 3)
--games-per-set    Games per keyword set                  (default: 2)
--rounds           Rounds per game, max 24                (default: 8)
--seed             Random seed for reproducibility
--encryptor-model  Model shortcut or full OpenRouter ID
--decoder-model    Model shortcut or full OpenRouter ID
--interceptor-model Model shortcut or full OpenRouter ID
--temperature      Sampling temperature                   (default: 0.7)
--max-tokens       Max output tokens per call             (default: 256)
--output           Output JSON file                       (default: results.json)
```

---

## Benchmark Models

Ten models selected to cover the current frontier, mid-tier, and value segments across major providers:

| Shortcut | Model | Provider | Input $/MTok | Output $/MTok |
|----------|-------|----------|:---:|:---:|
| `claude-opus` | Claude Opus 4.6 | Anthropic | $5.00 | $25.00 |
| `claude-haiku` | Claude Haiku 4.5 | Anthropic | $1.00 | $5.00 |
| `gpt-5.2` | GPT-5.2 | OpenAI | $1.75 | $14.00 |
| `gpt-5-mini` | GPT-5 Mini | OpenAI | $0.25 | $2.00 |
| `gemini-pro` | Gemini 2.5 Pro | Google | $1.25 | $10.00 |
| `grok` | Grok 4 | xAI | $3.00 | $15.00 |
| `deepseek` | DeepSeek V3.2 | DeepSeek | $0.28 | $0.40 |
| `llama` | Llama 4 Maverick | Meta | $0.15 | $0.60 |
| `qwen` | Qwen3 235B A22B | Alibaba | $0.46 | $1.82 |
| `kimi` | Kimi K2.5 | Moonshot AI | $0.60 | $3.00 |

All models are accessed via [OpenRouter](https://openrouter.ai/models). Model IDs and pricing may change — verify at `openrouter.ai/models`.

---

## Output Format

`results.json` contains a list of game summaries:

```json
[
  {
    "keyword_set_id": 1,
    "game_id": 1,
    "total_rounds": 8,
    "successful_transmissions": 5,
    "decoder_failures": 2,
    "interceptor_successes": 1,
    "rounds": [
      {
        "round": 1,
        "code": [2, 4, 1],
        "clues": ["rain", "float", "fortress"],
        "interceptor_guess": [1, 3, 2],
        "decoder_guess": [2, 4, 1],
        "decoder_correct": true,
        "interceptor_correct": false,
        "successful": true
      }
    ]
  }
]
```

---

## Project Structure

```
decrypto/
├── main.py          # CLI entry point
├── runner.py        # Experiment orchestrator (multiple games/keyword sets)
├── game.py          # Single game loop (8 rounds)
├── agents.py        # LLM wrappers for each role (via OpenRouter)
├── prompts.py       # System and user prompt templates
├── keywords.py      # Random keyword generation from word list
├── models.py        # Data types: KeywordPair, RoundRecord, GameResult
├── config.py        # Model registry and experiment configuration
└── requirements.txt
```

---

## Suggested Experiment Design

To compare models systematically (analogous to the Mini-Mafia benchmark approach), fix two roles with a reference model and vary the third:

```bash
REFERENCE="deepseek"

for MODEL in claude-opus claude-haiku gpt-5.2 gpt-5-mini gemini-pro grok deepseek llama qwen kimi; do
  # Vary the Interceptor
  python main.py \
    --encryptor-model $REFERENCE \
    --decoder-model $REFERENCE \
    --interceptor-model $MODEL \
    --keyword-sets 3 --games-per-set 5 --seed 42 \
    --output results_interceptor_${MODEL}.json
done
```

Repeat for Encryptor and Decoder. This yields 30 configurations — manageable and directly comparable across roles.

---

## Reference

```bibtex
@article{costa2026minimalfia,
  title   = {Deceive, Detect, and Disclose: Large Language Models Play Mini-Mafia},
  author  = {Costa, Davi Bastos and Vicente, Renato},
  journal = {arXiv preprint arXiv:2509.23023},
  year    = {2026}
}
```

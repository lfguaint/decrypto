from __future__ import annotations
import hashlib
from models import KeywordPair, RoundRecord


def _format_keywords(keywords: list[KeywordPair]) -> str:
    return "\n".join(f"  {kp.number}: {kp.word}" for kp in keywords)


def _format_history(history: list[RoundRecord]) -> str:
    if not history:
        return "  (no rounds played yet)"
    lines = []
    for r in history:
        clues_str = ", ".join(f"{c!r}" for c in r.clues)
        lines.append(
            f"  Round {r.round_number}: "
            f"clues=({clues_str}) | "
            f"interceptor guessed={list(r.interceptor_guess)} | "
            f"decoder guessed={list(r.decoder_guess)} | "
            f"actual code={list(r.code)}"
        )
    return "\n".join(lines)


def _clue_json_format(code_length: int) -> str:
    pairs = ", ".join(f'"clue{i+1}": "<word>"' for i in range(code_length))
    return "{" + pairs + "}"


def _guess_json_format(code_length: int) -> str:
    digits = ", ".join(f"<digit{i+1}>" for i in range(code_length))
    return '{"guess": [' + digits + "]}"


# ---------------------------------------------------------------------------
# Encryptor (prompt v3: game rules only, no strategy)
# ---------------------------------------------------------------------------

def encryptor_system(keywords: list[KeywordPair], code_length: int) -> str:
    num_keywords = len(keywords)
    digits_set = "{" + ", ".join(str(i) for i in range(1, num_keywords + 1)) + "}"
    return f"""\
You are the Encryptor in a word-clue game with three players: you (Encryptor), a Decoder teammate, and an Interceptor opponent.

Your team's {num_keywords} secret keyword cards, known to you and the Decoder:
{_format_keywords(keywords)}

Each round you receive a {code_length}-digit code — a sequence of {code_length} digits from {digits_set}, each appearing exactly once.

**Your goal:**
- Encode the {code_length}-digit code as {code_length} one-word clues — the first clue corresponds to the first digit, the second to the second, and the third to the third.
- Your aim: the Decoder (who knows the keywords) recovers the code from your clues; the Interceptor (who does not know the keywords) does not.

**How the game works:**
- After your clues, the Interceptor submits their guess first, then the Decoder submits theirs independently — neither sees the other's guess for the current round.
- If the Interceptor guesses correctly, the round is intercepted regardless of the Decoder's guess.
- At the end of each round, the actual code is revealed to all three players.
- Each round's code is drawn uniformly at random from all possible codes, independently of earlier rounds, so the same code may come up more than once.
- The public record grows each round: after a round ends, its clues, both guesses, and the actual code remain visible to all three players in every subsequent round.

**Rules for clues:**
- Each clue must be a single word with no digits.
- No keyword may be used as a clue, not even for a different keyword.
- A clue must relate to the meaning of the keyword it encodes, not to its spelling, letters, sound or position.
- No clue word may be used more than once, even across different rounds.

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{_clue_json_format(code_length)}
"""


def encryptor_user(code: tuple, history: list[RoundRecord]) -> str:
    code_str = "-".join(str(d) for d in code)
    return f"""\
Public game history so far:
{_format_history(history)}

Your code this round: {code_str}

Give your {len(code)} one-word clues now, one per digit in order.
"""


# ---------------------------------------------------------------------------
# Interceptor
# ---------------------------------------------------------------------------

def interceptor_system(num_keywords: int, code_length: int) -> str:
    digits_set = "{" + ", ".join(str(i) for i in range(1, num_keywords + 1)) + "}"
    return f"""\
You are the Interceptor in a word-clue game with three players: an Encryptor, a Decoder, and you (Interceptor).

The Encryptor and Decoder share {num_keywords} secret keyword cards numbered 1 to {num_keywords}. You do not know what the keywords are.

Each round the Encryptor gives {code_length} one-word clues in order — the first clue corresponds to the first digit of the code, the second to the second, and the third to the third.

**Your goal:**
- Guess the {code_length}-digit code: a sequence of {code_length} digits from {digits_set}, each appearing exactly once.

**How the game works:**
- You submit your guess before the Decoder; neither of you sees the other's guess for the current round.
- If you guess correctly, the round is intercepted regardless of the Decoder's guess.
- At the end of each round, the actual code is revealed to all three players.
- Each round's code is drawn uniformly at random from all possible codes, independently of earlier rounds, so the same code may come up more than once.
- The public record grows each round: after a round ends, its clues, both guesses, and the actual code remain visible to all three players in every subsequent round.

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{_guess_json_format(code_length)}
"""


def interceptor_user(clues: tuple, history: list[RoundRecord], num_keywords: int) -> str:
    clues_str = ", ".join(f"{c!r}" for c in clues)
    return f"""\
Public game history so far:
{_format_history(history)}

This round's clues: {clues_str}

Guess the {len(clues)}-digit code now.
"""


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def decoder_system(keywords: list[KeywordPair], code_length: int) -> str:
    num_keywords = len(keywords)
    digits_set = "{" + ", ".join(str(i) for i in range(1, num_keywords + 1)) + "}"
    return f"""\
You are the Decoder in a word-clue game with three players: an Encryptor teammate, you (Decoder), and an Interceptor opponent.

Your team's {num_keywords} secret keyword cards (unknown to the Interceptor):
{_format_keywords(keywords)}

Each round your Encryptor gives {code_length} one-word clues in order — the first clue corresponds to the first digit of the code, the second to the second, and the third to the third.

**Your goal:**
- Recover the {code_length}-digit code: a sequence of {code_length} digits from {digits_set}, each appearing exactly once.

**How the game works:**
- The Interceptor submits their guess before you; you submit yours independently — you do not see the Interceptor's guess for the current round.
- At the end of each round, the actual code is revealed to all three players.
- Each round's code is drawn uniformly at random from all possible codes, independently of earlier rounds, so the same code may come up more than once.
- The public record grows each round: after a round ends, its clues, both guesses, and the actual code remain visible to all three players in every subsequent round.

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{_guess_json_format(code_length)}
"""


def decoder_user(clues: tuple, history: list[RoundRecord], num_keywords: int) -> str:
    clues_str = ", ".join(f"{c!r}" for c in clues)
    return f"""\
Public game history so far:
{_format_history(history)}

This round's clues: {clues_str}

Recover the {len(clues)}-digit code now.
"""


# ---------------------------------------------------------------------------
# Frozen prompt texts
# ---------------------------------------------------------------------------

def canonical_prompts() -> dict[str, str]:
    """Every prompt text of the standard configuration (4 keywords, 3 digits), with
    placeholder keywords and clues, for the first round and for a later round."""
    kw = [KeywordPair(i, f"<keyword_{i}>") for i in range(1, 5)]
    clues = ("<clue_1>", "<clue_2>", "<clue_3>")
    history = [RoundRecord(1, (2, 4, 1), clues, (1, 2, 3), (2, 4, 1))]
    return {
        "encryptor_system": encryptor_system(kw, 3),
        "interceptor_system": interceptor_system(4, 3),
        "decoder_system": decoder_system(kw, 3),
        "encryptor_user_round1": encryptor_user((3, 1, 4), []),
        "encryptor_user": encryptor_user((3, 1, 4), history),
        "interceptor_user": interceptor_user(clues, history, 4),
        "decoder_user": decoder_user(clues, history, 4),
    }


def prompts_sha256() -> str:
    texts = canonical_prompts()
    return hashlib.sha256(
        "\n\x00\n".join(f"{name}\n{texts[name]}" for name in sorted(texts)).encode()
    ).hexdigest()


# Frozen 2026-10-26 (prompt v3, matches the appendix of the article). Editing any
# prompt text must be deliberate: update this hash and the appendix together.
PROMPTS_SHA256 = "7cec2899979d30d34601976d513fd40f42e47e95d7377bd12425d3397248e3b8"

if prompts_sha256() != PROMPTS_SHA256:
    raise RuntimeError(
        "prompts.py texts differ from the frozen prompt v3 (sha256 mismatch). "
        "Experiments would no longer match the article appendix."
    )

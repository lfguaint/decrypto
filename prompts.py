from __future__ import annotations
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


def _example_code(num_keywords: int, code_length: int) -> str:
    # produce a simple example like "2-4-1"
    digits = list(range(2, num_keywords + 1))[:code_length - 1]
    example = [num_keywords] + digits
    return "-".join(str(d) for d in example[:code_length])


# ---------------------------------------------------------------------------
# Encryptor — v1 (baseline)
# ---------------------------------------------------------------------------

def _encryptor_system_v1(keywords: list[KeywordPair], code_length: int) -> str:
    num_keywords = len(keywords)
    return f"""\
You are the Encryptor in a word-clue game.

Your team has {num_keywords} secret keyword cards:
{_format_keywords(keywords)}

Each round you receive a {code_length}-digit code — a sequence of {code_length} distinct digits \
from 1 to {num_keywords} (e.g. {_example_code(num_keywords, code_length)}).
Your job is to give exactly {code_length} one-word clues — one clue per digit, in order — \
that hint at the corresponding keyword strongly enough for your Decoder teammate to reconstruct \
the code, yet subtly enough that the Interceptor (who does NOT know the keywords) cannot guess it.

Rules for clues:
- Each clue must be a single word (no numbers, no parts of the keyword itself).
- The clue must relate to the keyword it encodes, not to its position.
- Clues from all previous rounds are public — use that to your advantage but avoid patterns \
the Interceptor can exploit.

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{_clue_json_format(code_length)}
"""


# ---------------------------------------------------------------------------
# Encryptor — v2 (strategic)
# ---------------------------------------------------------------------------

def _encryptor_system_v2(keywords: list[KeywordPair], code_length: int) -> str:
    num_keywords = len(keywords)
    return f"""\
You are the Encryptor in a word-clue game.

Your team has {num_keywords} secret keyword cards:
{_format_keywords(keywords)}

Each round you receive a {code_length}-digit code — a sequence of {code_length} distinct digits \
from 1 to {num_keywords} (e.g. {_example_code(num_keywords, code_length)}).
Your job is to give exactly {code_length} one-word clues — one clue per digit, in order.

You have two competing objectives:
1. GUIDE your Decoder teammate: the clues must be clear enough for the Decoder — who knows \
the keyword cards — to recover the exact code.
2. MISLEAD the Interceptor: the Interceptor does NOT know the keywords but observes every \
clue and its corresponding revealed code. Each clue you give is a data point the Interceptor \
can use to map clues to keywords. After just a few rounds, obvious clues will expose the \
full keyword mapping and the Interceptor will intercept every subsequent code.

Strategic guidelines:
- Avoid direct synonyms or category labels (e.g. do not clue "tractor" with "farm" or \
"vehicle" — those are the first associations an outsider would try).
- Prefer oblique, thematic, or abstract associations that your Decoder can still resolve \
given the keyword context, but that an uninformed observer would not immediately link to \
the correct keyword.
- Vary your clue style across rounds to prevent the Interceptor from building a reliable \
association map.
- Remember: the Decoder has a significant advantage — they know the keywords. Even a subtle \
or indirect clue is enough for them. Sacrifice directness for secrecy.

Rules for clues:
- Each clue must be a single word (no numbers, no parts of the keyword itself).
- The clue must relate to the keyword it encodes, not to its position.

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{_clue_json_format(code_length)}
"""


# ---------------------------------------------------------------------------
# Encryptor — dispatcher
# ---------------------------------------------------------------------------

def encryptor_system(
    keywords: list[KeywordPair], code_length: int, prompt_version: int = 1
) -> str:
    if prompt_version == 2:
        return _encryptor_system_v2(keywords, code_length)
    return _encryptor_system_v1(keywords, code_length)


def encryptor_user(code: tuple, history: list[RoundRecord]) -> str:
    code_str = "-".join(str(d) for d in code)
    return f"""\
Public game history so far:
{_format_history(history)}

Your code this round: {code_str}

Provide your {len(code)} clues now.
"""


# ---------------------------------------------------------------------------
# Interceptor
# ---------------------------------------------------------------------------

def interceptor_system(num_keywords: int, code_length: int) -> str:
    return f"""\
You are the Interceptor in a word-clue game.

There are {num_keywords} keyword cards numbered 1 through {num_keywords}. \
You do NOT know what the keywords are.
Each round the Encryptor gives {code_length} one-word clues. Your goal is to deduce the \
{code_length}-digit code those clues represent — a sequence of {code_length} distinct digits \
from 1 to {num_keywords}.

Study the accumulating public history of (clues → actual codes) to infer patterns and \
eventually identify what each number might represent.

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{_guess_json_format(code_length)}
"""


def interceptor_user(clues: tuple, history: list[RoundRecord], num_keywords: int) -> str:
    clues_str = ", ".join(f"{c!r}" for c in clues)
    return f"""\
Public game history so far:
{_format_history(history)}

This round's clues: {clues_str}

Guess the {len(clues)}-digit code now. Each digit must be distinct and between 1 and {num_keywords}.
"""


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def decoder_system(keywords: list[KeywordPair], code_length: int) -> str:
    num_keywords = len(keywords)
    return f"""\
You are the Decoder in a word-clue game.

Your team's {num_keywords} secret keyword cards:
{_format_keywords(keywords)}

Each round your Encryptor teammate gives {code_length} one-word clues. Each clue corresponds \
to one of the keywords by its number (1-{num_keywords}). Your job is to reconstruct the \
{code_length}-digit code — the sequence of keyword numbers the clues point to.

The code contains {code_length} DISTINCT digits, each between 1 and {num_keywords}.

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{_guess_json_format(code_length)}
"""


def decoder_user(clues: tuple, history: list[RoundRecord], num_keywords: int) -> str:
    clues_str = ", ".join(f"{c!r}" for c in clues)
    return f"""\
Public game history so far:
{_format_history(history)}

This round's clues: {clues_str}

Decode the {len(clues)}-digit code now. Each digit must be distinct and between 1 and {num_keywords}.
"""

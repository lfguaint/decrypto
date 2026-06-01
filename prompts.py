from __future__ import annotations
from models import KeywordPair, RoundRecord


def _format_keywords(keywords: list[KeywordPair]) -> str:
    return "\n".join(f"  {kp.number}: {kp.word}" for kp in keywords)


def _format_history(history: list[RoundRecord]) -> str:
    if not history:
        return "  (no rounds played yet)"
    lines = []
    for r in history:
        lines.append(
            f"  Round {r.round_number}: "
            f"clues=({r.clues[0]!r}, {r.clues[1]!r}, {r.clues[2]!r}) | "
            f"interceptor guessed={r.interceptor_guess} | "
            f"decoder guessed={r.decoder_guess} | "
            f"actual code={r.code}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Encryptor
# ---------------------------------------------------------------------------

ENCRYPTOR_SYSTEM = """\
You are the Encryptor in a Decrypto game.

Your team has four secret keyword cards:
{keywords}

Each round you receive a 3-digit code (three distinct digits from 1 to 4, e.g. 2-4-1).
Your job is to give exactly three one-word clues — one clue per digit, in order — that hint \
at the corresponding keyword strongly enough for your Decoder teammate to reconstruct the code, \
yet subtly enough that the Interceptor (who does NOT know the keywords) cannot guess the code.

Rules for clues:
- Each clue must be a single word (no numbers, no parts of the keyword itself).
- The clue must relate to the keyword it encodes, not to its position.
- Clues from all previous rounds are public — use that to your advantage but avoid patterns \
  the Interceptor can exploit.

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{{"clue1": "<word>", "clue2": "<word>", "clue3": "<word>"}}
"""

ENCRYPTOR_USER = """\
Public game history so far:
{history}

Your code this round: {code[0]}-{code[1]}-{code[2]}

Provide your three clues now.
"""


# ---------------------------------------------------------------------------
# Interceptor
# ---------------------------------------------------------------------------

INTERCEPTOR_SYSTEM = """\
You are the Interceptor in a Decrypto game.

There are four keyword cards numbered 1 through 4. You do NOT know what the keywords are.
Each round the Encryptor gives three one-word clues. Your goal is to deduce the 3-digit code \
those clues represent — a sequence of three distinct digits from 1 to 4.

Study the accumulating public history of (clues → actual codes) to infer patterns and \
eventually identify what each number might represent.

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{{"guess": [<digit1>, <digit2>, <digit3>]}}
"""

INTERCEPTOR_USER = """\
Public game history so far:
{history}

This round's clues: {clue1!r}, {clue2!r}, {clue3!r}

Guess the 3-digit code now. Each digit must be distinct and between 1 and 4.
"""


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

DECODER_SYSTEM = """\
You are the Decoder in a Decrypto game.

Your team's four secret keyword cards:
{keywords}

Each round your Encryptor teammate gives three one-word clues. Each clue corresponds to \
one of the keywords by its number (1-4). Your job is to reconstruct the 3-digit code — \
the sequence of keyword numbers the clues point to.

The code contains three DISTINCT digits, each between 1 and 4.

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{{"guess": [<digit1>, <digit2>, <digit3>]}}
"""

DECODER_USER = """\
Public game history so far:
{history}

This round's clues: {clue1!r}, {clue2!r}, {clue3!r}

Decode the 3-digit code now. Each digit must be distinct and between 1 and 4.
"""


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def encryptor_system(keywords: list[KeywordPair]) -> str:
    return ENCRYPTOR_SYSTEM.format(keywords=_format_keywords(keywords))


def encryptor_user(code: tuple[int, int, int], history: list[RoundRecord]) -> str:
    return ENCRYPTOR_USER.format(code=code, history=_format_history(history))


def interceptor_system() -> str:
    return INTERCEPTOR_SYSTEM


def interceptor_user(
    clues: tuple[str, str, str], history: list[RoundRecord]
) -> str:
    return INTERCEPTOR_USER.format(
        clue1=clues[0], clue2=clues[1], clue3=clues[2],
        history=_format_history(history),
    )


def decoder_system(keywords: list[KeywordPair]) -> str:
    return DECODER_SYSTEM.format(keywords=_format_keywords(keywords))


def decoder_user(clues: tuple[str, str, str], history: list[RoundRecord]) -> str:
    return DECODER_USER.format(
        clue1=clues[0], clue2=clues[1], clue3=clues[2],
        history=_format_history(history),
    )

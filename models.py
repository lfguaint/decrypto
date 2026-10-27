from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


def count_multiword(clues) -> int:
    """Clues that break the one-word rule (whitespace inside; hyphenated words count as one)."""
    return sum(1 for c in clues if len(str(c).split()) > 1)


@dataclass
class KeywordPair:
    number: int
    word: str


@dataclass
class CallRecord:
    """One API call of one agent; fields the provider does not return stay None."""
    model_requested: str
    model_served: Optional[str] = None
    provider: Optional[str] = None
    response_id: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cost: Optional[float] = None
    started_at: Optional[str] = None  # local time the (successful) request was sent, ms precision
    elapsed_seconds: float = 0.0
    content: str = ""
    reasoning: Optional[str] = None
    attempt: int = 1                # 1..MAX_ATTEMPTS within the same game stage
    status: str = "ok"               # ok | format_error | rule_violation
    error: Optional[str] = None


@dataclass
class RoundRecord:
    round_number: int
    code: tuple            # tuple[int, ...] of length code_length
    clues: tuple           # tuple[str, ...] of length code_length
    interceptor_guess: tuple
    decoder_guess: tuple
    elapsed_seconds: float = 0.0
    calls: list = field(default_factory=list)   # [(agent, CallRecord)], every attempt

    @property
    def decoder_correct(self) -> bool:
        return self.decoder_guess == self.code

    @property
    def interceptor_correct(self) -> bool:
        return self.interceptor_guess == self.code

    @property
    def transmission_successful(self) -> bool:
        return self.decoder_correct and not self.interceptor_correct

    @property
    def multiword_clues(self) -> int:
        return count_multiword(self.clues)


@dataclass
class GameResult:
    keyword_set_id: int
    game_id: int
    keywords: list[KeywordPair]
    prompt_version: int = 1
    total_seconds: float = 0.0
    rounds: list[RoundRecord] = field(default_factory=list)
    aborted: bool = False          # an agent failed MAX_ATTEMPTS times; game is not saved
    abort_reason: Optional[str] = None
    pending_calls: list = field(default_factory=list)  # [(agent, CallRecord)] of the unfinished round

    @property
    def invalid_calls(self) -> list[tuple[int, str, CallRecord]]:
        """(round_number, agent, call) for every unusable attempt, incl. the unfinished round."""
        out = [(r.round_number, a, c) for r in self.rounds for a, c in r.calls if c.status != "ok"]
        out += [(len(self.rounds) + 1, a, c) for a, c in self.pending_calls if c.status != "ok"]
        return out

    @property
    def successful_transmissions(self) -> int:
        return sum(1 for r in self.rounds if r.transmission_successful)

    @property
    def decoder_failures(self) -> int:
        return sum(1 for r in self.rounds if not r.decoder_correct and not r.interceptor_correct)

    @property
    def interceptor_successes(self) -> int:
        return sum(1 for r in self.rounds if r.interceptor_correct)

    @property
    def total_cost(self) -> Optional[float]:
        """Sum of reported call costs (USD); None if the provider reported none."""
        all_calls = [c for r in self.rounds for _, c in r.calls] + [c for _, c in self.pending_calls]
        costs = [c.cost for c in all_calls if c.cost is not None]
        return sum(costs) if costs else None

    def summary(self) -> dict:
        return {
            "keyword_set_id": self.keyword_set_id,
            "game_id": self.game_id,
            "prompt_version": self.prompt_version,
            "keywords": {kp.number: kp.word for kp in self.keywords},
            "total_rounds": len(self.rounds),
            "successful_transmissions": self.successful_transmissions,
            "decoder_failures": self.decoder_failures,
            "interceptor_successes": self.interceptor_successes,
            "rounds": [
                {
                    "round": r.round_number,
                    "code": list(r.code),
                    "clues": list(r.clues),
                    "interceptor_guess": list(r.interceptor_guess),
                    "decoder_guess": list(r.decoder_guess),
                    "decoder_correct": r.decoder_correct,
                    "interceptor_correct": r.interceptor_correct,
                    "successful": r.transmission_successful,
                    "elapsed_seconds": round(r.elapsed_seconds, 2),
                    "invalid_attempts": sum(1 for _, c in r.calls if c.status != "ok"),
                }
                for r in self.rounds
            ],
        }

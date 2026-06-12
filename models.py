from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KeywordPair:
    number: int
    word: str


@dataclass
class RoundRecord:
    round_number: int
    code: tuple            # tuple[int, ...] of length code_length
    clues: tuple           # tuple[str, ...] of length code_length
    interceptor_guess: tuple
    decoder_guess: tuple
    elapsed_seconds: float = 0.0
    encryptor_fallback: bool = False    # parser fell back to random clues
    interceptor_fallback: bool = False  # parser fell back to random guess
    decoder_fallback: bool = False      # parser fell back to random guess

    @property
    def decoder_correct(self) -> bool:
        return self.decoder_guess == self.code

    @property
    def interceptor_correct(self) -> bool:
        return self.interceptor_guess == self.code

    @property
    def transmission_successful(self) -> bool:
        return self.decoder_correct and not self.interceptor_correct


@dataclass
class GameResult:
    keyword_set_id: int
    game_id: int
    keywords: list[KeywordPair]
    prompt_version: int = 1
    total_seconds: float = 0.0
    rounds: list[RoundRecord] = field(default_factory=list)

    @property
    def successful_transmissions(self) -> int:
        return sum(1 for r in self.rounds if r.transmission_successful)

    @property
    def decoder_failures(self) -> int:
        return sum(1 for r in self.rounds if not r.decoder_correct)

    @property
    def interceptor_successes(self) -> int:
        return sum(1 for r in self.rounds if r.interceptor_correct)

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
                    "encryptor_fallback": r.encryptor_fallback,
                    "interceptor_fallback": r.interceptor_fallback,
                    "decoder_fallback": r.decoder_fallback,
                }
                for r in self.rounds
            ],
        }

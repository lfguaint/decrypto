from __future__ import annotations
import itertools
import random

from openai import OpenAI

from config import ExperimentConfig
from models import KeywordPair, RoundRecord, GameResult
from agents import Encryptor, Interceptor, Decoder


def _all_codes(num_keywords: int, code_length: int) -> list[tuple]:
    """All permutations of [1..num_keywords] taken code_length at a time."""
    return list(itertools.permutations(range(1, num_keywords + 1), code_length))


def run_game(
    client: OpenAI,
    cfg: ExperimentConfig,
    keywords: list[KeywordPair],
    keyword_set_id: int,
    game_id: int,
    rng: random.Random,
) -> GameResult:
    codes = _all_codes(cfg.num_keywords, cfg.code_length)
    rng.shuffle(codes)
    codes = codes[: cfg.rounds_per_game]

    encryptor = Encryptor(client, cfg.agents, cfg.backend, keywords, cfg.code_length)
    interceptor = Interceptor(client, cfg.agents, cfg.backend, cfg.num_keywords, cfg.code_length)
    decoder = Decoder(client, cfg.agents, cfg.backend, keywords, cfg.code_length)

    result = GameResult(keyword_set_id=keyword_set_id, game_id=game_id, keywords=keywords)

    for round_num, code in enumerate(codes, start=1):
        history = result.rounds[:]

        print(f"    Round {round_num}/{cfg.rounds_per_game} | code={list(code)}", end="", flush=True)

        clues = encryptor.give_clues(code, history)
        print(f" | clues={list(clues)}", end="", flush=True)

        interceptor_guess = interceptor.guess(clues, history)
        decoder_guess = decoder.guess(clues, history)
        print(f" | interceptor={list(interceptor_guess)} decoder={list(decoder_guess)}", flush=True)

        record = RoundRecord(
            round_number=round_num,
            code=code,
            clues=clues,
            interceptor_guess=interceptor_guess,
            decoder_guess=decoder_guess,
        )
        result.rounds.append(record)

    return result

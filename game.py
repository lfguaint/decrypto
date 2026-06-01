from __future__ import annotations
import itertools
import random

from openai import OpenAI

from config import ExperimentConfig
from models import KeywordPair, RoundRecord, GameResult
from agents import Encryptor, Interceptor, Decoder


def _all_codes() -> list[tuple[int, int, int]]:
    """All 24 permutations of [1,2,3,4] taken 3 at a time (no repeats)."""
    return list(itertools.permutations(range(1, 5), 3))


def run_game(
    client: OpenAI,
    cfg: ExperimentConfig,
    keywords: list[KeywordPair],
    keyword_set_id: int,
    game_id: int,
    rng: random.Random,
) -> GameResult:
    codes = _all_codes()
    rng.shuffle(codes)
    codes = codes[: cfg.rounds_per_game]

    encryptor = Encryptor(client, cfg.agents, keywords)
    interceptor = Interceptor(client, cfg.agents)
    decoder = Decoder(client, cfg.agents, keywords)

    result = GameResult(keyword_set_id=keyword_set_id, game_id=game_id, keywords=keywords)

    for round_num, code in enumerate(codes, start=1):
        history = result.rounds[:]  # public history up to this round

        print(f"    Round {round_num}/{cfg.rounds_per_game} | code={code}", end="", flush=True)

        clues = encryptor.give_clues(code, history)
        print(f" | clues={clues}", end="", flush=True)

        interceptor_guess = interceptor.guess(clues, history)
        decoder_guess = decoder.guess(clues, history)
        print(f" | interceptor={interceptor_guess} decoder={decoder_guess}", flush=True)

        record = RoundRecord(
            round_number=round_num,
            code=code,
            clues=clues,
            interceptor_guess=interceptor_guess,
            decoder_guess=decoder_guess,
        )
        result.rounds.append(record)

    return result

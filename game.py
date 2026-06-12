from __future__ import annotations
import itertools
import time

from openai import OpenAI

from config import ExperimentConfig
from models import KeywordPair, RoundRecord, GameResult
from agents import Encryptor, Interceptor, Decoder


def all_codes(num_keywords: int, code_length: int) -> list[tuple]:
    """All permutations of [1..num_keywords] taken code_length at a time."""
    return list(itertools.permutations(range(1, num_keywords + 1), code_length))


def run_game(
    client: OpenAI,
    cfg: ExperimentConfig,
    keywords: list[KeywordPair],
    keyword_set_id: int,
    game_id: int,
    codes: list[tuple],
    verbose: bool = True,
) -> GameResult:
    """Play one game over the pre-drawn `codes` sequence (one code per round)."""
    encryptor = Encryptor(client, cfg.agents, cfg.backend, keywords, cfg.code_length)
    interceptor = Interceptor(client, cfg.agents, cfg.backend, cfg.num_keywords, cfg.code_length)
    decoder = Decoder(client, cfg.agents, cfg.backend, keywords, cfg.code_length)

    result = GameResult(
        keyword_set_id=keyword_set_id,
        game_id=game_id,
        keywords=keywords,
        prompt_version=cfg.agents.prompt_version,
    )
    game_start = time.time()

    for round_num, code in enumerate(codes, start=1):
        history = result.rounds[:]
        round_start = time.time()

        if verbose:
            print(f"    Round {round_num}/{len(codes)} | code={list(code)}", end="", flush=True)

        clues, enc_fb = encryptor.give_clues(code, history)
        if verbose:
            print(f" | clues={list(clues)}", end="", flush=True)

        interceptor_guess, int_fb = interceptor.guess(clues, history)
        decoder_guess, dec_fb = decoder.guess(clues, history)

        elapsed = time.time() - round_start
        if verbose:
            print(f" | interceptor={list(interceptor_guess)} decoder={list(decoder_guess)} | {elapsed:.1f}s", flush=True)

        record = RoundRecord(
            round_number=round_num,
            code=code,
            clues=clues,
            interceptor_guess=interceptor_guess,
            decoder_guess=decoder_guess,
            elapsed_seconds=elapsed,
            encryptor_fallback=enc_fb,
            interceptor_fallback=int_fb,
            decoder_fallback=dec_fb,
        )
        result.rounds.append(record)

    result.total_seconds = time.time() - game_start
    return result

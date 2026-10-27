from __future__ import annotations
import itertools
import time

from config import ExperimentConfig
from models import KeywordPair, RoundRecord, GameResult
from agents import Encryptor, Interceptor, Decoder, _make_client


def all_codes(num_keywords: int, code_length: int) -> list[tuple]:
    """All permutations of [1..num_keywords] taken code_length at a time."""
    return list(itertools.permutations(range(1, num_keywords + 1), code_length))


MAX_ATTEMPTS = 3


def _agent_backend(per_agent: str | None, default: str) -> str:
    return per_agent if per_agent is not None else default


class _StageFailed(Exception):
    pass


def _stage(agent: str, act, calls: list, verbose: bool):
    """Resend the identical request until the response is usable (up to MAX_ATTEMPTS)."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        value, call = act()
        call.attempt = attempt
        calls.append((agent, call))
        if value is not None:
            return value
        if verbose:
            print(f" [{agent} attempt {attempt}/{MAX_ATTEMPTS}: {call.status}: {call.error}]", end="", flush=True)
    raise _StageFailed(f"{agent} gave no usable response in {MAX_ATTEMPTS} attempts "
                       f"(last: {call.status}: {call.error})")


def run_game(
    cfg: ExperimentConfig,
    keywords: list[KeywordPair],
    keyword_set_id: int,
    game_id: int,
    codes: list[tuple],
    verbose: bool = True,
) -> GameResult:
    """Play one game over the pre-drawn `codes` sequence (one code per round)."""
    enc_backend = _agent_backend(cfg.agents.encryptor_backend, cfg.backend)
    dec_backend = _agent_backend(cfg.agents.decoder_backend, cfg.backend)
    int_backend = _agent_backend(cfg.agents.interceptor_backend, cfg.backend)

    encryptor = Encryptor(_make_client(enc_backend), cfg.agents, enc_backend, keywords, cfg.code_length)
    interceptor = Interceptor(_make_client(int_backend), cfg.agents, int_backend, cfg.num_keywords, cfg.code_length)
    decoder = Decoder(_make_client(dec_backend), cfg.agents, dec_backend, keywords, cfg.code_length)

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

        calls: list = []
        try:
            clues = _stage("encryptor", lambda: encryptor.give_clues(code, history), calls, verbose)
            if verbose:
                print(f" | clues={list(clues)}", end="", flush=True)
            interceptor_guess = _stage("interceptor", lambda: interceptor.guess(clues, history), calls, verbose)
            decoder_guess = _stage("decoder", lambda: decoder.guess(clues, history), calls, verbose)
        except _StageFailed as e:
            result.aborted, result.abort_reason = True, f"round {round_num}: {e}"
            result.pending_calls = calls
            if verbose:
                print(f" | ABORTED: {e}", flush=True)
            break

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
            calls=calls,
        )
        result.rounds.append(record)

    result.total_seconds = time.time() - game_start
    return result

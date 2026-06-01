from __future__ import annotations
import json
import random

from agents import _make_client
from config import ExperimentConfig
from keywords import generate_keyword_pairs
from game import run_game
from models import GameResult


def run_experiment(cfg: ExperimentConfig) -> list[GameResult]:
    rng = random.Random(cfg.seed)
    client = _make_client()
    all_results: list[GameResult] = []

    for ks_id in range(1, cfg.num_keyword_sets + 1):
        keywords = generate_keyword_pairs(rng)
        kw_str = ", ".join(f"{kp.number}:{kp.word}" for kp in keywords)
        print(f"\n=== Keyword Set {ks_id}/{cfg.num_keyword_sets}: [{kw_str}] ===")

        for g_id in range(1, cfg.games_per_keyword_set + 1):
            print(f"  --- Game {g_id}/{cfg.games_per_keyword_set} ---")
            result = run_game(client, cfg, keywords, ks_id, g_id, rng)
            all_results.append(result)
            s = result.summary()
            print(
                f"  => successful={s['successful_transmissions']}  "
                f"decoder_fail={s['decoder_failures']}  "
                f"intercepted={s['interceptor_successes']}"
            )

    return all_results


def save_results(results: list[GameResult], path: str) -> None:
    data = [r.summary() for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")

from __future__ import annotations
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import storage
from agents import _make_client
from config import ExperimentConfig
from keywords import generate_keyword_pairs
from game import run_game, all_codes
from models import GameResult


def _prepare_tasks(cfg: ExperimentConfig, rng: random.Random) -> list:
    """Pre-draw ALL randomness (keyword sets + code sequences) in the main
    thread, so workers are deterministic and the rng is never shared."""
    code_pool = all_codes(cfg.num_keywords, cfg.code_length)
    seen_sets: set = set()
    tasks = []
    for ks_id in range(1, cfg.num_keyword_sets + 1):
        keywords = generate_keyword_pairs(cfg.num_keywords, rng, seen_sets)
        for g_id in range(1, cfg.games_per_keyword_set + 1):
            codes = code_pool[:]
            rng.shuffle(codes)
            tasks.append((ks_id, g_id, keywords, codes[: cfg.rounds_per_game]))
    return tasks


def run_experiment(
    cfg: ExperimentConfig,
    db_path: str = "results/decrypto.db",
    workers: int = 1,
) -> list[GameResult]:
    rng = random.Random(cfg.seed)
    client = _make_client(cfg.backend)

    conn = storage.get_conn(db_path)
    experiment_id = storage.insert_experiment(conn, cfg)
    print(f"Experiment #{experiment_id} → {db_path}")

    tasks = _prepare_tasks(cfg, rng)
    total = len(tasks)
    verbose = workers <= 1
    print(f"{total} games | {workers} worker(s)\n")

    all_results: list[GameResult] = []
    completed = failed = 0
    t0 = time.time()

    # Workers only play games; ALL DB writes happen here in the main thread.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_game, client, cfg, kw, ks_id, g_id, codes, verbose):
                (ks_id, g_id)
            for ks_id, g_id, kw, codes in tasks
        }
        for fut in as_completed(futures):
            ks_id, g_id = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                failed += 1
                print(f"  [FAILED] set {ks_id} game {g_id}: {e}")
                continue
            storage.insert_game(conn, experiment_id, result)  # incremental persistence
            all_results.append(result)
            completed += 1
            s = result.summary()
            rate = (time.time() - t0) / completed
            eta_min = rate * (total - completed - failed) / 60
            print(
                f"  [{completed}/{total}] set {result.keyword_set_id} game {result.game_id} | "
                f"S={s['successful_transmissions']} I={s['interceptor_successes']} "
                f"F={s['decoder_failures']} | {result.total_seconds:.0f}s | ETA {eta_min:.0f}min"
            )

    conn.close()
    elapsed_min = (time.time() - t0) / 60
    print(f"\nDone: {completed} games saved, {failed} failed | {elapsed_min:.1f} min total")
    return all_results


def save_results(results: list[GameResult], path: str) -> None:
    data = [r.summary() for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {path}")

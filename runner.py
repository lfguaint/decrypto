from __future__ import annotations
import json
import os
import random
import signal
import sqlite3
import time
from collections import deque
from datetime import datetime, timedelta
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import prompts
import storage
from agents import _resolve_model
from config import Backend, ExperimentConfig, openrouter_routing
from keywords import generate_keyword_pairs, load_keyword_sets
from game import run_game, all_codes
from models import GameResult


def _prepare_tasks(cfg: ExperimentConfig) -> list:
    """Pre-draw ALL randomness (keyword sets + code sequences) in the main
    thread, so workers are deterministic and no rng is shared.

    Keywords and codes use separate streams: keyword set k depends only on
    (keyword_seed, k), and the codes of game (k, g) only on (seed, k, g), so
    changing rounds/games per set or the code-drawing scheme never alters the sets.
    With cfg.keyword_sets_file, the sets are read from that file instead of drawn."""
    seed = cfg.seed
    if seed is None:
        seed = random.SystemRandom().randrange(2**63)
    keyword_seed = cfg.keyword_seed if cfg.keyword_seed is not None else seed
    code_pool = all_codes(cfg.num_keywords, cfg.code_length)
    if cfg.keyword_sets_file:
        keyword_sets = load_keyword_sets(
            cfg.keyword_sets_file, cfg.num_keywords, cfg.num_keyword_sets, keyword_seed)
    else:
        kw_rng = random.Random(f"{keyword_seed}-keywords")
        seen_sets: set = set()
        keyword_sets = [generate_keyword_pairs(cfg.num_keywords, kw_rng, seen_sets)
                        for _ in range(cfg.num_keyword_sets)]
    tasks = []
    for ks_id, keywords in enumerate(keyword_sets, start=1):
        for g_id in range(1, cfg.games_per_keyword_set + 1):
            code_rng = random.Random(f"{seed}-codes-{ks_id}-{g_id}")
            codes = [code_rng.choice(code_pool) for _ in range(cfg.rounds_per_game)]
            tasks.append((ks_id, g_id, keywords, codes))
    return tasks


def _check_provider_pins(cfg: ExperimentConfig) -> None:
    """Fail before any API call if an OpenRouter agent has no pinned provider."""
    a = cfg.agents
    for model, backend in ((a.encryptor_model, a.encryptor_backend),
                           (a.decoder_model, a.decoder_backend),
                           (a.interceptor_model, a.interceptor_backend)):
        backend = backend if backend is not None else cfg.backend
        if backend == Backend.OPENROUTER:
            openrouter_routing(_resolve_model(model, backend))


def _cost_str(cost: float | None) -> str:
    return "cost n/a" if cost is None else f"${cost:.4f}"


def invalid_log_path(db_path: str) -> str:
    return os.path.splitext(db_path)[0] + "_invalid_responses.jsonl"


def _log_invalid(path: str, experiment_id: int, result: GameResult) -> int:
    """Append one JSON line per unusable attempt (format error or clue-rule violation)."""
    entries = result.invalid_calls
    if not entries:
        return 0
    with open(path, "a") as f:
        for round_number, agent, c in entries:
            f.write(json.dumps({
                "logged_at": datetime.now().isoformat(timespec="seconds"),
                "experiment_id": experiment_id,
                "keyword_set_id": result.keyword_set_id,
                "game_number": result.game_id,
                "round": round_number,
                "agent": agent,
                "attempt": c.attempt,
                "status": c.status,
                "error": c.error,
                "game_aborted": result.aborted,
                "model_requested": c.model_requested,
                "response_id": c.response_id,
                "content": c.content,
            }, ensure_ascii=False) + "\n")
    return len(entries)


PAUSE, CANCEL = "pause", "cancel"
MAX_CONSECUTIVE_API_FAILURES = 3


def control_files(db_path: str) -> tuple[str, str]:
    """Creating <db>.pause or <db>.cancel (e.g. from another terminal) pauses or cancels the queue."""
    base = os.path.splitext(db_path)[0]
    return base + ".pause", base + ".cancel"


def _play(conn, experiment_id: int, cfg: ExperimentConfig, tasks: list, workers: int,
          db_path: str) -> list[GameResult]:
    """Keep up to `workers` games running; when one finishes, the next queued game starts.
    All status changes and DB writes happen here, in the main thread.

    Pause  (Ctrl+C once, or <db>.pause): start no new game, finish and save the running ones.
    Cancel (Ctrl+C twice, <db>.cancel, or MAX_CONSECUTIVE_API_FAILURES API failures in a row):
           stop at once; the interrupted games go back to the queue.
    Either way, resume later with --retry-experiment."""
    pause_file, cancel_file = control_files(db_path)
    for f in (pause_file, cancel_file):
        if os.path.exists(f):
            os.remove(f)
    stop = {"request": None}

    def on_sigint(signum, frame):
        stop["request"] = CANCEL if stop["request"] else PAUSE
        if stop["request"] == PAUSE:
            print("\n  PAUSE requested: no new games; waiting for the running ones (Ctrl+C again to cancel now)",
                  flush=True)

    total = len(tasks)
    verbose = workers <= 1
    log_path = invalid_log_path(db_path)
    pending = deque(tasks)
    running: dict = {}
    all_results: list[GameResult] = []
    completed = failed = aborted = invalid = consecutive_failures = 0
    spent = 0.0
    t0 = time.time()
    previous_handler = signal.signal(signal.SIGINT, on_sigint)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        while pending or running:
            if os.path.exists(cancel_file):
                stop["request"] = CANCEL
            elif os.path.exists(pause_file) and stop["request"] is None:
                stop["request"] = PAUSE
                print("\n  PAUSE requested by file: no new games; waiting for the running ones", flush=True)
            if stop["request"] == CANCEL:
                break
            while pending and len(running) < workers and stop["request"] is None:
                ks_id, g_id, kw, codes = pending.popleft()
                storage.set_status(conn, experiment_id, ks_id, g_id, storage.RUNNING)
                running[pool.submit(run_game, cfg, kw, ks_id, g_id, codes, verbose)] = (ks_id, g_id)
            if not running:
                break
            done, _ = wait(running, timeout=1.0, return_when=FIRST_COMPLETED)
            for fut in done:
                ks_id, g_id = running.pop(fut)
                try:
                    result = fut.result()
                except Exception as e:
                    failed += 1
                    consecutive_failures += 1
                    storage.set_status(conn, experiment_id, ks_id, g_id, storage.ERROR,
                                       f"API: {type(e).__name__}: {e}")
                    print(f"  [ERROR] set {ks_id} game {g_id}: {type(e).__name__}: {e}", flush=True)
                    if consecutive_failures >= MAX_CONSECUTIVE_API_FAILURES:
                        stop["request"] = CANCEL
                        print(f"  {consecutive_failures} API failures in a row: cancelling the queue "
                              "(systemic error?)", flush=True)
                    continue
                consecutive_failures = 0
                invalid += _log_invalid(log_path, experiment_id, result)
                spent += result.total_cost or 0.0
                if result.aborted:
                    aborted += 1
                    storage.set_status(conn, experiment_id, ks_id, g_id, storage.ERROR, result.abort_reason)
                    print(f"  [ERROR] set {ks_id} game {g_id}: {result.abort_reason}", flush=True)
                    continue
                storage.insert_game(conn, experiment_id, result)  # also marks the game done
                all_results.append(result)
                completed += 1
                s = result.summary()
                rate = (time.time() - t0) / completed
                eta_min = rate * (len(pending) + len(running)) / 60 if stop["request"] is None else 0
                print(
                    f"  [{completed}/{total}] set {result.keyword_set_id} game {result.game_id} | "
                    f"S={s['successful_transmissions']} I={s['interceptor_successes']} "
                    f"F={s['decoder_failures']} | {result.total_seconds:.0f}s | "
                    f"{_cost_str(result.total_cost)} | ETA {eta_min:.0f}min", flush=True
                )
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        if running:
            # Cancelled (or crashed): threads cannot be stopped, so requeue their games and exit now.
            for ks_id, g_id in running.values():
                storage.set_status(conn, experiment_id, ks_id, g_id, storage.QUEUED)
            conn.close()
            print(f"\nCANCELLED: {completed} games saved this run; {len(running)} interrupted games and "
                  f"{len(pending)} not started remain queued. Resume with --retry-experiment {experiment_id}.",
                  flush=True)
            os._exit(130)
        pool.shutdown(wait=True)
    elapsed_min = (time.time() - t0) / 60
    state = {CANCEL: "CANCELLED", PAUSE: "PAUSED"}.get(stop["request"], "Done") if pending else "Done"
    print(f"\n{state}: {completed} games saved, {aborted + failed} errors ({aborted} invalid responses, "
          f"{failed} API) | {len(pending)} still queued | {elapsed_min:.1f} min | reported cost ${spent:.4f}")
    if invalid:
        print(f"{invalid} invalid responses logged to {log_path}")
    if pending or aborted or failed:
        print(f"Resume (queued and errored games) with --retry-experiment {experiment_id}")
    return all_results


def _all_keys(cfg: ExperimentConfig) -> list[tuple[int, int]]:
    return [(ks, g) for ks in range(1, cfg.num_keyword_sets + 1)
            for g in range(1, cfg.games_per_keyword_set + 1)]


def _select(tasks: list, keys: list[tuple[int, int]], limit: int | None) -> list:
    """The tasks of the first `limit` keys (queue order)."""
    keep = set(keys[:limit] if limit is not None else keys)
    return [t for t in tasks if (t[0], t[1]) in keep]


def run_experiment(
    cfg: ExperimentConfig,
    db_path: str = "results/decrypto.db",
    workers: int = 1,
    limit: int | None = None,
) -> list[GameResult]:
    """Create the experiment, queue all its games at once, and play them (or the first `limit`)."""
    _check_provider_pins(cfg)
    conn = storage.get_conn(db_path)
    experiment_id = storage.insert_experiment(conn, cfg)
    keys = _all_keys(cfg)
    storage.enqueue_games(conn, experiment_id, keys)
    print(f"Experiment #{experiment_id} → {db_path}")
    tasks = _select(_prepare_tasks(cfg), keys, limit)
    print(f"{len(keys)} games queued, playing {len(tasks)} | {workers} worker(s)\n")
    results = _play(conn, experiment_id, cfg, tasks, workers, db_path)
    conn.close()
    return results


def retry_missing_games(
    experiment_id: int,
    db_path: str = "results/decrypto.db",
    workers: int = 1,
    limit: int | None = None,
) -> list[GameResult]:
    """Resume an experiment: requeue its errored and interrupted games, then play the queue
    (all, or the first `limit`). The stored seeds and keyword-set file regenerate the same games."""
    conn = storage.get_conn(db_path)
    cfg = storage.load_experiment(conn, experiment_id)
    _check_provider_pins(cfg)
    stored = storage.stored_prompt_sha256(conn, experiment_id)
    if stored is not None and stored != prompts.prompts_sha256():
        conn.close()
        raise RuntimeError(
            f"Experiment #{experiment_id} was run with different prompts (sha256 {stored[:12]}…); "
            "continuing it would mix prompt texts within one configuration."
        )
    storage.ensure_status_rows(conn, experiment_id, _all_keys(cfg))
    requeued = storage.requeue_unfinished(conn, experiment_id)
    queued = storage.queued_games(conn, experiment_id)
    if not queued:
        print(f"Experiment #{experiment_id} is already complete — nothing to play.")
        conn.close()
        return []
    tasks = _select(_prepare_tasks(cfg), queued, limit)
    print(f"Experiment #{experiment_id}: {len(queued)} games queued ({requeued} requeued after error or "
          f"interruption), playing {len(tasks)} | {workers} worker(s)")
    results = _play(conn, experiment_id, cfg, tasks, workers, db_path)
    conn.close()
    return results


def show_status(experiment_id: int, db_path: str) -> None:
    """Print the queue state of an experiment; read-only, safe while it is running."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cfg = storage.load_experiment(conn, experiment_id)
    s = storage.status_summary(conn, experiment_id)
    total = cfg.num_keyword_sets * cfg.games_per_keyword_set
    if not s:
        done = conn.execute("SELECT COUNT(*) FROM games WHERE experiment_id = ?", (experiment_id,)).fetchone()[0]
        print(f"Experiment #{experiment_id}: no queue table (older run); {done} of {total} games saved.")
        conn.close()
        return
    c = s["counts"]
    print(f"Experiment #{experiment_id} ({db_path}): {total} games | "
          f"done {c.get(storage.DONE, 0)} | running {c.get(storage.RUNNING, 0)} | "
          f"queued {c.get(storage.QUEUED, 0)} | error {c.get(storage.ERROR, 0)}")
    n_done, mean_s, max_s = conn.execute(
        "SELECT COUNT(*), AVG(total_seconds), MAX(total_seconds) FROM games WHERE experiment_id = ?",
        (experiment_id,)).fetchone()
    hm = lambda t: t.strftime("%H:%M")
    latest = None
    for ks, g, started in s["running"]:
        if n_done:
            t0 = datetime.fromisoformat(started)
            expected, worst = t0 + timedelta(seconds=mean_s), t0 + timedelta(seconds=max_s)
            latest = max(latest or worst, worst)
            print(f"  running  set {ks} game {g} since {hm(t0)} | expected end ~{hm(expected)} "
                  f"(at the latest ~{hm(worst)})")
        else:
            print(f"  running  set {ks} game {g} since {started}")
    if s["running"]:
        if latest:
            print(f"  All running games should end by ~{hm(latest)} (from {n_done} finished games: "
                  f"mean {mean_s / 60:.1f} min, longest {max_s / 60:.1f} min)")
        else:
            print("  No finished game yet to estimate end times.")
    for ks, g, starts, err in s["errors"]:
        print(f"  error    set {ks} game {g} (started {starts}x): {err}")
    conn.close()


def save_results(results: list[GameResult], path: str) -> None:
    data = [r.summary() for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {path}")

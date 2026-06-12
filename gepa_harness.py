"""GEPA harness: prompt optimization for the Encryptor.

Three pieces:
  1. extract_lessons()  — mines local game JSONs for failure patterns (ASI)
  2. build_objective()  — the GEPA objective with lessons embedded
  3. make_evaluator()   — runs a batch of games with a candidate prompt
                          template and returns the successful-transmission rate

Offline preparation (no API key needed):
    python3 gepa_harness.py                 # extract lessons, show readiness

Run optimization (requires OPENROUTER_API_KEY and `pip install gepa`):
    python3 gepa_harness.py --run
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import random
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace

from config import AgentConfig, ExperimentConfig, Backend
from game import run_game
from runner import _prepare_tasks

# Train seed — MUST differ from the benchmark seed so optimization keyword
# sets are disjoint from evaluation ones.
TRAIN_SEED = 1337

LESSONS_PATH = "results/gepa_lessons.md"


# ---------------------------------------------------------------------------
# Seed candidate: prompt v2 as a placeholder template
# ---------------------------------------------------------------------------

SEED_TEMPLATE = """\
You are the Encryptor in a word-clue game.

Your team has {num_keywords} secret keyword cards:
{keywords_block}

Each round you receive a {code_length}-digit code — a sequence of {code_length} distinct digits \
from 1 to {num_keywords} (e.g. {example_code}).
Your job is to give exactly {code_length} one-word clues — one clue per digit, in order.

You have two competing objectives:
1. GUIDE your Decoder teammate: the clues must be clear enough for the Decoder — who knows \
the keyword cards — to recover the exact code.
2. MISLEAD the Interceptor: the Interceptor does NOT know the keywords but observes every \
clue and its corresponding revealed code. Each clue you give is a data point the Interceptor \
can use to map clues to keywords. After just a few rounds, obvious clues will expose the \
full keyword mapping and the Interceptor will intercept every subsequent code.

Strategic guidelines:
- Avoid direct synonyms or category labels.
- Prefer oblique, thematic, or abstract associations that your Decoder can still resolve \
given the keyword context, but that an uninformed observer would not immediately link to \
the correct keyword.
- Vary your clue style across rounds to prevent the Interceptor from building a reliable \
association map.
- Remember: the Decoder has a significant advantage — they know the keywords. Even a subtle \
or indirect clue is enough for them. Sacrifice directness for secrecy.

Rules for clues:
- Each clue must be a single word (no numbers, no parts of the keyword itself).
- The clue must relate to the keyword it encodes, not to its position.

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{json_format}
"""


# ---------------------------------------------------------------------------
# 1. Lessons extraction from local games
# ---------------------------------------------------------------------------

def extract_lessons(results_dir: str = "results") -> str:
    paths = sorted(glob.glob(os.path.join(results_dir, "results*.json")))
    games = []
    for p in paths:
        try:
            data = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue
        for g in data:
            if "keywords" in g and g.get("rounds"):
                games.append((os.path.basename(p), g))

    kw_clues: dict = {}   # keyword -> list[(clue, round_intercepted)]
    n_rounds = n_int = n_decfail = 0
    first_interceptions = []

    for _, g in games:
        kws = g["keywords"]
        first_int = None
        for r in g["rounds"]:
            n_rounds += 1
            if r["interceptor_correct"]:
                n_int += 1
                if first_int is None:
                    first_int = r["round"]
            if not r["decoder_correct"]:
                n_decfail += 1
            for i, clue in enumerate(r["clues"]):
                kw = kws[str(r["code"][i])]
                kw_clues.setdefault(kw, []).append((clue, bool(r["interceptor_correct"])))
        if first_int is not None:
            first_interceptions.append(first_int)

    lines = [
        "# Lessons from local pilot games (gpt-oss:20b, qwen3:8b)",
        "",
        f"- Games analyzed: {len(games)} | rounds: {n_rounds}",
        f"- Interceptions: {n_int}/{n_rounds} ({100*n_int/max(n_rounds,1):.0f}%) | "
        f"decoder failures: {n_decfail}/{n_rounds} ({100*n_decfail/max(n_rounds,1):.0f}%)",
    ]
    if first_interceptions:
        lines.append(
            f"- First interception round (median): "
            f"{statistics.median(first_interceptions):.0f} "
            f"(range {min(first_interceptions)}–{max(first_interceptions)})"
        )
    lines += ["", "## Clues used per keyword (× = round was intercepted)", ""]
    for kw in sorted(kw_clues):
        entries = ", ".join(f"{c}×" if hit else c for c, hit in kw_clues[kw])
        lines.append(f"- **{kw}**: {entries}")

    lines += [
        "",
        "## Curated failure modes",
        "",
        "1. CATEGORICAL CASCADE — clues that are co-hyponyms or attributes of the keyword's "
        "obvious category (plow/harvester/combine/mower for 'tractor') let the Interceptor "
        "map the full keyword set by round 2 and intercept every round afterwards.",
        "2. PERSISTENCE — once the Interceptor cracks a mapping it never misses again; "
        "the prompt must PREVENT early leakage, not recover from it.",
        "3. OBLIQUE WINS — indirect associations (caboose, silo, ottoman, aurora) delayed "
        "interception without hurting the Decoder, who knows the keywords.",
        "4. OBLIQUITY LIMIT — too-creative clues (yoke for 'tractor', blizzard for 'tractor') "
        "confuse the Decoder itself; the sweet spot is oblique-but-resolvable given the keyword.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Objective
# ---------------------------------------------------------------------------

def build_objective(lessons: str) -> str:
    return f"""\
Optimize the system prompt TEMPLATE for the Encryptor agent in a three-player word-clue game.

GAME: the Encryptor and a Decoder share {{num_keywords}} secret keywords (numbered). Each \
round the Encryptor gets a secret code (distinct digits) and emits one single-word clue per \
digit. An adversarial Interceptor sees every clue and every revealed code from past rounds, \
but never the keywords. The Decoder and Interceptor then each guess the code.

METRIC (maximize): successful-transmission rate = fraction of rounds where the Decoder \
guesses the exact code AND the Interceptor does not.

HARD CONSTRAINTS on the template:
- It must keep the literal placeholders {{num_keywords}}, {{keywords_block}}, \
{{code_length}}, {{example_code}}, {{json_format}} — they are substituted at runtime.
- The agent's reply must remain ONLY a JSON object in the {{json_format}} format; keep an \
explicit instruction to that effect, or parsing fails and the round is wasted.
- Clues must remain single words that relate to the keyword meaning (game rule).

KNOWN FAILURE MODES AND EVIDENCE (from pilot games — use these to guide mutations):

{lessons}
"""


# ---------------------------------------------------------------------------
# 3. Evaluator
# ---------------------------------------------------------------------------

def make_evaluator(
    client,
    base_cfg: ExperimentConfig,
    eval_sets: int = 6,
    workers: int = 6,
    log=print,
):
    """Returns evaluate(candidate_template) -> successful-transmission rate.

    Keyword sets and codes are drawn ONCE with TRAIN_SEED, so every candidate
    is scored on the identical batch (paired comparison, lower variance)."""
    eval_cfg = replace(base_cfg, num_keyword_sets=eval_sets, games_per_keyword_set=1)
    tasks = _prepare_tasks(eval_cfg, random.Random(TRAIN_SEED))

    def evaluate(candidate: str) -> float:
        cfg = replace(
            eval_cfg,
            agents=replace(eval_cfg.agents, encryptor_system_template=candidate),
        )
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [
                pool.submit(run_game, client, cfg, kw, ks, g, codes, False)
                for ks, g, kw, codes in tasks
            ]
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as e:
                    log(f"[evaluator] game failed: {e}")

        rounds = [r for res in results for r in res.rounds]
        if not rounds:
            return 0.0
        n = len(rounds)
        s = sum(r.transmission_successful for r in rounds)
        i = sum(r.interceptor_correct for r in rounds)
        f = sum(not r.decoder_correct for r in rounds)
        fb = sum(r.encryptor_fallback for r in rounds)

        log(f"[evaluator] S={s}/{n} I={i}/{n} F={f}/{n} enc_fallbacks={fb}")
        # Diagnostics for reflection: every non-successful round with its mapping
        for res in results:
            kws = {kp.number: kp.word for kp in res.keywords}
            for r in res.rounds:
                if not r.transmission_successful:
                    mapping = ", ".join(
                        f"'{c}'→{kws[d]}" for c, d in zip(r.clues, r.code))
                    why = "INTERCEPTED" if r.interceptor_correct else "DECODER MISSED"
                    log(f"[evaluator] round {r.round_number} {why}: {mapping}")
        return s / n

    return evaluate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="GEPA harness for Encryptor prompt optimization")
    ap.add_argument("--run", action="store_true", help="Run GEPA optimization (needs API key + gepa)")
    ap.add_argument("--task-model", default="deepseek-flash")
    ap.add_argument("--eval-sets", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-metric-calls", type=int, default=80)
    args = ap.parse_args()

    lessons = extract_lessons()
    os.makedirs(os.path.dirname(LESSONS_PATH), exist_ok=True)
    with open(LESSONS_PATH, "w") as f:
        f.write(lessons)
    print(f"Lessons written to {LESSONS_PATH}\n")
    print(lessons)

    if not args.run:
        print("\n--- Readiness ---")
        print(f"Seed template:   OK ({len(SEED_TEMPLATE)} chars, v2-based)")
        print(f"Objective:       OK ({len(build_objective(lessons))} chars)")
        print(f"API key:         {'OK' if os.environ.get('OPENROUTER_API_KEY') else 'MISSING (export OPENROUTER_API_KEY)'}")
        try:
            import gepa  # noqa: F401
            print("gepa package:    OK")
        except ImportError:
            print("gepa package:    MISSING (pip install gepa)")
        print("\nRun with --run to start optimization.")
        return

    from agents import _make_client
    import gepa.optimize_anything as oa
    from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig

    base_cfg = ExperimentConfig(
        backend=Backend.OPENROUTER,
        agents=AgentConfig(
            encryptor_model=args.task_model,
            decoder_model=args.task_model,
            interceptor_model=args.task_model,
        ),
        seed=TRAIN_SEED,
    )
    client = _make_client(Backend.OPENROUTER)
    evaluate = make_evaluator(client, base_cfg, args.eval_sets, args.workers, log=oa.log)

    result = optimize_anything(
        seed_candidate=SEED_TEMPLATE,
        evaluator=evaluate,
        objective=build_objective(lessons),
        config=GEPAConfig(engine=EngineConfig(max_metric_calls=args.max_metric_calls)),
    )
    out = "results/gepa_best_prompt.txt"
    with open(out, "w") as f:
        f.write(result.best_candidate)
    print(f"\nBest prompt saved to {out}")


if __name__ == "__main__":
    main()

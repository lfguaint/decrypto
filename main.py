import argparse

from config import (
    AgentConfig, ExperimentConfig,
    Backend, OLLAMA_MODELS, OPENROUTER_MODELS,
)
from runner import run_experiment, retry_missing_games, save_results, show_status


def parse_args():
    ollama_list = "\n  ".join(f"{k}: {v}" for k, v in OLLAMA_MODELS.items())
    openrouter_list = "\n  ".join(f"{k}: {v}" for k, v in OPENROUTER_MODELS.items())

    parser = argparse.ArgumentParser(
        description="Decrypto LLM benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Ollama model shortcuts:\n  {ollama_list}\n\n"
            f"OpenRouter model shortcuts:\n  {openrouter_list}"
        ),
    )
    parser.add_argument(
        "--backend", choices=[Backend.OLLAMA, Backend.OPENROUTER, Backend.CLUSTER],
        default=Backend.OLLAMA,
        help="Default inference backend (default: ollama)",
    )
    parser.add_argument("--num-keywords", type=int, default=4,
                        help="Number of keyword cards (default: 4)")
    parser.add_argument("--code-length", type=int, default=3,
                        help="Number of digits per code (default: 3; must be < num-keywords)")
    parser.add_argument("--keyword-sets", type=int, default=3)
    parser.add_argument("--games-per-set", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5,
                        help="Rounds per game (default: 5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Required for new experiments: 3 = benchmark, 42 = pilots/tests")
    parser.add_argument("--keyword-seed", type=int, default=None,
                        help="Seed for the keyword sets (default: same as --seed; benchmark uses 3)")
    parser.add_argument("--keyword-sets-file", type=str, default=None,
                        help="Read keyword sets from this file instead of drawing them "
                             "(benchmark: data/keyword_sets_seed3.json)")
    _backend_choices = [Backend.OLLAMA, Backend.OPENROUTER, Backend.CLUSTER]
    parser.add_argument(
        "--encryptor-model", type=str, default="gpt-oss-20b",
        help="Model shortcut or full model ID",
    )
    parser.add_argument(
        "--encryptor-backend", choices=_backend_choices, default=None,
        help="Backend for Encryptor (default: --backend)",
    )
    parser.add_argument(
        "--decoder-model", type=str, default="gpt-oss-20b",
        help="Model shortcut or full model ID",
    )
    parser.add_argument(
        "--decoder-backend", choices=_backend_choices, default=None,
        help="Backend for Decoder (default: --backend)",
    )
    parser.add_argument(
        "--interceptor-model", type=str, default="gpt-oss-20b",
        help="Model shortcut or full model ID",
    )
    parser.add_argument(
        "--interceptor-backend", choices=_backend_choices, default=None,
        help="Backend for Interceptor (default: --backend)",
    )
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature for all agents (default: provider default, parameter omitted)")
    parser.add_argument("--encryptor-temperature", type=float, default=None,
                        help="Temperature for Encryptor (overrides --temperature)")
    parser.add_argument("--decoder-temperature", type=float, default=None,
                        help="Temperature for Decoder (overrides --temperature)")
    parser.add_argument("--interceptor-temperature", type=float, default=None,
                        help="Temperature for Interceptor (overrides --temperature)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max output tokens per call (default: no limit)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent games (1 = sequential with per-round logs)")
    parser.add_argument("--db", type=str, default="results/decrypto.db",
                        help="SQLite database path (default: results/decrypto.db)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON export path (DB is always written)")
    parser.add_argument("--retry-experiment", type=int, default=None, metavar="ID",
                        help="Resume experiment ID: requeue its errored/interrupted games and play the queue")
    parser.add_argument("--status", type=int, default=None, metavar="ID",
                        help="Show the queue of experiment ID (done/running/queued/error) and exit")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Play only the first N games of the queue; the rest stay queued. "
                             "Pause: Ctrl+C once or create <db>_exp<ID>.pause; "
                             "cancel: Ctrl+C twice or <db>_exp<ID>.cancel")
    args = parser.parse_args()
    if args.status is None and args.retry_experiment is None and args.seed is None:
        parser.error("--seed is required (3 = benchmark, 42 = pilots/tests)")
    if args.retry_experiment is not None and (
            args.seed is not None or args.keyword_seed is not None or args.keyword_sets_file is not None):
        parser.error("--seed/--keyword-seed/--keyword-sets-file are not used with --retry-experiment: "
                     "the experiment's stored values are reused")

    agents = AgentConfig(
        encryptor_model=args.encryptor_model,
        decoder_model=args.decoder_model,
        interceptor_model=args.interceptor_model,
        encryptor_backend=args.encryptor_backend,
        decoder_backend=args.decoder_backend,
        interceptor_backend=args.interceptor_backend,
        encryptor_temperature=(args.encryptor_temperature
                               if args.encryptor_temperature is not None else args.temperature),
        decoder_temperature=(args.decoder_temperature
                             if args.decoder_temperature is not None else args.temperature),
        interceptor_temperature=(args.interceptor_temperature
                                 if args.interceptor_temperature is not None else args.temperature),
        max_tokens=args.max_tokens,
    )
    cfg = ExperimentConfig(
        backend=args.backend,
        num_keywords=args.num_keywords,
        code_length=args.code_length,
        num_keyword_sets=args.keyword_sets,
        games_per_keyword_set=args.games_per_set,
        rounds_per_game=args.rounds,
        seed=args.seed,
        keyword_seed=args.keyword_seed,
        keyword_sets_file=args.keyword_sets_file,
        agents=agents,
    )
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return cfg, args.db, args.output, args.workers, args.retry_experiment, args.limit, args.status


def main():
    cfg, db_path, output_path, workers, retry_id, limit, status_id = parse_args()

    if status_id is not None:
        show_status(status_id, db_path)
        return
    if retry_id is not None:
        retry_missing_games(retry_id, db_path, workers, limit)
        return
    def _t(v):
        return "default" if v is None else v

    def _b(per_agent, default):
        return per_agent if per_agent is not None else default

    enc_b = _b(cfg.agents.encryptor_backend, cfg.backend)
    dec_b = _b(cfg.agents.decoder_backend, cfg.backend)
    int_b = _b(cfg.agents.interceptor_backend, cfg.backend)

    print("Decrypto benchmark starting...")
    print(f"  Default backend: {cfg.backend}")
    print(f"  Keywords:        {cfg.num_keywords}  (codes: 1–{cfg.num_keywords})")
    print(f"  Code length:     {cfg.code_length}  (possible codes: {cfg.num_possible_codes})")
    print(f"  Keyword sets:    {cfg.num_keyword_sets}")
    print(f"  Games per set:   {cfg.games_per_keyword_set}")
    print(f"  Rounds per game: {cfg.rounds_per_game}")
    print(f"  Seeds:           codes={cfg.seed}  keywords={cfg.keyword_seed}")
    print(f"  Sets source:     {cfg.keyword_sets_file or 'drawn from keyword seed'}")
    print(f"  Temperature:     enc={_t(cfg.agents.encryptor_temperature)}  dec={_t(cfg.agents.decoder_temperature)}  int={_t(cfg.agents.interceptor_temperature)}")
    print(f"  Max tokens:      {cfg.agents.max_tokens or 'no limit'}")
    print(f"  Prompt version:  {cfg.agents.prompt_version}")
    print(f"  Encryptor:       {cfg.agents.encryptor_model}  [{enc_b}]")
    print(f"  Decoder:         {cfg.agents.decoder_model}  [{dec_b}]")
    print(f"  Interceptor:     {cfg.agents.interceptor_model}  [{int_b}]")

    results = run_experiment(cfg, db_path, workers, limit)
    if output_path:
        save_results(results, output_path)

    total_rounds = sum(len(r.rounds) for r in results)
    total_success = sum(r.successful_transmissions for r in results)
    total_dec_fail = sum(r.decoder_failures for r in results)
    total_intercept = sum(r.interceptor_successes for r in results)

    print("\n=== Aggregate Results ===")
    print(f"  Total rounds:             {total_rounds}")
    print(f"  Successful transmissions: {total_success} ({100*total_success/total_rounds:.1f}%)")
    print(f"  Decoder failures:         {total_dec_fail} ({100*total_dec_fail/total_rounds:.1f}%)")
    print(f"  Interceptor successes:    {total_intercept} ({100*total_intercept/total_rounds:.1f}%)")


if __name__ == "__main__":
    main()

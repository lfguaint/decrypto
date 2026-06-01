import argparse

from config import AgentConfig, ExperimentConfig, MODELS
from runner import run_experiment, save_results


def parse_args():
    model_list = "\n  ".join(f"{k}: {v}" for k, v in MODELS.items())
    parser = argparse.ArgumentParser(
        description="Decrypto 3-agent LLM simulation (via OpenRouter)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Recommended model shortcuts:\n  {model_list}",
    )
    parser.add_argument("--keyword-sets", type=int, default=3, help="Number of keyword sets")
    parser.add_argument("--games-per-set", type=int, default=2, help="Games per keyword set")
    parser.add_argument("--rounds", type=int, default=8, help="Rounds per game (max 24)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--encryptor-model", type=str, default="deepseek-v3",
        help="OpenRouter model ID or shortcut (e.g. 'grok-4', 'gpt-5-mini')",
    )
    parser.add_argument(
        "--decoder-model", type=str, default="deepseek-v3",
        help="OpenRouter model ID or shortcut",
    )
    parser.add_argument(
        "--interceptor-model", type=str, default="deepseek-v3",
        help="OpenRouter model ID or shortcut",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=str, default="results.json", help="Output JSON file")
    args = parser.parse_args()

    # Resolve shortcut names → full OpenRouter ID
    def resolve(model: str) -> str:
        return MODELS.get(model, model)

    agents = AgentConfig(
        encryptor_model=resolve(args.encryptor_model),
        decoder_model=resolve(args.decoder_model),
        interceptor_model=resolve(args.interceptor_model),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    cfg = ExperimentConfig(
        num_keyword_sets=args.keyword_sets,
        games_per_keyword_set=args.games_per_set,
        rounds_per_game=min(args.rounds, 24),
        seed=args.seed,
        agents=agents,
    )
    return cfg, args.output


def main():
    cfg, output_path = parse_args()
    print("Decrypto simulation starting...")
    print(f"  Keyword sets:    {cfg.num_keyword_sets}")
    print(f"  Games per set:   {cfg.games_per_keyword_set}")
    print(f"  Rounds per game: {cfg.rounds_per_game}")
    print(f"  Temperature:     {cfg.agents.temperature}")
    print(f"  Encryptor:   {cfg.agents.encryptor_model}")
    print(f"  Decoder:     {cfg.agents.decoder_model}")
    print(f"  Interceptor: {cfg.agents.interceptor_model}")

    results = run_experiment(cfg)
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

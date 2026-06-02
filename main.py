import argparse

from config import (
    AgentConfig, ExperimentConfig,
    Backend, OLLAMA_MODELS, OPENROUTER_MODELS,
)
from runner import run_experiment, save_results


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
        "--backend", choices=[Backend.OLLAMA, Backend.OPENROUTER],
        default=Backend.OLLAMA,
        help="Inference backend (default: ollama)",
    )
    parser.add_argument("--num-keywords", type=int, default=4,
                        help="Number of keyword cards (default: 4)")
    parser.add_argument("--code-length", type=int, default=3,
                        help="Number of digits per code (default: 3; must be < num-keywords)")
    parser.add_argument("--keyword-sets", type=int, default=3)
    parser.add_argument("--games-per-set", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5,
                        help="Rounds per game (must be < P(num-keywords, code-length))")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--encryptor-model", type=str, default="gpt-oss-20b",
        help="Model shortcut or full model ID",
    )
    parser.add_argument(
        "--decoder-model", type=str, default="gpt-oss-20b",
        help="Model shortcut or full model ID",
    )
    parser.add_argument(
        "--interceptor-model", type=str, default="gpt-oss-20b",
        help="Model shortcut or full model ID",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--prompt-version", type=int, default=1, choices=[1, 2],
                        help="Encryptor prompt version: 1=baseline, 2=strategic (default: 1)")
    parser.add_argument("--output", type=str, default="results.json")
    args = parser.parse_args()

    agents = AgentConfig(
        encryptor_model=args.encryptor_model,
        decoder_model=args.decoder_model,
        interceptor_model=args.interceptor_model,
        temperature=args.temperature,
        prompt_version=args.prompt_version,
    )
    cfg = ExperimentConfig(
        backend=args.backend,
        num_keywords=args.num_keywords,
        code_length=args.code_length,
        num_keyword_sets=args.keyword_sets,
        games_per_keyword_set=args.games_per_set,
        rounds_per_game=args.rounds,
        seed=args.seed,
        agents=agents,
    )
    return cfg, args.output


def main():
    cfg, output_path = parse_args()
    print("Decrypto benchmark starting...")
    print(f"  Backend:         {cfg.backend}")
    print(f"  Keywords:        {cfg.num_keywords}  (codes: 1–{cfg.num_keywords})")
    print(f"  Code length:     {cfg.code_length}  (possible codes: {cfg.num_possible_codes})")
    print(f"  Keyword sets:    {cfg.num_keyword_sets}")
    print(f"  Games per set:   {cfg.games_per_keyword_set}")
    print(f"  Rounds per game: {cfg.rounds_per_game}")
    print(f"  Temperature:     {cfg.agents.temperature}")
    print(f"  Prompt version:  {cfg.agents.prompt_version}")
    print(f"  Encryptor:       {cfg.agents.encryptor_model}")
    print(f"  Decoder:         {cfg.agents.decoder_model}")
    print(f"  Interceptor:     {cfg.agents.interceptor_model}")

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

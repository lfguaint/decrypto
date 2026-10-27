"""Quick connectivity test for all configured backends.

Usage:
    python3 test_clients.py                  # testa backends disponíveis
    python3 test_clients.py --game           # roda 1 jogo de 1 rodada com os backends testados
    python3 test_clients.py --game --enc-backend cluster --dec-backend openrouter --int-backend ollama
"""
from __future__ import annotations
import argparse
import os
import sys
import time

from agents import _make_client, _resolve_model
from config import Backend, BACKEND_SETTINGS, OLLAMA_MODELS, OPENROUTER_MODELS, DGX_MODELS


PING_PROMPT = "Reply with exactly one word: OK"

# Fast/cheap model to use per backend for connectivity checks
DEFAULT_MODELS = {
    Backend.OLLAMA:     "gpt-oss:20b",
    Backend.OPENROUTER: "deepseek/deepseek-v4-flash",
    Backend.CLUSTER:    "gemma-4-31b",   # fast DGX model
}


def ping_backend(backend: str, model: str) -> tuple[bool, str]:
    """Send a single minimal request. Returns (ok, message)."""
    try:
        client = _make_client(backend)
        t0 = time.time()
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": PING_PROMPT}],
            max_tokens=8,
        )
        resp = client.chat.completions.create(**kwargs)
        content = (resp.choices[0].message.content or "").strip()
        elapsed = time.time() - t0
        return True, f"{content!r}  ({elapsed:.1f}s)"
    except Exception as e:
        return False, str(e)


def check_prerequisites(backend: str) -> tuple[bool, str]:
    """Check env vars / reachability before attempting a call."""
    if backend == Backend.OPENROUTER:
        if not os.environ.get("OPENROUTER_API_KEY"):
            return False, "OPENROUTER_API_KEY not set"
    if backend == Backend.CLUSTER:
        if not os.environ.get("DGX_API_KEY"):
            return False, "DGX_API_KEY not set"
        url = BACKEND_SETTINGS[Backend.CLUSTER]["base_url"]
        return True, f"url={url}"
    return True, ""


def main():
    ap = argparse.ArgumentParser(description="Backend connectivity test")
    ap.add_argument("--backends", nargs="+",
                    choices=[Backend.OLLAMA, Backend.OPENROUTER, Backend.CLUSTER],
                    default=[Backend.OLLAMA, Backend.OPENROUTER, Backend.CLUSTER],
                    help="Backends to test (default: all three)")
    ap.add_argument("--game", action="store_true",
                    help="After connectivity check, run a 1-round game with mixed backends")
    ap.add_argument("--enc-backend", choices=[Backend.OLLAMA, Backend.OPENROUTER, Backend.CLUSTER],
                    default=None, help="Encryptor backend for --game")
    ap.add_argument("--dec-backend", choices=[Backend.OLLAMA, Backend.OPENROUTER, Backend.CLUSTER],
                    default=None, help="Decoder backend for --game")
    ap.add_argument("--int-backend", choices=[Backend.OLLAMA, Backend.OPENROUTER, Backend.CLUSTER],
                    default=None, help="Interceptor backend for --game")
    args = ap.parse_args()

    # ── Connectivity check ──────────────────────────────────────────────────
    print("=== Backend connectivity ===\n")
    reachable = {}
    for backend in args.backends:
        ok_pre, pre_msg = check_prerequisites(backend)
        if not ok_pre:
            print(f"  {backend:12s}  SKIP  {pre_msg}")
            reachable[backend] = False
            continue
        if pre_msg:
            print(f"  {backend:12s}  checking...  ({pre_msg})", flush=True)
        else:
            print(f"  {backend:12s}  checking...", end="", flush=True)
        model = DEFAULT_MODELS[backend]
        ok, msg = ping_backend(backend, model)
        status = "OK  " if ok else "FAIL"
        print(f"\r  {backend:12s}  {status}  {msg}")
        reachable[backend] = ok

    if not args.game:
        return

    # ── Functional game test ─────────────────────────────────────────────────
    print("\n=== 1-round game with per-agent backends ===\n")

    # Pick a reachable backend for each role (fallback: first reachable)
    first_ok = next((b for b, ok in reachable.items() if ok), None)
    if first_ok is None:
        print("No reachable backend — cannot run game test.")
        sys.exit(1)

    enc_b = args.enc_backend or first_ok
    dec_b = args.dec_backend or first_ok
    int_b = args.int_backend or first_ok

    for role, backend in [("Encryptor", enc_b), ("Decoder", dec_b), ("Interceptor", int_b)]:
        if not reachable.get(backend, False):
            print(f"  {role} backend '{backend}' is not reachable — aborting game test.")
            sys.exit(1)

    enc_model = DEFAULT_MODELS[enc_b]
    dec_model = DEFAULT_MODELS[dec_b]
    int_model = DEFAULT_MODELS[int_b]

    print(f"  Encryptor:   {enc_model}  [{enc_b}]")
    print(f"  Decoder:     {dec_model}  [{dec_b}]")
    print(f"  Interceptor: {int_model}  [{int_b}]\n")

    from config import AgentConfig, ExperimentConfig
    from game import run_game
    import random

    cfg = ExperimentConfig(
        backend=first_ok,
        num_keyword_sets=1,
        games_per_keyword_set=1,
        rounds_per_game=1,
        agents=AgentConfig(
            encryptor_model=enc_model,
            encryptor_backend=enc_b,
            decoder_model=dec_model,
            decoder_backend=dec_b,
            interceptor_model=int_model,
            interceptor_backend=int_b,
        ),
        seed=42,
    )

    from keywords import generate_keyword_pairs
    from game import all_codes
    rng = random.Random(42)
    keywords = generate_keyword_pairs(cfg.num_keywords, rng, set())
    codes = all_codes(cfg.num_keywords, cfg.code_length)

    try:
        result = run_game(cfg, keywords, keyword_set_id=1, game_id=1,
                          codes=[rng.choice(codes)], verbose=True)
        r = result.rounds[0]
        print(f"\n  code={list(r.code)}  clues={list(r.clues)}")
        print(f"  decoder={list(r.decoder_guess)} ({'✓' if r.decoder_correct else '✗'})  "
              f"interceptor={list(r.interceptor_guess)} ({'✓' if r.interceptor_correct else '✗'})")
        print(f"  Transmission: {'SUCCESS' if r.transmission_successful else 'FAIL'}")
    except Exception as e:
        print(f"\n  Game failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

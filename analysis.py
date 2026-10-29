from __future__ import annotations
import argparse
import json
import math
import random
import sqlite3
from collections import Counter

PERMUTATIONS = 10_000
PERMUTATION_SEED = 0


# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _fmt(k: int, n: int) -> str:
    if n == 0:
        return "       —"
    p = k / n
    lo, hi = wilson(k, n)
    return f"{100*p:5.1f}% [{100*lo:.1f}, {100*hi:.1f}]"


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def list_experiments(conn: sqlite3.Connection) -> list:
    return conn.execute("""
        SELECT e.id, e.created_at, e.encryptor_model, e.decoder_model,
               e.interceptor_model, e.prompt_version,
               COUNT(DISTINCT g.id) AS games
        FROM experiments e LEFT JOIN games g ON g.experiment_id = e.id
        GROUP BY e.id ORDER BY e.id
    """).fetchall()


def chance_rate(conn: sqlite3.Connection, experiment_id: int) -> float:
    """1 / number of possible codes: the hit rate of any guess independent of the code."""
    n, k = conn.execute("SELECT num_keywords, code_length FROM experiments WHERE id = ?",
                        (experiment_id,)).fetchone()
    return 1 / math.perm(n, k)


def per_round_outcomes(conn: sqlite3.Connection, experiment_id: int) -> list:
    """Per round: (round, n, S, I with decoder right, I with decoder wrong, F)."""
    return conn.execute("""
        SELECT r.round_number, COUNT(*),
               SUM(r.decoder_correct AND NOT r.interceptor_correct),
               SUM(r.interceptor_correct AND r.decoder_correct),
               SUM(r.interceptor_correct AND NOT r.decoder_correct),
               SUM(NOT r.decoder_correct AND NOT r.interceptor_correct)
        FROM rounds r JOIN games g ON r.game_id = g.id
        WHERE g.experiment_id = ?
        GROUP BY r.round_number ORDER BY r.round_number
    """, (experiment_id,)).fetchall()


def guesses_by_round(conn: sqlite3.Connection, experiment_id: int, who: str) -> dict:
    """round -> list of (code, guess) as tuples; `who` is 'interceptor' or 'decoder'."""
    col = {"interceptor": "r.interceptor_guess", "decoder": "r.decoder_guess"}[who]
    out: dict = {}
    for rnd, code, guess in conn.execute(f"""
            SELECT r.round_number, r.code, {col} FROM rounds r JOIN games g ON r.game_id = g.id
            WHERE g.experiment_id = ? ORDER BY g.keyword_set_id, g.game_number""", (experiment_id,)):
        out.setdefault(rnd, []).append((tuple(json.loads(code)), tuple(json.loads(guess))))
    return out


def permutation_p(pairs: list, rng: random.Random, permutations: int = PERMUTATIONS) -> float:
    """One-sided p that the hits are this high if guesses ignored the code: shuffle the
    guesses across games (keeping the drawn codes and the model's guessing habits)."""
    codes = [c for c, _ in pairs]
    guesses = [g for _, g in pairs]
    observed = sum(c == g for c, g in pairs)
    at_least = 0
    for _ in range(permutations):
        rng.shuffle(guesses)
        if sum(c == g for c, g in zip(codes, guesses)) >= observed:
            at_least += 1
    return (at_least + 1) / (permutations + 1)


def survival_no_interception(conn: sqlite3.Connection, experiment_id: int) -> list:
    """P(no interception through round t): (round, games_alive, total_games)."""
    rows = conn.execute("""
        SELECT g.id, r.round_number, r.interceptor_correct
        FROM rounds r JOIN games g ON r.game_id = g.id
        WHERE g.experiment_id = ?
        ORDER BY g.id, r.round_number
    """, (experiment_id,)).fetchall()

    games: dict = {}
    max_round = 0
    for game_id, rnd, intercepted in rows:
        games.setdefault(game_id, {})[rnd] = intercepted
        max_round = max(max_round, rnd)

    n = len(games)
    out = []
    for t in range(1, max_round + 1):
        alive = sum(
            1 for rounds in games.values()
            if all(rounds.get(u, 0) == 0 for u in range(1, t + 1))
        )
        out.append((t, alive, n))
    return out


def interceptor_win_rounds(
    conn: sqlite3.Connection,
    experiment_id: int,
    count_round1_interception: bool,
    horizon: int,
    tokens_to_win: int = 2,
) -> list:
    """Post-hoc 3-player win rule, scored like S/I/F: an interception gives the
    Interceptor one token and ends the round (the decoder's answer is not scored);
    otherwise a decoder miss gives one token. Per game: round the Interceptor
    reached `tokens_to_win` within `horizon`, or None if the Encryptor team held."""
    rows = conn.execute("""
        SELECT g.id, r.round_number, r.interceptor_correct, r.decoder_correct
        FROM rounds r JOIN games g ON r.game_id = g.id
        WHERE g.experiment_id = ? AND r.round_number <= ?
        ORDER BY g.id, r.round_number
    """, (experiment_id, horizon)).fetchall()

    decided: dict = {}
    tokens: dict = {}
    for game_id, rnd, intercepted, decoded in rows:
        decided.setdefault(game_id, None)
        if decided[game_id] is not None:
            continue
        # The rulebook forbids interception in round 1, so there the decoder's answer is what counts.
        if intercepted and (rnd > 1 or count_round1_interception):
            gained = 1
        else:
            gained = 1 - decoded
        tokens[game_id] = tokens.get(game_id, 0) + gained
        if tokens[game_id] >= tokens_to_win:
            decided[game_id] = rnd
    return list(decided.values())


def interceptor_points(
    conn: sqlite3.Connection,
    experiment_id: int,
    count_round1_interception: bool,
    horizon: int,
) -> list:
    """Interceptor points per game over rounds 1..horizon, scored like interceptor_win_rounds:
    an interception (I) or a decoder miss in a round not intercepted (F) is one point, S is zero.
    The Interceptor wins a game with 2 or more points."""
    rows = conn.execute("""
        SELECT g.id, r.round_number, r.interceptor_correct, r.decoder_correct
        FROM rounds r JOIN games g ON r.game_id = g.id
        WHERE g.experiment_id = ? AND r.round_number <= ?
        ORDER BY g.id, r.round_number
    """, (experiment_id, horizon)).fetchall()
    points: dict = {}
    for game_id, rnd, intercepted, decoded in rows:
        counted = intercepted and (rnd > 1 or count_round1_interception)
        points[game_id] = points.get(game_id, 0) + (1 if counted else 1 - decoded)
    return list(points.values())


def t_quantile_975(df: int) -> float:
    """97.5% quantile of Student's t (Cornish-Fisher expansion around z; error < 1e-3 for df >= 10)."""
    z = 1.959964
    return (z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * df ** 3))


def mean_ci(values: list) -> tuple:
    """95% t interval of the mean over games (the independent unit): mean ± t * sd / sqrt(n)."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, mean
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = t_quantile_975(n - 1) * sd / math.sqrt(n)
    return mean - half, mean + half


def chance_win_rate(chance: float, scored_rounds: int, tokens_to_win: int = 2) -> float:
    """P(Interceptor wins) if it guesses blindly and the decoder never misses: Binomial tail."""
    return sum(math.comb(scored_rounds, k) * chance ** k * (1 - chance) ** (scored_rounds - k)
               for k in range(tokens_to_win, scored_rounds + 1))


def call_stats(conn: sqlite3.Connection, experiment_id: int) -> list:
    """Per agent: calls, invalid attempts (format / rule), mean tokens and seconds, cost, providers."""
    return conn.execute("""
        SELECT k.agent, COUNT(*),
               SUM(k.status = 'format_error'), SUM(k.status = 'rule_violation'),
               AVG(k.prompt_tokens), AVG(k.completion_tokens), AVG(k.reasoning_tokens),
               AVG(k.elapsed_seconds), SUM(k.cost), GROUP_CONCAT(DISTINCT k.provider)
        FROM calls k JOIN rounds r ON k.round_id = r.id JOIN games g ON r.game_id = g.id
        WHERE g.experiment_id = ?
        GROUP BY k.agent ORDER BY CASE k.agent WHEN 'encryptor' THEN 1 WHEN 'interceptor' THEN 2 ELSE 3 END
    """, (experiment_id,)).fetchall()


def legacy_fallback_rates(conn: sqlite3.Connection, experiment_id: int) -> tuple:
    """Older runs replaced invalid responses with random ones; newer runs never do."""
    return conn.execute("""
        SELECT COUNT(*), SUM(r.encryptor_fallback), SUM(r.interceptor_fallback), SUM(r.decoder_fallback)
        FROM rounds r JOIN games g ON r.game_id = g.id WHERE g.experiment_id = ?
    """, (experiment_id,)).fetchone()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(db_path: str, experiment_id: int | None) -> None:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)  # safe while a run is writing

    experiments = list_experiments(conn)
    if not experiments:
        print("No experiments found.")
        return

    if experiment_id is None:
        print("Experiments in database:\n")
        print(f"{'id':>4}  {'created_at':<20} {'enc/dec/int models':<46} {'pv':>2} {'games':>5}")
        for (eid, ts, enc, dec, icp, pv, games) in experiments:
            models = enc if enc == dec == icp else f"{enc} / {dec} / {icp}"
            print(f"{eid:>4}  {ts:<20} {models:<46} {pv:>2} {games:>5}")
        print("\nRun with --experiment <id> for the full report.")
        return

    enc, dec, icp = conn.execute(
        "SELECT encryptor_model, decoder_model, interceptor_model FROM experiments WHERE id = ?",
        (experiment_id,)).fetchone()
    n_games, mean_s = conn.execute(
        "SELECT COUNT(*), AVG(total_seconds) FROM games WHERE experiment_id = ?", (experiment_id,)).fetchone()
    print(f"=== Experiment {experiment_id}: Enc {enc} | Dec {dec} | Int {icp} | {n_games} games ===")
    if _has_table(conn, "game_status"):
        status = dict(conn.execute("SELECT status, COUNT(*) FROM game_status WHERE experiment_id = ? "
                                   "GROUP BY status", (experiment_id,)))
        restarted = conn.execute("SELECT COUNT(*) FROM game_status WHERE experiment_id = ? AND starts > 1",
                                 (experiment_id,)).fetchone()[0]
        print(f"Queue: {status} | games started more than once: {restarted}")
    print()

    print("Per-round outcomes — Wilson 95% CI  (I_S: intercepted, decoder right; I_F: intercepted, decoder wrong)")
    print(f"{'round':>5} {'n':>5}  {'S':>22}  {'I_S':>22}  {'I_F':>22}  {'F':>22}")
    # No pooled "all rounds" row: the 5 rounds of a game are correlated, so a Wilson CI
    # over rounds treated as independent would be too narrow.
    for rnd, n, s, i_s, i_f, f in per_round_outcomes(conn, experiment_id):
        print(f"{rnd:>5} {n:>5}  {_fmt(s, n):>22}  {_fmt(i_s, n):>22}  {_fmt(i_f, n):>22}  {_fmt(f, n):>22}")

    chance = chance_rate(conn, experiment_id)
    rng = random.Random(PERMUTATION_SEED)
    print(f"\nInterceptor vs chance (any guess independent of the code hits {100*chance:.2f}%)")
    print(f"permutation test: {PERMUTATIONS} shuffles of the guesses within each round, one-sided")
    print(f"{'round':>5} {'n':>5}  {'hit rate':>22}  {'excess':>8}  {'p':>7}  most common guess")
    for rnd, pairs in sorted(guesses_by_round(conn, experiment_id, "interceptor").items()):
        k = sum(c == g for c, g in pairs)
        top, top_n = Counter(g for _, g in pairs).most_common(1)[0]
        print(f"{rnd:>5} {len(pairs):>5}  {_fmt(k, len(pairs)):>22}  {100*(k/len(pairs) - chance):+7.1f}pp"
              f"  {permutation_p(pairs, rng):7.4f}  {'-'.join(map(str, top))} ({100*top_n/len(pairs):.0f}%)")

    print("\nSurvival — P(no interception through round t)")
    print(f"{'round':>5}  {'alive':>5}  {'rate':>22}")
    for t, alive, n in survival_no_interception(conn, experiment_id):
        print(f"{t:>5}  {alive:>5}  {_fmt(alive, n):>22}")

    max_round = conn.execute("""
        SELECT MAX(r.round_number) FROM rounds r JOIN games g ON r.game_id = g.id
        WHERE g.experiment_id = ?
    """, (experiment_id,)).fetchone()[0] or 0
    horizon = min(5, max_round)
    variants = (("R1 interception counts", True), ("rulebook: no R1 intercept", False))
    print(f"\nWin condition, post hoc (I or F = 1 point to the Interceptor, S = 0; "
          f"Interceptor wins with 2+ points in {horizon} rounds)")
    print(f"{'variant':<28} {'games':>9}  {'Encryptor team wins':>22}  {'Interceptor wins':>22}  {'chance':>6}"
          f"  decided by round 1..{horizon}")
    for label, count_r1 in variants:
        decided = interceptor_win_rounds(conn, experiment_id, count_r1, horizon)
        n = len(decided)
        wins = sum(1 for d in decided if d is not None)
        by_round = [sum(1 for d in decided if d is not None and d <= t) for t in range(1, horizon + 1)]
        p0 = chance_win_rate(chance, horizon if count_r1 else horizon - 1)
        print(f"{label:<28} {f'{n - wins}:{wins}':>9}  {_fmt(n - wins, n):>22}  {_fmt(wins, n):>22}"
              f"  {100*p0:5.1f}%  {by_round}")
    print("chance: Interceptor guessing blindly against a decoder that never misses")

    print(f"\nInterceptor points per game (0..{horizon}); mean CI: Student's t over games")
    print(f"{'variant':<28} {'mean':>5} {'95% CI':>14} {'sd':>5} {'median':>6} {'chance':>6}"
          f"  games with 0..{horizon} points")
    for label, count_r1 in variants:
        points = interceptor_points(conn, experiment_id, count_r1, horizon)
        if not points:
            continue
        n = len(points)
        mean = sum(points) / n
        sd = math.sqrt(sum((p - mean) ** 2 for p in points) / (n - 1)) if n > 1 else 0.0
        lo, hi = mean_ci(points)
        dist = [points.count(k) for k in range(horizon + 1)]
        chance_mean = chance * (horizon if count_r1 else horizon - 1)
        print(f"{label:<28} {mean:5.2f} [{lo:4.2f}, {hi:4.2f}] {sd:5.2f} {sorted(points)[n // 2]:>6}"
              f" {chance_mean:6.2f}  {dist}")

    if _has_table(conn, "calls"):
        rows = call_stats(conn, experiment_id)
        print("\nCalls per agent (invalid = unusable attempt, resent; rate over all attempts)")
        print(f"{'agent':<12} {'calls':>6}  {'format error':>22}  {'rule violation':>22}"
              f"  {'in tok':>6} {'out tok':>7} {'reason':>7} {'sec':>6} {'cost $':>8}  provider")
        total_cost = 0.0
        for agent, calls, fmt_err, rule, tin, tout, treason, secs, cost, prov in rows:
            total_cost += cost or 0
            print(f"{agent:<12} {calls:>6}  {_fmt(fmt_err or 0, calls):>22}  {_fmt(rule or 0, calls):>22}"
                  f"  {tin or 0:6.0f} {tout or 0:7.0f} {treason or 0:7.0f} {secs or 0:6.1f} {cost or 0:8.4f}  {prov}")
        if n_games:
            print(f"Total cost ${total_cost:.4f} (${total_cost / n_games:.4f} per game) | "
                  f"mean game duration {mean_s / 60:.1f} min")

    n, enc_fb, int_fb, dec_fb = legacy_fallback_rates(conn, experiment_id)
    if (enc_fb or 0) + (int_fb or 0) + (dec_fb or 0):
        print(f"\nLegacy random fallbacks over {n} rounds: "
              f"Enc {_fmt(enc_fb or 0, n)} | Int {_fmt(int_fb or 0, n)} | Dec {_fmt(dec_fb or 0, n)}")

    conn.close()


def main():
    ap = argparse.ArgumentParser(description="Decrypto benchmark analysis (Wilson 95% CIs)")
    ap.add_argument("--db", default="results/decrypto.db")
    ap.add_argument("--experiment", type=int, default=None,
                    help="Experiment id (omit to list all)")
    args = ap.parse_args()
    report(args.db, args.experiment)


if __name__ == "__main__":
    main()

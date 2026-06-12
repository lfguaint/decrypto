from __future__ import annotations
import argparse
import math
import sqlite3


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


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def _fallback_filter(exclude_fallbacks: bool) -> str:
    if exclude_fallbacks:
        return ("AND r.encryptor_fallback = 0 "
                "AND r.interceptor_fallback = 0 "
                "AND r.decoder_fallback = 0")
    return ""


def list_experiments(conn: sqlite3.Connection) -> list:
    return conn.execute("""
        SELECT e.id, e.created_at, e.encryptor_model, e.decoder_model,
               e.interceptor_model, e.prompt_version,
               COUNT(DISTINCT g.id) AS games
        FROM experiments e LEFT JOIN games g ON g.experiment_id = e.id
        GROUP BY e.id ORDER BY e.id
    """).fetchall()


def per_round_outcomes(
    conn: sqlite3.Connection, experiment_id: int, exclude_fallbacks: bool = False
) -> list:
    """Per round: (round, n, successes, interceptions, decoder_failures)."""
    return conn.execute(f"""
        SELECT r.round_number,
               COUNT(*) AS n,
               SUM(r.successful) AS s,
               SUM(r.interceptor_correct) AS i,
               SUM(1 - r.decoder_correct) AS f
        FROM rounds r
        JOIN games g ON r.game_id = g.id
        WHERE g.experiment_id = ? {_fallback_filter(exclude_fallbacks)}
        GROUP BY r.round_number ORDER BY r.round_number
    """, (experiment_id,)).fetchall()


def survival_no_interception(
    conn: sqlite3.Connection, experiment_id: int
) -> list:
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


def fallback_rates(conn: sqlite3.Connection, experiment_id: int) -> tuple:
    row = conn.execute("""
        SELECT COUNT(*),
               SUM(r.encryptor_fallback),
               SUM(r.interceptor_fallback),
               SUM(r.decoder_fallback)
        FROM rounds r JOIN games g ON r.game_id = g.id
        WHERE g.experiment_id = ?
    """, (experiment_id,)).fetchone()
    return row


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(db_path: str, experiment_id: int | None, exclude_fallbacks: bool) -> None:
    conn = sqlite3.connect(db_path)

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

    suffix = " (excluding fallback rounds)" if exclude_fallbacks else ""

    print(f"=== Experiment {experiment_id}{suffix} ===\n")

    print("Per-round outcome rates — Wilson 95% CI")
    print(f"{'round':>5} {'n':>5}  {'Success':>22}  {'Intercepted':>22}  {'Decoder fail':>22}")
    for rnd, n, s, i, f in per_round_outcomes(conn, experiment_id, exclude_fallbacks):
        print(f"{rnd:>5} {n:>5}  {_fmt(s, n):>22}  {_fmt(i, n):>22}  {_fmt(f, n):>22}")

    print("\nSurvival — P(no interception through round t)")
    print(f"{'round':>5}  {'alive':>5}  {'rate':>22}")
    for t, alive, n in survival_no_interception(conn, experiment_id):
        print(f"{t:>5}  {alive:>5}  {_fmt(alive, n):>22}")

    n, enc_fb, int_fb, dec_fb = fallback_rates(conn, experiment_id)
    print(f"\nFallback rates over {n} rounds: "
          f"Enc {_fmt(enc_fb or 0, n)} | Int {_fmt(int_fb or 0, n)} | Dec {_fmt(dec_fb or 0, n)}")

    conn.close()


def main():
    ap = argparse.ArgumentParser(description="Decrypto benchmark analysis (Wilson 95% CIs)")
    ap.add_argument("--db", default="results/decrypto.db")
    ap.add_argument("--experiment", type=int, default=None,
                    help="Experiment id (omit to list all)")
    ap.add_argument("--exclude-fallbacks", action="store_true",
                    help="Drop rounds where any agent fell back to a random response")
    args = ap.parse_args()
    report(args.db, args.experiment, args.exclude_fallbacks)


if __name__ == "__main__":
    main()

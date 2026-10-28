"""Per-round figures for the article, side by side for one or more experiments.

Usage: python3 plots.py --db results/benchmark_seed3.db --experiments 1 2 [--name NAME] [--width PT]

Three panels over rounds: interception rate, survival (no interception through round t)
and decoder accuracy, each with Wilson 95% bands and the chance line (1 / possible codes).
Writes figures/NAME.pdf (vector, for LaTeX) and figures/NAME.png. Read-only on the database.
"""
from __future__ import annotations
import argparse
import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from analysis import chance_rate, per_round_outcomes, survival_no_interception, wilson

# Shortcut (as stored in the database) -> name printed in the figures.
DISPLAY_NAMES = {
    "gemma": "Gemma 4 31B",
    "dgx-gemma": "Gemma 4 31B",
    "gpt-oss-120b": "gpt-oss-120b",
    "qwen3-32b": "Qwen3 32B",
    "deepseek-flash": "DeepSeek V4 Flash",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gpt-6-luna": "GPT-6 Luna",
    "claude-haiku": "Claude Haiku 4.5",
    "deepseek-pro": "DeepSeek V4 Pro",
    "kimi-k3": "Kimi K3",
    "gpt-6.1-sol": "GPT-6.1 Sol",
}

# Series are told apart by color, line style and marker, so the figure survives grayscale print.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "D", "s", "^"]
CHANCE_COLOR = "#898781"


def _name(model: str) -> str:
    return DISPLAY_NAMES.get(model, model)


def experiment_label(conn: sqlite3.Connection, experiment_id: int) -> str:
    enc, dec, icp = conn.execute(
        "SELECT encryptor_model, decoder_model, interceptor_model FROM experiments WHERE id = ?",
        (experiment_id,)).fetchone()
    n = conn.execute("SELECT COUNT(*) FROM games WHERE experiment_id = ?", (experiment_id,)).fetchone()[0]
    if enc == dec == icp:
        return f"{_name(enc)} (n = {n})"
    return f"Enc {_name(enc)} / Dec {_name(dec)} / Int {_name(icp)} (n = {n})"


def series(conn: sqlite3.Connection, experiment_id: int) -> dict:
    """panel -> (rounds, rate, lo, hi), all rates as fractions."""
    def triple(pairs):
        rounds = [t for t, _, _ in pairs]
        rate = [k / n for _, k, n in pairs]
        bounds = [wilson(k, n) for _, k, n in pairs]
        return rounds, rate, [b[0] for b in bounds], [b[1] for b in bounds]

    outcomes = per_round_outcomes(conn, experiment_id)
    return {
        "interception": triple([(r, i_s + i_f, n) for r, n, s, i_s, i_f, f in outcomes]),
        "survival": triple([(t, alive, n) for t, alive, n in survival_no_interception(conn, experiment_id)]),
        "decoder": triple([(r, s + i_s, n) for r, n, s, i_s, i_f, f in outcomes]),
    }


PANELS = [
    ("interception", "Interception rate"),
    ("survival", "Survival (no interception yet)"),
    ("decoder", "Decoder accuracy"),
]


def chance_curve(key: str, chance: float, rounds: list) -> list:
    """What guesses independent of the code give: a hit rate of `chance` every round, so
    survival under pure guessing is (1 - chance)^t rather than a flat line."""
    if key == "survival":
        return [(1 - chance) ** t for t in rounds]
    return [chance for _ in rounds]


def plot(db_path: str, experiment_ids: list, name: str, width_pt: float, out_dir: str) -> list:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    chances = {chance_rate(conn, e) for e in experiment_ids}
    if len(chances) != 1:
        raise SystemExit("experiments have different numbers of possible codes; plot them separately")
    chance = chances.pop()

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True, "font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    })
    width_in = width_pt / 72.27
    fig, axes = plt.subplots(1, len(PANELS), figsize=(width_in, width_in * 0.42), sharey=True)

    handles = []
    for idx, e in enumerate(experiment_ids):
        data = series(conn, e)
        color = COLORS[idx % len(COLORS)]
        for ax, (key, _) in zip(axes, PANELS):
            rounds, rate, lo, hi = data[key]
            ax.fill_between(rounds, lo, hi, color=color, alpha=0.18, linewidth=0)
            line, = ax.plot(rounds, rate, color=color, linestyle=LINESTYLES[idx % len(LINESTYLES)],
                            marker=MARKERS[idx % len(MARKERS)], markersize=3, linewidth=1.2)
        line.set_label(experiment_label(conn, e))
        handles.append(line)

    for ax, (key, title) in zip(axes, PANELS):
        rounds = list(range(1, 6))
        chance_line, = ax.plot(rounds, chance_curve(key, chance, rounds), color=CHANCE_COLOR,
                               linestyle=(0, (1, 2)), linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.set_xticks(range(1, 6))
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.5)
        ax.set_axisbelow(True)
    chance_line.set_label(f"Chance (1/{round(1 / chance)})")
    handles.append(chance_line)

    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.02), handlelength=2.4)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Decrypto per-round figures (Wilson 95% bands)")
    ap.add_argument("--db", default="results/benchmark_seed3.db")
    ap.add_argument("--experiments", type=int, nargs="+", required=True, help="experiment ids, plotted in this order")
    ap.add_argument("--name", default=None, help="output file name without extension (default: rounds_exp<ids>)")
    ap.add_argument("--width", type=float, default=345.0, help="figure width in pt (article \\textwidth: ~345)")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    name = args.name or "rounds_exp" + "_".join(map(str, args.experiments))
    for path in plot(args.db, args.experiments, name, args.width, args.out_dir):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

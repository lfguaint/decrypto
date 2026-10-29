"""Per-round figures for the article, side by side for one or more experiments.

Usage: python3 plots.py --db results/benchmark_seed3.db --experiments 1 2 [--name NAME] [--width PT]

Two figures over rounds, each panel with Wilson 95% bands:
  NAME_outcomes          success S, interception I and failure F (the round outcomes)
  NAME_decoder_survival  decoder accuracy and survival (no interception through round t)
  NAME_score             Interceptor points per game (I or F = 1, S = 0) and the winner
                         (Interceptor with 2+ points, else the Encryptor team)
Interception and decoder panels show the chance line (1 / possible codes); survival shows
(1 - chance)^t; the winner panel shows what a blind Interceptor would get against a decoder
that never misses. Writes figures/*.pdf (vector, for LaTeX) and *.png. Read-only on the database.
"""
from __future__ import annotations
import argparse
import math
import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from analysis import chance_rate, interceptor_points, per_round_outcomes, survival_no_interception, wilson

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
        "success": triple([(r, s, n) for r, n, s, i_s, i_f, f in outcomes]),
        "interception": triple([(r, i_s + i_f, n) for r, n, s, i_s, i_f, f in outcomes]),
        "failure": triple([(r, f, n) for r, n, s, i_s, i_f, f in outcomes]),
        "decoder": triple([(r, s + i_s, n) for r, n, s, i_s, i_f, f in outcomes]),
        "survival": triple([(t, alive, n) for t, alive, n in survival_no_interception(conn, experiment_id)]),
    }


# Figure suffix -> panels (series key, title). The round outcomes S / I / F come first.
FIGURES = {
    "outcomes": [
        ("success", "Success (S)"),
        ("interception", "Interception (I)"),
        ("failure", "Failure (F)"),
    ],
    "decoder_survival": [
        ("decoder", "Decoder accuracy"),
        ("survival", "Survival (no interception yet)"),
    ],
}


def chance_curve(key: str, chance: float, rounds: list) -> list | None:
    """What guesses independent of the code give: a hit rate of `chance` every round, so
    survival under pure guessing is (1 - chance)^t (exponential, nearly straight over 5
    rounds since chance is small). S and F mix both guessers and have no single chance line."""
    if key == "survival":
        return [(1 - chance) ** t for t in rounds]
    if key in ("interception", "decoder"):
        return [chance for _ in rounds]
    return None


ROUNDS = 5
HATCHES = ["", "////", "....", "xxxx"]


def plot(db_path: str, experiment_ids: list, name: str, width_pt: float, out_dir: str) -> list:
    _style()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    chances = {chance_rate(conn, e) for e in experiment_ids}
    if len(chances) != 1:
        raise SystemExit("experiments have different numbers of possible codes; plot them separately")
    chance = chances.pop()
    paths = []
    for suffix, panels in FIGURES.items():
        paths += _plot_figure(conn, experiment_ids, f"{name}_{suffix}", panels, chance, width_pt, out_dir)
    paths += _plot_score(conn, experiment_ids, f"{name}_score", chance, width_pt, out_dir)
    conn.close()
    return paths


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True, "font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    })


def _save(fig, name: str, out_dir: str) -> list:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _plot_score(conn: sqlite3.Connection, experiment_ids: list, name: str, chance: float,
                width_pt: float, out_dir: str) -> list:
    """Histogram of Interceptor points per game (I or F = 1, S = 0; R1 counts). Bars at 0-1
    points (the Encryptor team wins) are lighter than at 2+ (the Interceptor wins), and each
    side is labelled with the games each experiment won there."""
    width_in = width_pt / 72.27
    fig, ax = plt.subplots(figsize=(width_in, width_in * 0.48))
    k = len(experiment_ids)
    bar_w = 0.8 / k
    ax.axvspan(-0.5, 1.5, color="#f0efec", linewidth=0, zorder=0)
    ax.axvline(1.5, color="#c3c2b7", linewidth=0.8, zorder=1)
    ax.text(0.5, 0.99, "Encryptor team wins (0-1)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.5)
    ax.text(3.5, 0.99, "Interceptor wins (2+)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.5)

    handles = []
    for idx, e in enumerate(experiment_ids):
        points = interceptor_points(conn, e, True, ROUNDS)
        n = len(points)
        color = COLORS[idx % len(COLORS)]
        counts = [points.count(p) for p in range(ROUNDS + 1)]
        xs = [p - 0.4 + bar_w * (idx + 0.5) for p in range(ROUNDS + 1)]
        rates = [c / n for c in counts]
        bounds = [wilson(c, n) for c in counts]
        bars = ax.bar(xs, rates, width=bar_w, color=color, hatch=HATCHES[idx % len(HATCHES)],
                      edgecolor="white", linewidth=0, zorder=2)
        for p, bar in enumerate(bars):
            bar.set_alpha(0.45 if p < 2 else 0.9)
        ax.errorbar(xs, rates, yerr=[[r - b[0] for r, b in zip(rates, bounds)],
                                     [b[1] - r for r, b in zip(rates, bounds)]],
                    fmt="none", ecolor="#52514e", elinewidth=0.6, capsize=1.5, zorder=3)
        # Legend swatch at full opacity: the bars themselves mix two shades.
        handles.append(Patch(facecolor=color, hatch=HATCHES[idx % len(HATCHES)], edgecolor="white",
                             linewidth=0, alpha=0.9,
                             label=f"{experiment_label(conn, e)[:-1]}; mean {sum(points) / n:.2f})"))

        wins = sum(p >= 2 for p in points)
        # Left counts sit right of x = 0 to clear the chance marker there (~81%).
        for x_side, games in ((0.85, n - wins), (3.5, wins)):
            ax.text(x_side, 0.9 - 0.07 * idx, f"{games} of {n} ({100 * games / n:.1f}%)",
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color=color)

    blind = [math.comb(ROUNDS, p) * chance ** p * (1 - chance) ** (ROUNDS - p) for p in range(ROUNDS + 1)]
    chance_line, = ax.plot(range(ROUNDS + 1), blind, color=CHANCE_COLOR, linestyle=(0, (1, 2)),
                           linewidth=0.8, marker="x", markersize=3, zorder=4,
                           label="Chance (blind Interceptor)")
    handles.append(chance_line)

    ax.set_xlabel("Interceptor points per game (I or F = 1, S = 0)")
    ax.set_ylabel("Games")
    ax.set_xticks(range(ROUNDS + 1))
    ax.set_xlim(-0.5, ROUNDS + 0.5)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.5)
    ax.set_axisbelow(True)

    # Labels here are long (n and mean): more than three entries wrap to two columns to fit the width.
    ncol = len(handles) if len(handles) <= 3 else 2
    rows = math.ceil(len(handles) / ncol)
    fig.legend(handles=handles, loc="upper center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 1.02), handlelength=2.4)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.1 * rows))
    return _save(fig, name, out_dir)


def _plot_figure(conn: sqlite3.Connection, experiment_ids: list, name: str, panels: list, chance: float,
                 width_pt: float, out_dir: str) -> list:
    width_in = width_pt / 72.27
    fig, axes = plt.subplots(1, len(panels), figsize=(width_in, width_in * 0.42), sharey=True)

    handles = []
    for idx, e in enumerate(experiment_ids):
        data = series(conn, e)
        color = COLORS[idx % len(COLORS)]
        for ax, (key, _) in zip(axes, panels):
            rounds, rate, lo, hi = data[key]
            ax.fill_between(rounds, lo, hi, color=color, alpha=0.18, linewidth=0)
            line, = ax.plot(rounds, rate, color=color, linestyle=LINESTYLES[idx % len(LINESTYLES)],
                            marker=MARKERS[idx % len(MARKERS)], markersize=3, linewidth=1.2)
        line.set_label(experiment_label(conn, e))
        handles.append(line)

    chance_line = None
    for ax, (key, title) in zip(axes, panels):
        rounds = list(range(1, 6))
        curve = chance_curve(key, chance, rounds)
        if curve is not None:
            chance_line, = ax.plot(rounds, curve, color=CHANCE_COLOR, linestyle=(0, (1, 2)), linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.set_xticks(range(1, 6))
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.5)
        ax.set_axisbelow(True)
    if chance_line is not None:
        chance_line.set_label(f"Chance (1/{round(1 / chance)})")
        handles.append(chance_line)

    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.02), handlelength=2.4)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, name, out_dir)


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

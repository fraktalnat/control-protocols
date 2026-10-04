from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")


def plot_sus_distribution(
    clean_scores: list[float],
    backdoor_scores: list[float],
    out_path: Union[str, Path],
    threshold: Optional[float] = None,
) -> None:
    """Overlaid histogram of clean vs backdoor suspicion scores.

    Pass the 2%-FPR threshold from `analyze_detector_scores` to draw it
    as a dashed vertical line.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = 20
    ax.hist(clean_scores, bins=bins, alpha=0.6, label="clean", color="tab:blue")
    ax.hist(backdoor_scores, bins=bins, alpha=0.6, label="backdoor", color="tab:red")
    if threshold is not None:
        ax.axvline(
            threshold,
            color="black",
            linestyle="--",
            label=f"2%-FPR threshold = {threshold:.2f}",
        )
    ax.set_xlabel("Suspicion score")
    ax.set_ylabel("Count")
    ax.set_title("Suspicion-score distribution: clean vs backdoor")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_protocol_comparison(
    results: dict[str, float], out_path: Union[str, Path]
) -> None:
    """Bar chart of safety per protocol at a fixed backdoor rate.

    ``results`` is ``{protocol_name: safety}`` — typically the output of
    ``simulate_safety(...)``.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    values = [results[n] for n in names]
    ax.bar(
        names,
        values,
        color=["tab:gray", "tab:blue", "tab:green", "tab:red"][: len(names)],
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Safety (P[no backdoor submitted])")
    ax.set_title("Protocol safety comparison")
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_safety_usefulness_pareto(
    results: dict[str, tuple[float, float]], out_path: Union[str, Path]
) -> None:
    """Scatter plot of (safety, usefulness) per protocol with Pareto frontier marked.

    ``results`` is ``{protocol_name: (safety, usefulness)}``. Pareto-optimal
    protocols (no other point dominates both axes) are connected by a dashed line.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    names = list(results.keys())
    xs = [results[n][0] for n in names]
    ys = [results[n][1] for n in names]
    ax.scatter(xs, ys, color="tab:purple")
    for n, x, y in zip(names, xs, ys):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(6, 6))

    pareto = []
    for n in names:
        sx, sy = results[n]
        dominated = any(
            (results[m][0] > sx and results[m][1] >= sy)
            or (results[m][0] >= sx and results[m][1] > sy)
            for m in names
            if m != n
        )
        if not dominated:
            pareto.append((sx, sy))
    pareto.sort()
    if len(pareto) > 1:
        px, py = zip(*pareto)
        ax.plot(
            px,
            py,
            color="tab:purple",
            linestyle="--",
            alpha=0.6,
            label="Pareto frontier",
        )
        ax.legend()
    ax.set_xlabel("Safety")
    ax.set_ylabel("Usefulness")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Safety vs Usefulness (paper Figure 2)")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

"""Build Appendix E report figures from saved CSV and PNG artifacts."""

import csv
import json
import math
import os
import tempfile
from pathlib import Path

from sparse_attention import ERROR_LIMIT, LATENT_SCALE, RESOLUTIONS, WINDOW


METHODS = ("full", "sparse")
COLORS = {"full": "#4d79a4", "sparse": "#dc8346"}
TOKENS = [(size // LATENT_SCALE)**2 for size in RESOLUTIONS]
TOKEN_LABELS = [f"{tokens:,}" for tokens in TOKENS]


def load_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save_figure(plt, figure, directory, name):
    figure.savefig(directory / f"{name}.png", dpi=300)
    figure.savefig(directory / f"{name}.pdf")
    plt.close(figure)


def window_figure(plt, directory):
    from matplotlib.patches import Rectangle

    fig, (grid, pairs) = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    side = RESOLUTIONS[0] // LATENT_SCALE
    grid.add_patch(Rectangle((WINDOW, WINDOW), WINDOW, WINDOW,
                             facecolor=COLORS["sparse"], alpha=0.35))
    grid.plot(20.5, 17.5, "o", color="#79370b", label="Query (17, 20)")
    ticks = list(range(0, side + 1, WINDOW))
    grid.set(xlim=(0, side), ylim=(side, 0), xticks=ticks, yticks=ticks, aspect="equal",
             xlabel="Latent column", ylabel="Latent row", title="16×16 windows")
    grid.grid(alpha=0.5)
    grid.legend(loc="lower right", frameon=True)
    retained = [100 * WINDOW**2 / tokens for tokens in TOKENS]
    bars = pairs.bar(TOKEN_LABELS, retained, color=COLORS["sparse"])
    pairs.bar_label(bars, labels=[f"{percent:.2f}%" for percent in retained], padding=4)
    pairs.set(xlabel="Tokens", ylabel="Retained pairs (%)", title="Theoretical window connectivity")
    pairs.margins(y=0.22)
    save_figure(plt, fig, directory, "window_layout")


def qualification_figure(plt, directory, rows):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), layout="constrained")
    for ax, field, title in zip(
        axes, ("max_abs_error", "relative_rms_error"), ("Maximum absolute error", "Relative RMS error"),
    ):
        ax.plot(TOKEN_LABELS, [float(rows[size][field]) for size in RESOLUTIONS], "o-",
                color=COLORS["sparse"])
        ax.axhline(ERROR_LIMIT, color="#555555", linestyle="--", label="Limit (0.005)")
        ax.set(xlabel="Tokens", title=title, ylim=(0, None))
        ax.legend(frameon=False)
    save_figure(plt, fig, directory, "numerical_validation")


def performance_figure(plt, directory, summary):
    fig, (time_ax, memory_ax, speed_ax) = plt.subplots(
        1, 3, figsize=(12, 3.8), layout="constrained",
    )
    positions = list(range(len(RESOLUTIONS)))
    width = 0.34
    for offset, method in zip((-width / 2, width / 2), METHODS):
        rows = [summary[size, method] for size in RESOLUTIONS]
        medians = [float(row["median_ms"]) for row in rows]
        errors = [
            [max(0, median - float(row["p25_ms"])) for row, median in zip(rows, medians)],
            [max(0, float(row["p75_ms"]) - median) for row, median in zip(rows, medians)],
        ]
        x = [pos + offset for pos in positions]
        time_ax.bar(x, medians, width, yerr=errors, capsize=3,
                    color=COLORS[method], label=method.title())
        memory_ax.bar(x, [float(row["median_extra_alloc_mib"]) for row in rows], width,
                      color=COLORS[method], label=method.title())
    time_ax.set(title="Latency", ylabel="Median and IQR (ms)")
    time_ax.legend(frameon=False)
    memory_ax.set(title="Additional peak memory", ylabel="Median (MiB)")
    ratios = [float(summary[size, "sparse"]["full_over_sparse"]) for size in RESOLUTIONS]
    speed_ax.plot(positions, ratios, "o-", color=COLORS["sparse"])
    speed_ax.axhline(1.0, color="#555555", linestyle="--", label="Equal latency")
    speed_ax.set(title="Speedup", ylabel="Full / Sparse (×)")
    speed_ax.legend(frameon=False)
    for ax in (time_ax, memory_ax, speed_ax):
        ax.set(xticks=positions, xticklabels=TOKEN_LABELS, xlabel="Tokens")
    save_figure(plt, fig, directory, "module_performance")


def repeated_latency_figure(plt, directory, draws):
    fig, axes = plt.subplots(1, len(RESOLUTIONS), figsize=(9, 3.5), layout="constrained")
    for size, label, axis in zip(RESOLUTIONS, TOKEN_LABELS, axes):
        samples = [draws[size, method] for method in METHODS]
        boxes = axis.boxplot(samples, tick_labels=["Full", "Sparse"], patch_artist=True,
                             medianprops={"color": "#333333"}, showfliers=False)
        for index, (method, values, box) in enumerate(zip(METHODS, samples, boxes["boxes"]), start=1):
            box.set(facecolor=COLORS[method], alpha=0.35)
            axis.scatter([index] * len(values), values, s=10, color=COLORS[method], alpha=0.6)
        axis.set(title=f"{label} tokens", ylabel="Latency (ms)")
    save_figure(plt, fig, directory, "latency_repeats")


def plot_results(run, repeats):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sd-lab-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    qualification_rows = load_rows(run / "qualify.csv")
    qualifications = {int(row["resolution"]): row for row in qualification_rows}
    if (
        len(qualification_rows) != len(RESOLUTIONS) or set(qualifications) != set(RESOLUTIONS)
        or any(row["passed"] != "True" for row in qualification_rows)
    ):
        raise ValueError("The plot requires a successful qualification for each resolution.")
    summaries = load_rows(run / "benchmark_summary.csv")
    summary = {(int(row["resolution"]), row["method"]): row for row in summaries}
    keys = {(size, method) for size in RESOLUTIONS for method in METHODS}
    if len(summaries) != len(keys) or set(summary) != keys:
        raise ValueError("Each resolution needs one Full and one Sparse benchmark summary.")
    draws = {key: [] for key in keys}
    for row in load_rows(run / "benchmark_raw.csv"):
        key = (int(row["resolution"]), row["method"])
        elapsed = float(row["elapsed_ms"])
        if key not in draws or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("The raw benchmark CSV contains an invalid reading.")
        draws[key].append(elapsed)
    if any(len(draws[key]) != repeats or int(summary[key]["readings"]) != repeats for key in keys):
        raise ValueError(f"Expected {repeats} readings per method and resolution.")

    generation = json.loads((run / "generation.json").read_text(encoding="utf-8"))
    pictures = []
    for method in METHODS:
        filename = f"finetuned_{method}.png"
        result = generation["methods"][method]
        if result["file"] != filename or result["automatic_checks_passed"] is not True:
            raise ValueError(f"The {method} comparison image failed automatic validation.")
        with Image.open(run / "images" / filename) as image:
            pictures.append(image.copy())
    directory = run / "figures"
    directory.mkdir()
    window_figure(plt, directory)
    qualification_figure(plt, directory, qualifications)
    performance_figure(plt, directory, summary)
    repeated_latency_figure(plt, directory, draws)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.6), layout="constrained")
    for ax, method, picture in zip(axes, METHODS, pictures):
        ax.imshow(picture)
        ax.set_title(f"Finetuned {method.title()}")
        ax.set_axis_off()
    save_figure(plt, fig, directory, "generation_comparison")
    print(f"Figures saved to {directory}")

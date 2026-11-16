#!/usr/bin/env python3
"""Create publication/poster figures for one locked SURF Phase-1 result.

The script reads existing Phase-1 metrics and curves.  It does not recompute
the benchmark ranking or tune a threshold on test labels.  Qualitative cases
are selected deterministically at the 20th, 50th and 80th percentiles of the
already-computed patient AUPRC distribution to avoid cherry-picking.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np


GREEN = "#10A37F"
BLUE = "#3B82F6"
ORANGE = "#F59E0B"
RED = "#EF4444"
DARK = "#202123"
MID = "#6E6E73"
LIGHT = "#ECECF1"
PALE = "#E8F7F2"
BACKGROUND = "#F3F8FB"
GRID = "#DFE9EF"
FOOTER = "Created by ChatGPT · Designed by Hongyao Yang"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create poster-ready cDDPM figures from locked Phase-1 outputs.",
    )
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--ema-alpha", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.dpi < 150:
        parser.error("--dpi must be >= 150 for poster export")
    if not 0 < args.ema_alpha <= 1:
        parser.error("--ema-alpha must be in (0,1]")
    return args


def setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.edgecolor": "#B8B8BD",
            "axes.linewidth": 0.8,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "text.color": DARK,
            "axes.labelcolor": DARK,
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def require_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows


def scalar(value: np.ndarray, field: str, path: Path) -> str:
    if value.shape not in ((), (1,)):
        raise ValueError(f"{path}: {field} must be scalar")
    return str(value.item())


def value(row: dict[str, str], key: str) -> float:
    try:
        result = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing/invalid numeric column {key}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value in {key}")
    return result


def clean_axis(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)


def header(fig: Any, title: str, subtitle: str) -> None:
    fig.text(0.04, 0.965, title, fontsize=22, fontweight="bold", ha="left", va="top")
    fig.text(0.04, 0.925, subtitle, fontsize=10.5, color=MID, ha="left", va="top")
    fig.text(0.985, 0.018, FOOTER, fontsize=7.5, color="#8E8E93", ha="right", va="bottom")


def save(fig: Any, output: Path, stem: str, dpi: int) -> None:
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"{stem}.{extension}", dpi=dpi, bbox_inches="tight")


def locate_model_dir(analysis: Path, matrix_row: dict[str, str]) -> Path:
    model_dir = analysis / "model_results" / matrix_row["submission_id"]
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model result directory missing: {model_dir}")
    return model_dir


def plot_key_metrics(plt: Any, row: dict[str, str], output: Path, dpi: int) -> None:
    metrics = [
        ("residual_macro_pixel_auprc", "residual_macro_pixel_auprc_ci95_low", "residual_macro_pixel_auprc_ci95_high", "Macro AUPRC"),
        ("residual_macro_pixel_auroc", "residual_macro_pixel_auroc_ci95_low", "residual_macro_pixel_auroc_ci95_high", "Macro AUROC"),
    ]
    pooled = [
        ("residual_pooled_pixel_auprc", "Pooled AUPRC"),
        ("residual_pooled_pixel_auroc", "Pooled AUROC"),
    ]
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    positions = np.arange(4)
    means = [value(row, item[0]) for item in metrics] + [value(row, item[0]) for item in pooled]
    lower = [value(row, item[1]) for item in metrics]
    upper = [value(row, item[2]) for item in metrics]
    yerr_low = [means[i] - lower[i] for i in range(2)] + [0.0, 0.0]
    yerr_high = [upper[i] - means[i] for i in range(2)] + [0.0, 0.0]
    labels = [item[3] for item in metrics] + [item[1] for item in pooled]
    colors = [GREEN, BLUE, "#70CBB2", "#8AB4F8"]
    bars = ax.bar(positions, means, color=colors, width=0.62, alpha=0.94)
    ax.errorbar(
        positions,
        means,
        yerr=np.asarray([yerr_low, yerr_high]),
        fmt="none",
        ecolor=DARK,
        capsize=6,
        linewidth=1.5,
    )
    for bar, number in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, number + 0.025, f"{number:.3f}", ha="center", fontweight="bold")
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Threshold-free score")
    clean_axis(ax)
    header(
        fig,
        "cDDPM-SparK benchmark performance",
        "Macro AUPRC is the official patient-balanced ranking metric; whiskers show patient-bootstrap 95% CI.",
    )
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.10, right=0.97)
    save(fig, output, "poster_01_key_metrics", dpi)
    plt.close(fig)


def curve_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = read_csv(path)
    return tuple(
        np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in ("fpr", "tpr", "recall", "precision")
    )  # type: ignore[return-value]


def plot_performance_summary(
    plt: Any,
    row: dict[str, str],
    model_dir: Path,
    patient_rows: list[dict[str, str]],
    output: Path,
    dpi: int,
) -> None:
    """Four-panel performance figure designed as the main poster result."""
    residual = curve_arrays(model_dir / "curves" / "harmonized_residual_pooled_curve.csv")
    native = curve_arrays(model_dir / "curves" / "paper_native_pooled_curve.csv")
    aggregate = json.loads((model_dir / "aggregate_metrics.json").read_text(encoding="utf-8"))
    prevalence = (
        float(aggregate["counts"]["pooled_brats_tumor_voxels"])
        / float(aggregate["counts"]["pooled_brats_brain_voxels"])
    )
    brats = [
        item
        for item in patient_rows
        if item["dataset"] == "brats2020_test" and item["residual_pixel_auprc"]
    ]
    patient_auprc = np.asarray([float(item["residual_pixel_auprc"]) for item in brats])

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.0))
    pr_ax, metric_ax, dist_ax, roc_ax = axes.ravel()
    tracks = [
        (residual, "Harmonized residual", GREEN, "-"),
        (native, "Paper-native", BLUE, "--"),
    ]
    for arrays, label, color, linestyle in tracks:
        fpr, tpr, recall, precision = arrays
        pr_ax.plot(recall, precision, color=color, linestyle=linestyle, linewidth=2.5, label=label)
        roc_ax.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=2.5, label=label)
    pr_ax.axhline(prevalence, color="#B8B8BD", linestyle=":", linewidth=1.4, label=f"Random baseline {prevalence:.3f}")
    pr_ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
    pr_ax.set_title(f"A · Precision–recall · pooled AUPRC {value(row, 'residual_pooled_pixel_auprc'):.3f}", loc="left")
    pr_ax.legend(frameon=False, fontsize=8, loc="best")

    metric_names = ["Macro\nAUPRC", "Macro\nAUROC"]
    metric_values = np.asarray([
        value(row, "residual_macro_pixel_auprc"),
        value(row, "residual_macro_pixel_auroc"),
    ])
    lows = np.asarray([
        value(row, "residual_macro_pixel_auprc_ci95_low"),
        value(row, "residual_macro_pixel_auroc_ci95_low"),
    ])
    highs = np.asarray([
        value(row, "residual_macro_pixel_auprc_ci95_high"),
        value(row, "residual_macro_pixel_auroc_ci95_high"),
    ])
    positions = np.arange(2)
    metric_ax.errorbar(
        positions,
        metric_values,
        yerr=np.vstack([metric_values - lows, highs - metric_values]),
        fmt="o",
        markersize=12,
        color=GREEN,
        ecolor=DARK,
        capsize=7,
        linewidth=1.7,
    )
    for position, number in zip(positions, metric_values):
        metric_ax.text(position, min(1.0, number + 0.065), f"{number:.3f}", ha="center", fontweight="bold")
    metric_ax.set_xticks(positions, metric_names)
    metric_ax.set(ylim=(0, 1.04), ylabel="Patient-balanced score")
    metric_ax.set_title("B · Official macro metrics with 95% CI", loc="left")
    metric_ax.text(0, 0.08, "Official primary metric", ha="center", color=GREEN, fontsize=9, fontweight="bold")

    bins = np.linspace(0, 1, 26)
    dist_ax.hist(patient_auprc, bins=bins, color=GREEN, alpha=0.9, edgecolor=BACKGROUND)
    dist_ax.axvline(patient_auprc.mean(), color=DARK, linewidth=2, label=f"Mean {patient_auprc.mean():.3f}")
    dist_ax.axvline(np.median(patient_auprc), color=ORANGE, linewidth=2, linestyle="--", label=f"Median {np.median(patient_auprc):.3f}")
    dist_ax.set(xlim=(0, 1), xlabel="Per-patient pixel AUPRC", ylabel="Patients")
    dist_ax.set_title(f"C · Patient distribution · n={patient_auprc.size}", loc="left")
    dist_ax.legend(frameon=False, fontsize=8)

    roc_ax.plot([0, 1], [0, 1], color="#B8B8BD", linestyle=":", linewidth=1.3, label="Random ordering")
    roc_ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False-positive rate", ylabel="True-positive rate")
    roc_ax.set_title(f"D · ROC · pooled AUROC {value(row, 'residual_pooled_pixel_auroc'):.3f}", loc="left")
    roc_ax.legend(frameon=False, fontsize=8, loc="lower right")

    for ax in axes.ravel():
        clean_axis(ax)
    header(
        fig,
        "cDDPM-SparK tumor anomaly localization",
        "Official ranking: harmonized residual macro pixel AUPRC. PR is emphasized because tumor pixels are the minority.",
    )
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.08, right=0.98, hspace=0.34, wspace=0.22)
    save(fig, output, "poster_00_performance_summary", dpi)
    plt.close(fig)


def plot_curves(plt: Any, row: dict[str, str], model_dir: Path, output: Path, dpi: int) -> None:
    residual = curve_arrays(model_dir / "curves" / "harmonized_residual_pooled_curve.csv")
    native = curve_arrays(model_dir / "curves" / "paper_native_pooled_curve.csv")
    aggregate = json.loads((model_dir / "aggregate_metrics.json").read_text(encoding="utf-8"))
    pooled_voxels = float(aggregate["counts"]["pooled_brats_brain_voxels"])
    tumor_voxels = float(aggregate["counts"]["pooled_brats_tumor_voxels"])
    prevalence = tumor_voxels / pooled_voxels
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2))
    tracks = [
        (residual, "Harmonized residual", GREEN, "-"),
        (native, "Paper-native", BLUE, "--"),
    ]
    for arrays, label, color, linestyle in tracks:
        fpr, tpr, recall, precision = arrays
        axes[0].plot(recall, precision, color=color, linestyle=linestyle, linewidth=2.6, label=label)
        axes[1].plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=2.6, label=label)
    axes[0].axhline(
        prevalence,
        color="#B8B8BD",
        linestyle=":",
        linewidth=1.4,
        label=f"Random PR baseline ({prevalence:.3f})",
    )
    axes[1].plot([0, 1], [0, 1], "--", color="#B8B8BD", linewidth=1, label="Random ordering")
    axes[0].set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
    axes[1].set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False-positive rate", ylabel="True-positive rate")
    axes[0].set_title(f"Precision–recall · pooled AUPRC {value(row, 'residual_pooled_pixel_auprc'):.3f}", loc="left")
    axes[1].set_title(f"ROC · pooled AUROC {value(row, 'residual_pooled_pixel_auroc'):.3f}", loc="left")
    for ax in axes:
        clean_axis(ax)
        ax.legend(frameon=False, loc="best", fontsize=9)
    header(fig, "Threshold-free localization curves", "PR is emphasized because tumor pixels are the minority; ROC is complementary evidence.")
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.08, right=0.98, wspace=0.20)
    save(fig, output, "poster_02_pr_roc_curves", dpi)
    plt.close(fig)


def plot_patient_distribution(plt: Any, patient_rows: list[dict[str, str]], output: Path, dpi: int) -> None:
    rows = [row for row in patient_rows if row["dataset"] == "brats2020_test" and row["residual_pixel_auprc"]]
    residual = np.asarray([float(row["residual_pixel_auprc"]) for row in rows])
    native = np.asarray([float(row["native_pixel_auprc"]) for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.3))
    bins = np.linspace(0, 1, 26)
    axes[0].hist(residual, bins=bins, color=GREEN, alpha=0.85, edgecolor="white")
    axes[0].axvline(residual.mean(), color=DARK, linewidth=2, label=f"Mean {residual.mean():.3f}")
    axes[0].axvline(np.median(residual), color=ORANGE, linewidth=2, linestyle="--", label=f"Median {np.median(residual):.3f}")
    axes[0].set(xlabel="Per-patient pixel AUPRC", ylabel="Patients", xlim=(0, 1))
    axes[0].set_title("Distribution across BraTS2020 patients", loc="left")
    axes[0].legend(frameon=False)
    parts = axes[1].violinplot([residual, native], positions=[1, 2], showmeans=True, showmedians=True)
    for body, color in zip(parts["bodies"], [GREEN, BLUE]):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.65)
    axes[1].set_xticks([1, 2], ["Residual", "Native"])
    axes[1].set(ylabel="Per-patient pixel AUPRC", ylim=(0, 1.02))
    axes[1].set_title("Residual vs paper-native score", loc="left")
    for ax in axes:
        clean_axis(ax)
    header(fig, "Patient-level robustness", "A single mean can hide failures; the distribution reveals heterogeneity across 369 patients.")
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.08, right=0.98, wspace=0.20)
    save(fig, output, "poster_03_patient_distribution", dpi)
    plt.close(fig)


def choose_cases(patient_rows: list[dict[str, str]]) -> list[tuple[str, str, float]]:
    items = sorted(
        (float(row["residual_pixel_auprc"]), row["patient_id"])
        for row in patient_rows
        if row["dataset"] == "brats2020_test" and row["residual_pixel_auprc"]
    )
    selected: list[tuple[str, str, float]] = []
    for label, quantile in (("Challenging (20th percentile)", 0.20), ("Typical (median)", 0.50), ("Strong (80th percentile)", 0.80)):
        target = float(np.quantile([item[0] for item in items], quantile))
        score, patient_id = min(items, key=lambda item: abs(item[0] - target))
        selected.append((label, patient_id, score))
    return selected


def load_case(benchmark_path: Path, prediction_path: Path) -> dict[str, np.ndarray | str]:
    with np.load(benchmark_path, allow_pickle=False) as data:
        image = np.asarray(data["image"], dtype=np.float32)
        tumor = np.asarray(data["tumor_mask"]).astype(bool)
        brain = np.asarray(data["brain_mask"]).astype(bool)
        patient_id = scalar(np.asarray(data["patient_id"]), "patient_id", benchmark_path)
    with np.load(prediction_path, allow_pickle=False) as data:
        reconstruction = np.asarray(data["reconstruction"], dtype=np.float32)
        native = np.asarray(data["anomaly_map"], dtype=np.float32)
    residual = np.abs(image - reconstruction) * brain
    native = native * brain
    return {
        "patient_id": patient_id,
        "image": image,
        "reconstruction": reconstruction,
        "residual": residual,
        "native": native,
        "tumor": tumor,
    }


def plot_qualitative(
    plt: Any,
    patient_rows: list[dict[str, str]],
    benchmark_root: Path,
    handoff_root: Path,
    output: Path,
    dpi: int,
) -> None:
    selected = choose_cases(patient_rows)
    cases = [
        (label, score, load_case(
            benchmark_root / "brats2020" / "test" / f"{patient_id}.npz",
            handoff_root / "02_test_predictions" / "brats2020_test" / f"{patient_id}.npz",
        ))
        for label, patient_id, score in selected
    ]
    fig, axes = plt.subplots(3, 5, figsize=(15.0, 9.4))
    column_titles = ["Input T2", "Reconstruction", "Residual map", "Native map", "Tumor reference"]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold")
    for row_index, (label, score, case) in enumerate(cases):
        tumor = case["tumor"]
        assert isinstance(tumor, np.ndarray)
        slice_index = int(np.argmax(tumor.reshape(tumor.shape[0], -1).sum(axis=1)))
        image = np.asarray(case["image"])[slice_index]
        reconstruction = np.asarray(case["reconstruction"])[slice_index]
        residual = np.asarray(case["residual"])[slice_index]
        native = np.asarray(case["native"])[slice_index]
        mask = tumor[slice_index]
        map_vmax = max(float(np.quantile(residual[residual > 0], 0.99)) if np.any(residual > 0) else 1.0, 1e-6)
        native_vmax = max(float(np.quantile(native[native > 0], 0.99)) if np.any(native > 0) else 1.0, 1e-6)
        panels = [image, reconstruction, residual, native, image]
        for col, panel in enumerate(panels):
            ax = axes[row_index, col]
            if col in (2, 3):
                vmax = map_vmax if col == 2 else native_vmax
                ax.imshow(panel, cmap="magma", vmin=0, vmax=vmax)
            else:
                ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
            if col == 4 and np.any(mask):
                ax.contour(mask.astype(float), levels=[0.5], colors=[RED], linewidths=1.5)
            ax.axis("off")
        axes[row_index, 0].text(
            -0.06,
            0.5,
            f"{label}\n{case['patient_id']}\nAUPRC {score:.3f}\nslice {slice_index + 15}",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9,
            color=DARK,
        )
    header(
        fig,
        "Representative reconstruction and anomaly localization",
        "Cases are selected deterministically at patient-AUPRC percentiles; red contours are references used only for post-evaluation visualization.",
    )
    fig.subplots_adjust(top=0.86, bottom=0.04, left=0.15, right=0.99, hspace=0.10, wspace=0.04)
    save(fig, output, "poster_04_qualitative_cases", dpi)
    plt.close(fig)


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    state = float(values[0])
    for index, number in enumerate(values):
        state = float(number) if index == 0 else alpha * float(number) + (1 - alpha) * state
        result[index] = state
    return result


def plot_loss(plt: Any, handoff_root: Path, output: Path, dpi: int, alpha: float) -> None:
    path = handoff_root / "01_training_evidence" / "training_loss_steps.csv"
    rows = read_csv(path)
    steps = np.asarray([float(row["global_step"]) for row in rows])
    losses = np.asarray([float(row["total_loss"]) for row in rows])
    finite = np.isfinite(steps) & np.isfinite(losses)
    steps, losses = steps[finite], losses[finite]
    smooth = ema(losses, alpha)
    stride = max(1, int(math.ceil(len(steps) / 6000)))
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    ax.plot(steps[::stride], losses[::stride], color="#9ADBC9", linewidth=0.75, alpha=0.55, label="Raw total loss")
    ax.plot(steps, smooth, color=GREEN, linewidth=2.4, label=f"EMA (α={alpha:g})")
    ax.set(xlabel="Global training step", ylabel="Training objective")
    ax.set_title("Optimization convergence", loc="left")
    clean_axis(ax)
    ax.legend(frameon=False)
    header(fig, "cDDPM-SparK training convergence", "Loss demonstrates optimization stability; it is not a test-performance metric and must not rank different objectives.")
    fig.subplots_adjust(top=0.79, bottom=0.16, left=0.10, right=0.98)
    save(fig, output, "poster_05_training_loss", dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    plt = setup_matplotlib()
    analysis = require_dir(args.analysis_dir, "analysis-dir")
    benchmark = require_dir(args.benchmark_root, "benchmark-root")
    handoff = require_dir(args.handoff_root, "handoff-root")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}; pass --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    matrix_rows = read_csv(analysis / "performance_matrix.csv")
    if len(matrix_rows) != 1:
        raise ValueError(f"Poster script expects exactly one model/seed row, found {len(matrix_rows)}")
    row = matrix_rows[0]
    model_dir = locate_model_dir(analysis, row)
    patient_rows = read_csv(model_dir / "per_patient_metrics.csv")

    plot_key_metrics(plt, row, output, args.dpi)
    plot_performance_summary(plt, row, model_dir, patient_rows, output, args.dpi)
    plot_curves(plt, row, model_dir, output, args.dpi)
    plot_patient_distribution(plt, patient_rows, output, args.dpi)
    plot_qualitative(plt, patient_rows, benchmark, handoff, output, args.dpi)
    plot_loss(plt, handoff, output, args.dpi, args.ema_alpha)

    manifest = {
        "phase1_analysis": str(analysis),
        "handoff": str(handoff),
        "benchmark": str(benchmark),
        "submission_id": row["submission_id"],
        "official_primary_metric": "residual_macro_pixel_auprc",
        "figures": [
            "poster_00_performance_summary",
            "poster_01_key_metrics",
            "poster_02_pr_roc_curves",
            "poster_03_patient_distribution",
            "poster_04_qualitative_cases",
            "poster_05_training_loss",
        ],
        "formats": ["png", "pdf", "svg"],
        "metrics_recomputed": False,
        "threshold_tuned": False,
        "qualitative_selection": "20th, 50th and 80th percentiles of locked patient residual AUPRC",
    }
    (output / "poster_figures_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Poster figures completed successfully.")
    print(f"Output: {output}")
    for stem in manifest["figures"]:
        print(f"  {stem}.png / .pdf / .svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

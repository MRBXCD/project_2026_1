"""
GRPO Training Log Analyzer
===========================

Analyze GRPO training metrics from wandb CSV exports.

Produces:
    1. Terminal summary table with per-segment statistics.
    2. Trend analysis (is reward improving? is std converging?).
    3. Diagnostic checks (reward hacking, training collapse).
    4. PNG chart saved to the same directory as the input CSVs.

Usage:
    python -m utils.analyze_training --log_dir exp_log
    python -m utils.analyze_training --log_dir exp_log --window 30
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Data Loading
# ============================================================

def load_wandb_csvs(log_dir: Path) -> dict[str, dict]:
    """Load all wandb CSV exports from a directory.

    Auto-detects the metric name from each CSV header.

    Returns:
        dict mapping metric short name to {"steps": list, "values": list}.
    """
    metric_map = {
        "step_time": "train/step_time",
        "reward_fn_std": "train/rewards/reward_fn/std",
        "reward_fn_mean": "train/rewards/reward_fn/mean",
        "reward_std": "train/reward_std",
        "reward": "train/reward",
        "num_tokens": "train/num_tokens",
    }

    # Reverse: full metric name -> short name
    reverse_map = {v: k for k, v in metric_map.items()}

    data = {}
    for csv_path in sorted(log_dir.glob("*.csv")):
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)

        # Identify metric from header column 1
        full_metric = header[1].split(" - ")[-1] if " - " in header[1] else header[1]
        short_name = reverse_map.get(full_metric, full_metric)

        steps = []
        values = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row[header[0]]))
                values.append(float(row[header[1]]))

        data[short_name] = {"steps": np.array(steps), "values": np.array(values)}

    return data


# ============================================================
# Statistics
# ============================================================

def compute_segment_stats(values: np.ndarray, n_segments: int = 3) -> list[dict]:
    """Split values into N equal segments and compute stats for each."""
    segments = np.array_split(values, n_segments)
    stats = []
    for seg in segments:
        stats.append({
            "mean": float(np.mean(seg)),
            "std": float(np.std(seg)),
            "min": float(np.min(seg)),
            "max": float(np.max(seg)),
        })
    return stats


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Compute simple moving average with given window size."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    # "valid" mode avoids edge artifacts
    smoothed = np.convolve(values, kernel, mode="valid")
    return smoothed


# ============================================================
# Diagnostics
# ============================================================

def run_diagnostics(data: dict) -> list[str]:
    """Run diagnostic checks and return a list of findings."""
    findings = []

    if "reward_fn_mean" in data:
        vals = data["reward_fn_mean"]["values"]
        seg = compute_segment_stats(vals)

        # Trend: is reward improving?
        trend = seg[-1]["mean"] - seg[0]["mean"]
        if trend > 0.1:
            findings.append(
                f"[POSITIVE] Reward mean shows upward trend: "
                f"{seg[0]['mean']:.3f} -> {seg[-1]['mean']:.3f} (delta={trend:+.3f})"
            )
        elif trend < -0.1:
            findings.append(
                f"[WARNING] Reward mean is DECLINING: "
                f"{seg[0]['mean']:.3f} -> {seg[-1]['mean']:.3f} (delta={trend:+.3f})"
            )
        else:
            findings.append(
                f"[NEUTRAL] Reward mean is flat: "
                f"{seg[0]['mean']:.3f} -> {seg[-1]['mean']:.3f} (delta={trend:+.3f})"
            )

        # Reward hacking check: rapid spike in last segment
        last_third = vals[len(vals) * 2 // 3:]
        if len(last_third) > 10:
            slope = np.polyfit(range(len(last_third)), last_third, 1)[0]
            if slope > 0.01:
                findings.append(
                    f"[CAUTION] Reward accelerating in final segment "
                    f"(slope={slope:.4f}/step). Check for reward hacking."
                )

    if "reward_fn_std" in data:
        vals = data["reward_fn_std"]["values"]
        seg = compute_segment_stats(vals)

        std_trend = seg[-1]["mean"] - seg[0]["mean"]
        if std_trend < -0.05:
            findings.append(
                f"[INFO] Reward std decreasing: "
                f"{seg[0]['mean']:.3f} -> {seg[-1]['mean']:.3f}. "
                f"Model outputs becoming more uniform."
            )
        elif std_trend > 0.05:
            findings.append(
                f"[INFO] Reward std increasing: "
                f"{seg[0]['mean']:.3f} -> {seg[-1]['mean']:.3f}. "
                f"Reward diversity growing."
            )

    if "step_time" in data:
        vals = data["step_time"]["values"]
        cv = np.std(vals) / np.mean(vals)
        findings.append(
            f"[INFO] Step time: mean={np.mean(vals):.1f}s, "
            f"cv={cv:.2f} ({'stable' if cv < 0.3 else 'variable'})"
        )

    return findings


# ============================================================
# Plotting
# ============================================================

def plot_metrics(data: dict, window: int, output_path: Path) -> None:
    """Generate a multi-panel chart of all training metrics."""
    plot_configs = [
        ("reward_fn_mean", "Reward Mean", "Reward"),
        ("reward_fn_std", "Reward Std (intra-group)", "Std"),
        ("reward", "Raw Reward", "Reward"),
        ("reward_std", "Raw Reward Std", "Std"),
        ("step_time", "Step Time", "Seconds"),
    ]

    available = [(key, title, ylabel) for key, title, ylabel in plot_configs if key in data]
    n_plots = len(available)
    if n_plots == 0:
        print("No plottable metrics found.")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for ax, (key, title, ylabel) in zip(axes, available):
        steps = data[key]["steps"]
        vals = data[key]["values"]

        ax.plot(steps, vals, alpha=0.3, color="steelblue", linewidth=0.8, label="Raw")

        # Moving average
        smoothed = moving_average(vals, window)
        if len(smoothed) > 0:
            offset = window // 2
            smooth_steps = steps[offset : offset + len(smoothed)]
            ax.plot(smooth_steps, smoothed, color="darkorange", linewidth=2.0,
                    label=f"MA({window})")

        # Segment means as horizontal lines
        seg_stats = compute_segment_stats(vals)
        seg_size = len(vals) // 3
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]
        labels = ["First 1/3", "Mid 1/3", "Last 1/3"]
        for i, (stat, color, label) in enumerate(zip(seg_stats, colors, labels)):
            start_idx = i * seg_size
            end_idx = min((i + 1) * seg_size, len(steps) - 1)
            ax.hlines(stat["mean"], steps[start_idx], steps[end_idx],
                      colors=color, linewidth=2.5, linestyles="--",
                      label=f"{label}: {stat['mean']:.3f}")

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Global Step")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    print(f"Chart saved to: {output_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze GRPO training logs")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Directory containing wandb CSV exports")
    parser.add_argument("--window", type=int, default=20,
                        help="Moving average window size (default: 20)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    # Load data
    print("=" * 60)
    print("Loading training logs")
    print("=" * 60)
    data = load_wandb_csvs(log_dir)
    print(f"  Found {len(data)} metrics: {list(data.keys())}")
    for name, d in data.items():
        print(f"  {name}: {len(d['steps'])} data points, "
              f"steps {d['steps'][0]}-{d['steps'][-1]}")

    # Segment statistics
    print("\n" + "=" * 60)
    print("Segment Statistics (split into 3 equal parts)")
    print("=" * 60)
    for name, d in data.items():
        seg = compute_segment_stats(d["values"])
        step_ranges = np.array_split(d["steps"], 3)
        print(f"\n  [{name}]")
        labels = ["First 1/3", "Mid 1/3  ", "Last 1/3 "]
        for label, s, sr in zip(labels, seg, step_ranges):
            print(f"    {label} (step {sr[0]:>5}-{sr[-1]:>5}): "
                  f"mean={s['mean']:>8.3f}  std={s['std']:>7.3f}  "
                  f"range=[{s['min']:>8.3f}, {s['max']:>8.3f}]")

    # Diagnostics
    print("\n" + "=" * 60)
    print("Diagnostics")
    print("=" * 60)
    findings = run_diagnostics(data)
    for f in findings:
        print(f"  {f}")

    # Plot
    print("\n" + "=" * 60)
    print("Generating charts")
    print("=" * 60)
    output_path = log_dir / "analysis.png"
    plot_metrics(data, args.window, output_path)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()

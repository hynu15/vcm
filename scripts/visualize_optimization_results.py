"""
Visualization script for SAC-X265 CRF optimization results.
Generates publication-quality plots for thesis visualization.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_summary(summary_csv_path):
    """Load optimization summary CSV."""
    df = pd.read_csv(summary_csv_path)
    return df


def plot_rate_distortion_psnr(df, output_dir):
    """
    Plot Rate-Distortion curve: Bitrate (x-axis) vs SA-PSNR/PSNR_Trad (y-axis).
    Highlights baseline config (23/32) and best config.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # SAC bitrate vs SA-PSNR
    scatter_sac = ax.scatter(
        df["sac_bitrate_mbps"],
        df["SA-PSNR"],
        s=150,
        alpha=0.6,
        c=df["delta_SA_PSNR_vs_Trad"],
        cmap="RdYlGn",
        edgecolors="black",
        linewidth=1.5,
        label="SAC configs",
    )

    # Traditional baseline (single point, multiple CRF configs should average)
    trad_unique = df[["trad_bitrate_mbps", "PSNR_Trad"]].drop_duplicates()
    ax.scatter(
        trad_unique["trad_bitrate_mbps"],
        trad_unique["PSNR_Trad"],
        s=300,
        marker="X",
        c="red",
        edgecolors="darkred",
        linewidth=2,
        label="Traditional baseline",
        zorder=5,
    )

    # Highlight baseline config (23/32)
    baseline = df[(df["crf_roi"] == 23) & (df["crf_non"] == 32)]
    if not baseline.empty:
        ax.scatter(
            baseline["sac_bitrate_mbps"].iloc[0],
            baseline["SA-PSNR"].iloc[0],
            s=400,
            marker="D",
            facecolors="none",
            edgecolors="blue",
            linewidth=2.5,
            label="Baseline (CRF 23/32)",
            zorder=6,
        )

    # Highlight best config
    best = df.iloc[0]
    ax.scatter(
        best["sac_bitrate_mbps"],
        best["SA-PSNR"],
        s=400,
        marker="*",
        c="gold",
        edgecolors="darkgreen",
        linewidth=2,
        label=f"Best (CRF {int(best['crf_roi'])}/{int(best['crf_non'])})",
        zorder=7,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter_sac, ax=ax)
    cbar.set_label("Δ SA-PSNR vs Trad (dB)", fontsize=11, fontweight="bold")

    ax.set_xlabel("Bitrate (Mbps)", fontsize=12, fontweight="bold")
    ax.set_ylabel("SA-PSNR (dB)", fontsize=12, fontweight="bold")
    ax.set_title("Rate-Distortion: SAC-X265 CRF Optimization", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

    plt.tight_layout()
    output_path = output_dir / "rd_curve_psnr.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[PLOT] Saved: {output_path}")
    plt.close()


def plot_rate_distortion_ssim(df, output_dir):
    """
    Plot Rate-Distortion curve: Bitrate (x-axis) vs SA-SSIM/SSIM_Trad (y-axis).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter_sac = ax.scatter(
        df["sac_bitrate_mbps"],
        df["SA-SSIM"],
        s=150,
        alpha=0.6,
        c=df["delta_SA_SSIM_vs_Trad"],
        cmap="RdYlGn",
        edgecolors="black",
        linewidth=1.5,
        label="SAC configs",
    )

    trad_unique = df[["trad_bitrate_mbps", "SSIM_Trad"]].drop_duplicates()
    ax.scatter(
        trad_unique["trad_bitrate_mbps"],
        trad_unique["SSIM_Trad"],
        s=300,
        marker="X",
        c="red",
        edgecolors="darkred",
        linewidth=2,
        label="Traditional baseline",
        zorder=5,
    )

    baseline = df[(df["crf_roi"] == 23) & (df["crf_non"] == 32)]
    if not baseline.empty:
        ax.scatter(
            baseline["sac_bitrate_mbps"].iloc[0],
            baseline["SA-SSIM"].iloc[0],
            s=400,
            marker="D",
            facecolors="none",
            edgecolors="blue",
            linewidth=2.5,
            label="Baseline (CRF 23/32)",
            zorder=6,
        )

    best = df.iloc[0]
    ax.scatter(
        best["sac_bitrate_mbps"],
        best["SA-SSIM"],
        s=400,
        marker="*",
        c="gold",
        edgecolors="darkgreen",
        linewidth=2,
        label=f"Best (CRF {int(best['crf_roi'])}/{int(best['crf_non'])})",
        zorder=7,
    )

    cbar = plt.colorbar(scatter_sac, ax=ax)
    cbar.set_label("Δ SA-SSIM vs Trad", fontsize=11, fontweight="bold")

    ax.set_xlabel("Bitrate (Mbps)", fontsize=12, fontweight="bold")
    ax.set_ylabel("SA-SSIM", fontsize=12, fontweight="bold")
    ax.set_title("Rate-Distortion: SAC-X265 CRF Optimization (SSIM)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

    plt.tight_layout()
    output_path = output_dir / "rd_curve_ssim.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[PLOT] Saved: {output_path}")
    plt.close()


def plot_pareto_front(df, output_dir):
    """
    Plot Pareto front combining SA-PSNR improvement (objective1) vs bitrate efficiency.
    Shows trade-offs between quality and bitrate.
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    # Prepare data for Pareto visualization
    bitrate_cost = df["bitrate_delta_mbps"].values
    quality_gain = df["delta_SA_PSNR_vs_Trad"].values
    score = df["objective_score"].values

    scatter = ax.scatter(
        bitrate_cost,
        quality_gain,
        s=200,
        alpha=0.7,
        c=score,
        cmap="viridis",
        edgecolors="black",
        linewidth=1.5,
    )

    # Mark baseline
    baseline = df[(df["crf_roi"] == 23) & (df["crf_non"] == 32)]
    if not baseline.empty:
        baseline_x = baseline["bitrate_delta_mbps"].iloc[0]
        baseline_y = baseline["delta_SA_PSNR_vs_Trad"].iloc[0]
        ax.scatter(
            baseline_x,
            baseline_y,
            s=400,
            marker="D",
            facecolors="none",
            edgecolors="blue",
            linewidth=2.5,
            label="Baseline (23/32)",
            zorder=5,
        )
        ax.annotate(
            "Baseline\n(23/32)",
            xy=(baseline_x, baseline_y),
            xytext=(10, -30),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="blue"),
        )

    # Mark best
    best = df.iloc[0]
    best_x = best["bitrate_delta_mbps"]
    best_y = best["delta_SA_PSNR_vs_Trad"]
    ax.scatter(
        best_x,
        best_y,
        s=500,
        marker="*",
        c="gold",
        edgecolors="darkgreen",
        linewidth=2.5,
        label=f"Best ({int(best['crf_roi'])}/{int(best['crf_non'])})",
        zorder=6,
    )
    ax.annotate(
        f"Best\n({int(best['crf_roi'])}/{int(best['crf_non'])})",
        xy=(best_x, best_y),
        xytext=(10, 20),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="darkgreen"),
    )

    # Add reference lines
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Objective Score\n(SA-PSNR gain - λ·Bitrate cost)", fontsize=10, fontweight="bold")

    ax.set_xlabel("Δ Bitrate vs Traditional (Mbps)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Δ SA-PSNR vs Traditional (dB)", fontsize=12, fontweight="bold")
    ax.set_title("Quality-Efficiency Trade-off: SAC-X265 Configurations", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)

    plt.tight_layout()
    output_path = output_dir / "pareto_trade_off.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[PLOT] Saved: {output_path}")
    plt.close()


def plot_config_heatmap(df, output_dir):
    """
    Plot heatmap of SA-PSNR for each CRF combination.
    Useful to visualize which (CRF_ROI, CRF_NON) pairs are best.
    """
    # Create pivot table for heatmap
    pivot_psnr = df.pivot_table(
        values="SA-PSNR",
        index="crf_non",
        columns="crf_roi",
        aggfunc="first",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot_psnr,
        annot=True,
        fmt=".2f",
        cmap="YlGn",
        cbar_kws={"label": "SA-PSNR (dB)"},
        linewidths=1,
        linecolor="black",
        ax=ax,
    )

    ax.set_xlabel("CRF_ROI (lower = higher quality)", fontsize=12, fontweight="bold")
    ax.set_ylabel("CRF_NON (lower = higher quality)", fontsize=12, fontweight="bold")
    ax.set_title("SA-PSNR Heatmap: CRF Combination Effects", fontsize=14, fontweight="bold")

    plt.tight_layout()
    output_path = output_dir / "config_heatmap_sapsnr.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[PLOT] Saved: {output_path}")
    plt.close()

    # Also create delta heatmap
    pivot_delta = df.pivot_table(
        values="delta_SA_PSNR_vs_Trad",
        index="crf_non",
        columns="crf_roi",
        aggfunc="first",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot_delta,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "Δ SA-PSNR vs Traditional (dB)"},
        linewidths=1,
        linecolor="black",
        ax=ax,
    )

    ax.set_xlabel("CRF_ROI", fontsize=12, fontweight="bold")
    ax.set_ylabel("CRF_NON", fontsize=12, fontweight="bold")
    ax.set_title("Quality Gain Heatmap: Δ SA-PSNR vs Traditional", fontsize=14, fontweight="bold")

    plt.tight_layout()
    output_path = output_dir / "config_heatmap_delta.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[PLOT] Saved: {output_path}")
    plt.close()


def plot_metric_comparison_bars(df, output_dir):
    """
    Bar chart comparing top configs vs baseline on multiple metrics.
    """
    # Select top 3 + baseline for comparison
    top_3 = df.head(3).copy()
    baseline = df[(df["crf_roi"] == 23) & (df["crf_non"] == 32)].copy()
    comparison = pd.concat([baseline, top_3], ignore_index=True)
    comparison["config_label"] = comparison.apply(
        lambda row: f"({int(row['crf_roi'])}/{int(row['crf_non'])})",
        axis=1,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Performance Comparison: Best Configurations vs Baseline", fontsize=16, fontweight="bold")

    # SA-PSNR
    ax = axes[0, 0]
    colors = ["red" if i == 0 else "green" for i in range(len(comparison))]
    ax.bar(comparison["config_label"], comparison["SA-PSNR"], color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("SA-PSNR (dB)", fontsize=11, fontweight="bold")
    ax.set_title("SA-PSNR Comparison", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Δ SA-PSNR vs Traditional
    ax = axes[0, 1]
    ax.bar(comparison["config_label"], comparison["delta_SA_PSNR_vs_Trad"], color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Δ SA-PSNR (dB)", fontsize=11, fontweight="bold")
    ax.set_title("Quality Gain over Traditional", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Bitrate
    ax = axes[1, 0]
    x = np.arange(len(comparison))
    width = 0.35
    ax.bar(x - width / 2, comparison["sac_bitrate_mbps"], width, label="SAC", alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.bar(x + width / 2, comparison["trad_bitrate_mbps"], width, label="Traditional", alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Bitrate (Mbps)", fontsize=11, fontweight="bold")
    ax.set_title("Bitrate Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["config_label"])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Objective Score
    ax = axes[1, 1]
    ax.bar(comparison["config_label"], comparison["objective_score"], color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Objective Score", fontsize=11, fontweight="bold")
    ax.set_title("Optimization Objective (higher = better)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "metric_comparison_bars.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[PLOT] Saved: {output_path}")
    plt.close()


def plot_score_ranking(df, output_dir):
    """
    Bar chart showing objective score ranking for all configs (sorted).
    """
    df_sorted = df.sort_values(by="objective_score", ascending=False).reset_index(drop=True)
    df_sorted["config_label"] = df_sorted.apply(
        lambda row: f"CRF {int(row['crf_roi'])}/{int(row['crf_non'])}",
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["gold" if i == 0 else ("red" if row["crf_roi"] == 23 and row["crf_non"] == 32 else "steelblue")
              for i, row in df_sorted.iterrows()]

    bars = ax.barh(df_sorted["config_label"], df_sorted["objective_score"], color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

    # Add value labels
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax.text(row["objective_score"] + 0.02, i, f"{row['objective_score']:.3f}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Objective Score", fontsize=12, fontweight="bold")
    ax.set_title("Configuration Ranking by Objective Score", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Custom legend
    legend_elements = [
        mpatches.Patch(facecolor="gold", edgecolor="black", label="Best Config"),
        mpatches.Patch(facecolor="red", edgecolor="black", label="Baseline (23/32)"),
        mpatches.Patch(facecolor="steelblue", edgecolor="black", label="Other Configs"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "score_ranking.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[PLOT] Saved: {output_path}")
    plt.close()


def generate_all_plots(summary_csv_path, output_dir=None):
    """
    Generate all visualization plots.
    """
    if output_dir is None:
        output_dir = Path(summary_csv_path).parent / "plots"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading summary from: {summary_csv_path}")
    df = load_summary(summary_csv_path)

    print(f"[INFO] Configs loaded: {len(df)}")
    print(f"[INFO] Output directory: {output_dir}")

    # Generate all plots
    print("\n[PLOTTING] Generating Rate-Distortion curves...")
    plot_rate_distortion_psnr(df, output_dir)
    plot_rate_distortion_ssim(df, output_dir)

    print("[PLOTTING] Generating Pareto trade-off chart...")
    plot_pareto_front(df, output_dir)

    print("[PLOTTING] Generating configuration heatmaps...")
    plot_config_heatmap(df, output_dir)

    print("[PLOTTING] Generating comparison charts...")
    plot_metric_comparison_bars(df, output_dir)
    plot_score_ranking(df, output_dir)

    print(f"\n[SUCCESS] All plots saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Visualize SAC-X265 CRF optimization results")
    parser.add_argument("summary_csv", type=str, help="Path to crf_optimization_summary.csv")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for plots (default: plots/ in same dir)")
    args = parser.parse_args()

    summary_path = Path(args.summary_csv)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")

    generate_all_plots(str(summary_path), args.output_dir)


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

from utils.roofline import ridge_point


# ---------------------------------------------------------------------------
# Theme & save helpers
# ---------------------------------------------------------------------------

def setup_theme(style: str = "whitegrid") -> None:
    """Apply a consistent seaborn style."""
    import seaborn as sns

    sns.set_theme(style=style)


def save_show(fig, path: str | None = None, dpi: int = 150, bbox_inches: str = "tight") -> None:
    """Save figure (when *path* is given) and display it."""
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    plt.show()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _format_seqlen_axis(
    ax, ticks, *, label: str = "Sequence Length (N)", fontsize: int = 11,
) -> None:
    """Log-2 x-axis with scalar tick labels."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(label, fontsize=fontsize)


def _add_oom_markers(
    ax, x_values, y_pos, *,
    color: str = "tab:red", text: str = "Ref OOM",
    rotation: int = 90, va: str = "center",
) -> None:
    """Draw vertical dashed OOM lines with labels."""
    for x in x_values:
        ax.axvline(x=x, color=color, linestyle="--", alpha=0.45, linewidth=1.2)
        ax.text(
            x * 1.03, y_pos, text,
            color=color, fontsize=9, rotation=rotation, va=va,
        )


# ---------------------------------------------------------------------------
# Roofline helpers
# ---------------------------------------------------------------------------

def draw_roofline(
    ax,
    peak_tflops: float,
    peak_bw_gbs: float,
    label: str,
    color: str,
    linewidth: float = 2.2,
):
    """Draw roofline and ridge marker on a matplotlib axis."""
    ai = np.logspace(-1, 4, 800)
    mem = (peak_bw_gbs / 1e3) * ai
    comp = np.full_like(ai, peak_tflops)
    attainable = np.minimum(mem, comp)
    ax.plot(ai, attainable, color=color, linewidth=linewidth, label=label)
    ridge = ridge_point(peak_tflops, peak_bw_gbs)
    ax.axvline(ridge, color=color, linestyle=":", alpha=0.35, linewidth=1.0)
    return ridge


# ---------------------------------------------------------------------------
# Dual-pass line plots (fwd/bwd comparison across configs)
# ---------------------------------------------------------------------------

def plot_dual_pass_lines(
    df,
    seq_col: str,
    seq_ticks,
    configs,
    config_col: str,
    naive_col_fmt: str,
    flash_col_fmt: str,
    naive_palette: dict,
    flash_palette: dict,
    *,
    ylabel: str,
    title_fmt: str,
    output_path: str | None = None,
    ridges: list | None = None,
    figsize: tuple = (14, 5),
) -> None:
    """Two-panel (fwd/bwd) line comparison of naive vs flash across configs.

    Parameters
    ----------
    naive_col_fmt / flash_col_fmt : str with ``{direction}`` placeholder,
        e.g. ``"naive_{direction}_ms"``.
    title_fmt : str with ``{direction}`` placeholder,
        e.g. ``"{direction} pass runtime"``.
    ridges : optional list of ``(value, color, label)`` tuples drawn as
        horizontal reference lines.
    """
    setup_theme("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, direction, dir_title in [
        (axes[0], "fwd", "Forward"),
        (axes[1], "bwd", "Backward"),
    ]:
        for cfg in configs:
            sub = df[df[config_col] == cfg].sort_values(seq_col)
            ax.plot(
                sub[seq_col],
                sub[naive_col_fmt.format(direction=direction)],
                "o--", color=naive_palette[cfg], alpha=0.85,
                label=f"Naive  {config_col}={cfg}",
            )
            ax.plot(
                sub[seq_col],
                sub[flash_col_fmt.format(direction=direction)],
                "s-", color=flash_palette[cfg], linewidth=2,
                label=f"Flash  {config_col}={cfg}",
            )

        if ridges:
            for value, color, lbl in ridges:
                ax.axhline(value, color=color, linestyle=":", linewidth=1.2)
                ax.text(
                    seq_ticks[0] * 1.05, value * 1.10,
                    f"{lbl} \u2248 {value:.0f}", fontsize=8, color=color,
                )

        _format_seqlen_axis(ax, seq_ticks, label=f"Sequence length {seq_col}")
        ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(
            title_fmt.format(direction=dir_title),
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=8, ncol=2)

    save_show(fig, output_path)
    if output_path:
        print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Forward benchmark (runtime + memory, two panels)
# ---------------------------------------------------------------------------

def plot_fwd_benchmark(
    df,
    *,
    seq_col: str = "N",
    ref_time_col: str = "ref_ms",
    flash_time_col: str = "triton_ms",
    ref_mem_col: str = "ref_mem_mb",
    flash_mem_col: str = "tri_mem_mb",
    oom_col: str = "ref_ms",
    oom_value: str = "OOM",
    ref_label: str = "PyTorch Reference",
    flash_label: str = "FlashAttention (Triton)",
    output_path: str | None = None,
    figsize: tuple = (14, 6),
) -> None:
    """Two-panel forward benchmark: runtime + peak memory vs sequence length."""
    setup_theme("whitegrid")

    df_valid = df[df[oom_col] != oom_value].copy()
    df_valid[ref_time_col] = df_valid[ref_time_col].astype(float)
    df_valid[ref_mem_col] = df_valid[ref_mem_col].astype(float)
    df_plot = df.copy()
    df_plot[flash_time_col] = df_plot[flash_time_col].astype(float)
    df_plot[flash_mem_col] = df_plot[flash_mem_col].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ticks = df_plot[seq_col].values
    oom_ns = df[df[oom_col] == oom_value][seq_col].values

    # ── Runtime panel ─────────────────────────────────────────────
    axes[0].plot(
        df_valid[seq_col], df_valid[ref_time_col],
        marker="o", linewidth=2, label=ref_label, color="tab:red",
    )
    axes[0].plot(
        df_plot[seq_col], df_plot[flash_time_col],
        marker="s", linewidth=2, label=flash_label, color="tab:blue",
    )
    if len(oom_ns):
        oom_y = df_plot.loc[df_plot[seq_col].isin(oom_ns), flash_time_col].max()
        _add_oom_markers(axes[0], oom_ns, oom_y, va="bottom")
    axes[0].set_title("Forward Pass Runtime", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Execution Time (ms)", fontsize=11)
    _format_seqlen_axis(axes[0], ticks)
    axes[0].legend(fontsize=10)

    # ── Memory panel ──────────────────────────────────────────────
    axes[1].plot(
        df_valid[seq_col], df_valid[ref_mem_col],
        marker="o", linewidth=2, label=ref_label, color="tab:red",
    )
    axes[1].plot(
        df_plot[seq_col], df_plot[flash_mem_col],
        marker="s", linewidth=2, label=flash_label, color="tab:blue",
    )
    oom_mem_ns = df[df[ref_mem_col] == oom_value][seq_col].values
    if len(oom_mem_ns):
        oom_mem_y = df_plot.loc[df_plot[seq_col].isin(oom_mem_ns), flash_mem_col].max()
        _add_oom_markers(axes[1], oom_mem_ns, oom_mem_y, va="bottom")
    axes[1].set_title("Peak Memory Usage", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Memory (MB)", fontsize=11)
    _format_seqlen_axis(axes[1], ticks)
    axes[1].legend(fontsize=10)

    save_show(fig, output_path)
    if output_path:
        print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Backward benchmark (runtime + memory + speedup, three panels)
# ---------------------------------------------------------------------------

def plot_bwd_benchmark(
    df_bwd,
    *,
    seq_col: str = "N",
    ref_time_col: str = "ref_ms",
    flash_time_col: str = "triton_ms",
    ref_mem_col: str = "ref_mem_mb",
    flash_mem_col: str = "tri_mem_mb",
    speedup_col: str = "speedup",
    score_matrix_col: str | None = "score_matrix_mb",
    oom_col: str = "ref_ms",
    oom_value: str = "OOM",
    ref_label: str = "PyTorch Reference",
    flash_label: str = "FlashAttention (Triton)",
    memory_annotate_n: int | None = 8192,
    output_path: str = "backward_pass_benchmark.png",
    figsize: tuple = (18, 6),
) -> None:
    """Three-panel backward benchmark: runtime, memory, speedup bars."""
    setup_theme("whitegrid")

    df_valid = df_bwd[df_bwd[oom_col] != oom_value].copy()
    for col in [ref_time_col, ref_mem_col, flash_time_col, flash_mem_col, speedup_col]:
        df_valid[col] = df_valid[col].astype(float)
    df_all = df_bwd.copy()
    df_all[flash_time_col] = df_all[flash_time_col].astype(float)
    df_all[flash_mem_col] = df_all[flash_mem_col].astype(float)
    if score_matrix_col:
        df_all[score_matrix_col] = df_all[score_matrix_col].astype(float)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    ticks = df_all[seq_col].values
    oom_ns = df_bwd[df_bwd[oom_col] == oom_value][seq_col].values

    # ── Panel 1: Runtime ──────────────────────────────────────────
    axes[0].plot(
        df_valid[seq_col], df_valid[ref_time_col],
        marker="o", linewidth=2, label=ref_label, color="tab:red",
    )
    axes[0].plot(
        df_all[seq_col], df_all[flash_time_col],
        marker="s", linewidth=2, label=flash_label, color="tab:blue",
    )
    _add_oom_markers(axes[0], oom_ns, df_all[flash_time_col].max() * 0.62)
    axes[0].set_title("Backward Pass Runtime", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Time (ms)", fontsize=11)
    _format_seqlen_axis(axes[0], ticks)
    axes[0].legend(fontsize=10)

    # ── Panel 2: Memory ───────────────────────────────────────────
    axes[1].plot(
        df_valid[seq_col], df_valid[ref_mem_col],
        marker="o", linewidth=2, label=ref_label, color="tab:red",
    )
    axes[1].plot(
        df_all[seq_col], df_all[flash_mem_col],
        marker="s", linewidth=2, label=flash_label, color="tab:blue",
    )
    if score_matrix_col:
        axes[1].plot(
            df_all[seq_col], df_all[score_matrix_col],
            marker="^", linewidth=1.5, linestyle="--", color="gray",
            label="N\u00d7N score matrix (theoretical)",
        )
    if memory_annotate_n and memory_annotate_n in df_valid[seq_col].values:
        row = df_valid[df_valid[seq_col] == memory_annotate_n].iloc[0]
        reduction = row[ref_mem_col] / row[flash_mem_col]
        axes[1].annotate(
            f"{reduction:.0f}\u00d7 less memory\n"
            f"({flash_label} @ N={memory_annotate_n})",
            xy=(memory_annotate_n, row[flash_mem_col]),
            xytext=(memory_annotate_n / 2.2, row[flash_mem_col] * 6),
            fontsize=9, color="tab:blue",
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.2),
        )
    axes[1].set_title("Backward Pass Peak Memory", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Memory (MB)", fontsize=11)
    axes[1].set_yscale("log", base=2)
    _format_seqlen_axis(axes[1], ticks)
    axes[1].get_yaxis().set_major_formatter(plt.ScalarFormatter())
    axes[1].legend(fontsize=10)

    # ── Panel 3: Speedup bars ─────────────────────────────────────
    all_ns = df_all[seq_col].tolist()
    x_pos = np.arange(len(all_ns))
    speedups, colors, hatches = [], [], []
    for _, row in df_bwd.iterrows():
        if row[oom_col] == oom_value:
            speedups.append(0)
            colors.append("lightgray")
            hatches.append("////")
        else:
            speedups.append(float(row[speedup_col]))
            colors.append("tab:blue")
            hatches.append("")

    peak_speedup = max((s for s in speedups if s > 0), default=2.0)
    y_max = max(peak_speedup * 1.3, 2.6)
    axes[2].bar(
        x_pos, [s if s > 0 else y_max * 0.96 for s in speedups],
        color=colors, hatch=hatches, alpha=0.8, edgecolor="white",
    )
    axes[2].axhline(y=1.0, color="tab:red", linestyle="--", linewidth=1.5)
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels([str(n) for n in all_ns], fontsize=10)
    axes[2].set_title("Backward Speedup", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Sequence Length (N)", fontsize=11)
    axes[2].set_ylabel("Speedup (\u00d7)", fontsize=11)
    axes[2].set_ylim(0, y_max)

    for i, (s, h) in enumerate(zip(speedups, hatches)):
        if h:
            axes[2].text(
                i, y_max * 0.98, "Ref\nOOM", ha="center",
                fontsize=9, color="gray", fontweight="bold",
            )
        else:
            axes[2].text(
                i, s + 0.05, f"{s:.2f}\u00d7", ha="center",
                fontsize=10, fontweight="bold", color="tab:blue",
            )

    oom_patch = mpatches.Patch(
        facecolor="lightgray", hatch="////", edgecolor="gray", label="Ref OOM",
    )
    axes[2].legend(
        handles=[
            plt.Line2D([0], [0], color="tab:red", linestyle="--", label="1\u00d7 baseline"),
            oom_patch,
        ],
        fontsize=10,
    )

    save_show(fig, output_path)
    if output_path:
        print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Roofline composite plots
# ---------------------------------------------------------------------------

def plot_roofline_scatter_single(
    df,
    ai_col: str,
    perf_col: str,
    kernel_col: str,
    seq_col: str,
    ridge_point_value: float,
    peak_tflops: float,
    roof_label: str,
    title: str,
    output_path: str,
    xlim=(0.5, 2000),
    ylim_min: float = 1e-3,
) -> None:
    """Single-panel roofline scatter with naive/flash points and N labels."""
    setup_theme("whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7))

    ai_range = np.logspace(-1, 4, 500)
    peak_bw_gbs = (peak_tflops * 1e3) / ridge_point_value
    roof_mem = (peak_bw_gbs / 1e3) * ai_range
    roof_comp = np.full_like(ai_range, peak_tflops)
    attainable = np.minimum(roof_mem, roof_comp)
    ax.plot(ai_range, attainable, color="black", linewidth=2.5, label=roof_label)
    ax.axvline(x=ridge_point_value, color="black", linestyle=":", alpha=0.4, linewidth=1.2)
    ax.text(
        ridge_point_value * 1.08,
        peak_tflops * 0.85,
        f"Ridge\n{ridge_point_value:.0f} FLOPs/B",
        fontsize=9,
        color="gray",
    )

    colors_map = {"Naive": "tab:red", "Flash": "tab:blue"}
    markers_map = {"Naive": "o", "Flash": "s"}
    for _, row in df.iterrows():
        kernel = row[kernel_col]
        ai = row[ai_col]
        tflops = row[perf_col]
        color = colors_map.get(kernel, "tab:gray")
        marker = markers_map.get(kernel, "o")
        ax.scatter(ai, tflops, color=color, marker=marker, s=90, zorder=5)
        ax.annotate(
            f"N={int(row[seq_col])}",
            xy=(ai, tflops),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color=color,
            alpha=0.85,
        )

    ax.axvspan(0.1, ridge_point_value, alpha=0.04, color="red", label="Memory-bound region")
    ax.axvspan(ridge_point_value, 1e4, alpha=0.04, color="blue", label="Compute-bound region")

    legend_handles = [
        plt.Line2D([0], [0], color="black", linewidth=2, label=roof_label),
        plt.scatter([], [], color="tab:red", marker="o", s=70, label="Naive Attention"),
        plt.scatter([], [], color="tab:blue", marker="s", s=70, label="FlashAttention"),
        mpatches.Patch(color="red", alpha=0.15, label="Memory-bound"),
        mpatches.Patch(color="blue", alpha=0.15, label="Compute-bound"),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="upper left")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)", fontsize=12)
    ax.set_ylabel("Achieved Performance (TFLOP/s)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(*xlim)
    ax.set_ylim(ylim_min, peak_tflops * 2)
    save_show(fig, output_path)
    print(f"Saved: {output_path}")


def plot_roofline_dual_pass(
    df,
    cs,
    peak_tflops: float,
    peak_bw_gbs: float,
    ridge_flops_per_byte: float,
    naive_palette: dict,
    flash_palette: dict,
    output_path: str,
    *,
    gpu_roof_label: str = "A100 roof",
    title_gpu: str = "A100",
) -> None:
    """Two-panel roofline plot for forward/backward results."""
    setup_theme("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    for ax, direction, title in [(axes[0], "fwd", "Forward"), (axes[1], "bwd", "Backward")]:
        draw_roofline(ax, peak_tflops, peak_bw_gbs, gpu_roof_label, color="black")

        for c_val in cs:
            sub = df[df["C"] == c_val].sort_values("T")
            ax.scatter(
                sub[f"AI_naive_{direction}"],
                sub[f"naive_{direction}_TFLOPS"],
                marker="o",
                color=naive_palette[c_val],
                s=85,
                edgecolor="black",
                linewidths=0.6,
                label=f"Naive C={c_val}",
            )
            ax.scatter(
                sub[f"AI_flash_{direction}"],
                sub[f"flash_{direction}_TFLOPS"],
                marker="s",
                color=flash_palette[c_val],
                s=85,
                edgecolor="black",
                linewidths=0.6,
                label=f"Flash C={c_val}",
            )
            for _, row in sub.iterrows():
                flash_tflops = row[f"flash_{direction}_TFLOPS"]
                if not np.isnan(flash_tflops):
                    ax.annotate(
                        f"T={int(row['T'])}",
                        xy=(row[f"AI_flash_{direction}"], flash_tflops),
                        xytext=(4, 3),
                        textcoords="offset points",
                        fontsize=7,
                        color=flash_palette[c_val],
                    )

        ax.axvspan(0.1, ridge_flops_per_byte, alpha=0.04, color="red")
        ax.axvspan(ridge_flops_per_byte, 1e4, alpha=0.04, color="blue")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.5, 5000)
        ax.set_ylim(1e-2, peak_tflops * 2)
        ax.set_xlabel("Arithmetic intensity (FLOPs / byte)")
        ax.set_ylabel("Achieved TFLOP/s")
        ax.set_title(f"{title} pass roofline - Pallas Flash vs Naive on {title_gpu}")
        ax.legend(fontsize=7, ncol=2, loc="lower right")

    save_show(fig, output_path)
    print(f"Saved: {output_path}")


def plot_roofline_trajectory_and_efficiency(
    df_roof,
    ridge_point_value: float,
    peak_tflops: float,
    peak_bw_gbs: float,
    output_path: str,
    *,
    roofline_label: str = "A100 roofline",
) -> None:
    """Two-panel roofline trajectory + achieved-vs-ceiling bar chart."""
    setup_theme("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ax = axes[0]

    ai_range = np.logspace(-0.5, 4.5, 1000)
    memory_perf = (peak_bw_gbs / 1e3) * ai_range
    compute_perf = np.full_like(ai_range, peak_tflops)
    attainable = np.minimum(memory_perf, compute_perf)
    ax.plot(ai_range, attainable, color="black", linewidth=2.5, zorder=4, label=roofline_label)
    ax.axvline(x=ridge_point_value, color="black", linestyle=":", alpha=0.35, linewidth=1.2)
    ax.text(ridge_point_value * 1.06, 0.006, f"Ridge\n{ridge_point_value:.0f}", fontsize=8.5, color="gray", va="bottom")

    ax.axvspan(0.3, ridge_point_value, alpha=0.05, color="red")
    ax.axvspan(ridge_point_value, 2e4, alpha=0.05, color="blue")
    ax.text(2.0, 0.004, "Memory-bound", fontsize=9, color="firebrick", alpha=0.7)
    ax.text(1100, 0.004, "Compute-bound", fontsize=9, color="steelblue", alpha=0.7)

    seq_lens = sorted(df_roof["N"].unique())
    cmap_naive = cm.Reds
    cmap_flash = cm.Blues
    n_steps = len(seq_lens)

    for kernel, cmap_obj, marker in [("Naive", cmap_naive, "o"), ("Flash", cmap_flash, "s")]:
        rows = df_roof[df_roof["kernel"] == kernel].sort_values("N")
        ais = rows["AI (FLOPs/B)"].values
        tflops = rows["achieved"].values
        ns = rows["N"].values
        for idx, (ai, tf, seq_len) in enumerate(zip(ais, tflops, ns)):
            color = cmap_obj(0.4 + 0.5 * idx / max(1, n_steps - 1))
            ax.scatter(ai, tf, color=color, marker=marker, s=100, zorder=5, edgecolors="white", linewidths=0.5)
            ax.annotate(
                f"N={int(seq_len)}",
                xy=(ai, tf),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=7.5,
                color=color,
            )
        for idx in range(len(ais) - 1):
            color = cmap_obj(0.5 + 0.5 * idx / max(1, n_steps - 1))
            ax.annotate(
                "",
                xy=(ais[idx + 1], tflops[idx + 1]),
                xytext=(ais[idx], tflops[idx]),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2, "alpha": 0.6},
            )

    legend_handles = [
        plt.Line2D([0], [0], color="black", lw=2, label=roofline_label),
        plt.scatter([], [], color="tab:red", marker="o", s=70, label="Naive (N up = darker)"),
        plt.scatter([], [], color="tab:blue", marker="s", s=70, label="Flash (N up = darker)"),
        mpatches.Patch(color="red", alpha=0.15, label="Memory-bound"),
        mpatches.Patch(color="blue", alpha=0.15, label="Compute-bound"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.3, 2e4)
    ax.set_ylim(3e-3, peak_tflops * 2)
    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)", fontsize=11)
    ax.set_ylabel("Achieved Performance (TFLOP/s)", fontsize=11)
    ax.set_title("Roofline Trajectory as N Increases", fontsize=12, fontweight="bold")

    ax2 = axes[1]
    x = np.arange(len(seq_lens))
    width = 0.35
    for kernel in ["Naive", "Flash"]:
        rows = df_roof[df_roof["kernel"] == kernel].sort_values("N")
        efficiency = rows["efficiency_%"].values
        attainable_vals = rows["attainable"].values
        achieved_vals = rows["achieved"].values
        color = "tab:red" if kernel == "Naive" else "tab:blue"
        offset = -width / 2 if kernel == "Naive" else width / 2

        ax2.bar(
            x + offset,
            attainable_vals,
            width,
            label=f"{kernel} ceiling",
            color=color,
            alpha=0.25,
            edgecolor=color,
            linewidth=1.2,
        )
        ax2.bar(x + offset, achieved_vals, width, label=f"{kernel} achieved", color=color, alpha=0.85)

        for idx, (ach, eff) in enumerate(zip(achieved_vals, efficiency)):
            ax2.text(
                x[idx] + offset,
                ach + 0.3,
                f"{eff:.1f}%",
                ha="center",
                fontsize=7.5,
                color=color,
                fontweight="bold",
            )

    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(n) for n in seq_lens], fontsize=10)
    ax2.set_xlabel("Sequence Length (N)", fontsize=11)
    ax2.set_ylabel("Performance (TFLOP/s, log scale)", fontsize=11)
    ax2.set_title("Achieved vs. Attainable Roofline Ceiling", fontsize=12, fontweight="bold")
    handles, labels = ax2.get_legend_handles_labels()
    seen = {}
    for h, lbl in zip(handles, labels):
        if lbl not in seen:
            seen[lbl] = h
    ax2.legend(seen.values(), seen.keys(), fontsize=9, loc="upper left")

    save_show(fig, output_path)
    print(f"Saved: {output_path}")

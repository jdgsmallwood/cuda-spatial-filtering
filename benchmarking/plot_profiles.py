"""Parse ncu (Nsight Compute) CSV output and generate per-kernel profile figures.

ncu is run via benchmarking/run_profile.sh with:
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis_Chart

The CSV format has one row per (kernel, metric):
  "ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",
  "Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"

Figures produced in OUTPUT_DIR:
  profile_duration.png          — per-kernel GPU execution time (μs)
  profile_tensor_core.png       — tensor-core pipe utilization (%)
  profile_tensor_precision.png  — tensor pipe split by precision: HMMA (fp16/bf16)
                                   vs IMMA (int8/int4/binary) -- requires the
                                   explicit --metrics run_profile.sh now requests
  profile_occupancy.png         — achieved vs theoretical warp occupancy (%)
  profile_throughput.png        — SM and DRAM throughput as % of roofline peak
  profile_memory_bw.png         — L1/L2/DRAM bandwidth breakdown (% of peak)
  profile_stall_reasons.png     — per-kernel warp stall-reason breakdown (%),
                                   for finding *why* a kernel is bottlenecked
  profile_overview.png          — 3×2 summary combining the core five panels

Usage:
  python plot_profiles.py --results-dir benchmarking/20260707_120000_profile
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
except ImportError:
    sys.exit("matplotlib / numpy not found — pip install matplotlib numpy")

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas not found — pip install pandas")


# ---------------------------------------------------------------------------
# ncu CSV metric name → friendly label mappings
# ---------------------------------------------------------------------------

# SpeedOfLight section metric names (as they appear in ncu CSV "Metric Name" col).
# Different ncu versions use different friendly-name spellings for the same
# underlying counter, so each set lists every variant seen in practice.
SOL_SM_NAMES   = {"SM [%]", "Compute (SM) Throughput",
                  "sm__throughput.avg.pct_of_peak_sustained_elapsed"}
SOL_MEM_NAMES  = {"Memory [%]", "DRAM Throughput",
                  "dram__throughput.avg.pct_of_peak_sustained_elapsed"}

# Occupancy section
OCC_ACHIEVED   = {"Achieved Occupancy", "sm__warps_active.avg.pct_of_peak_sustained_active"}
OCC_THEORETICAL = {"Theoretical Occupancy", "theoretical_occupancy"}

# MemoryWorkloadAnalysis_Chart section — L1/L2/DRAM
MEM_L1_NAMES   = {"L1/TEX Cache [%]", "L1/TEX Cache Throughput",
                  "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"}
MEM_L2_NAMES   = {"L2 Cache [%]", "L2 Cache Throughput",
                  "lts__throughput.avg.pct_of_peak_sustained_elapsed"}
MEM_DRAM_NAMES = {"DRAM [%]", "DRAM Throughput",
                  "dram__throughput.avg.pct_of_peak_sustained_elapsed"}

# Tensor-core pipe utilization -- only present when ncu was
# run with a section that samples the tensor pipe (not always captured).
TENSOR_PCT_NAMES = {"Tensor Core Utilization", "Compute (Tensor) Throughput",
                    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
                    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"}

# Precision-specific tensor-pipe duty cycle -- HMMA = fp16/bf16, IMMA =
# int8/int4/binary. Only present if run_profile.sh's explicit --metrics list
# requested them (the SpeedOfLight tensor_pct above doesn't split by precision).
TENSOR_HMMA_NAMES = {"Tensor (FP16) Pipe Utilization",
                     "sm__pipe_tensor_op_hmma_cycles_active_v2.avg.pct_of_peak_sustained_elapsed"}
TENSOR_IMMA_NAMES = {"Tensor (INT8) Pipe Utilization",
                     "sm__pipe_tensor_op_imma_cycles_active_v2.avg.pct_of_peak_sustained_elapsed"}

# WarpStateStats stall-reason breakdown -- % of stall cycles attributable to
# each reason, for diagnosing *why* a kernel has low achieved occupancy/SM
# throughput. "selected" (issuing, not stalled) is intentionally excluded.
STALL_REASONS = [
    "barrier", "branch_resolving", "dispatch_stall", "drain", "imc_miss",
    "lg_throttle", "long_scoreboard", "math_pipe_throttle", "membar",
    "mio_throttle", "misc", "no_instruction", "not_selected",
    "short_scoreboard", "sleeping", "tex_throttle", "wait",
]

def _stall_names(reason: str) -> set:
    title = reason.replace("_", " ").title()
    return {f"Stall {title}",
            f"smsp__average_warps_issue_stalled_{reason}_per_issue_active.pct"}

def _wide_stall_col(reason: str) -> str:
    return f"smsp__average_warps_issue_stalled_{reason}_per_issue_active.pct"

# Metric unit → microseconds, for the "Duration" metric row (used when the
# CSV has no dedicated "Kernel Time" column -- newer ncu versions report
# kernel duration as a Metric Name="Duration" row instead).
_DURATION_UNIT_TO_US = {
    "ns": 1e-3, "nsecond": 1e-3,
    "us": 1.0, "usecond": 1.0,
    "ms": 1e3, "msecond": 1e3,
    "s": 1e6, "second": 1e6,
}

# ---------------------------------------------------------------------------
# Wide-format ncu CSV: `ncu --csv --page raw` exports one row per kernel
# launch with every metric as its own column (raw metric names), rather than
# one row per (kernel, metric) pair. Column names below are the raw ncu
# metric identifiers, found by inspecting the actual export -- this ncu
# version didn't emit a plain "dram__throughput...pct_of_peak" column, so the
# combined compute-memory-subsystem throughput is used as the closest
# equivalent for both the SoL-memory and DRAM-bandwidth slots.
WIDE_DURATION_COL       = "gpu__time_duration.sum"
WIDE_SM_PCT_COL         = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
WIDE_DRAM_PCT_COL       = "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
WIDE_OCC_ACHIEVED_COL   = "sm__warps_active.avg.pct_of_peak_sustained_active"
WIDE_OCC_THEORETICAL_COL = "sm__maximum_warps_per_active_cycle_pct"
WIDE_L1_PCT_COL         = "l1tex__throughput.avg.pct_of_peak_sustained_active"
WIDE_L2_PCT_COL         = "lts__throughput.avg.pct_of_peak_sustained_elapsed"
WIDE_TENSOR_PCT_COL     = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
WIDE_HMMA_PCT_COL       = "sm__pipe_tensor_op_hmma_cycles_active_v2.avg.pct_of_peak_sustained_elapsed"
WIDE_IMMA_PCT_COL       = "sm__pipe_tensor_op_imma_cycles_active_v2.avg.pct_of_peak_sustained_elapsed"


def _match(metric_name: str, candidates: set) -> bool:
    return metric_name in candidates


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _clean_kernel_name(raw: str) -> str:
    """Shorten a CUDA kernel name to something plot-legible."""
    # Strip trailing (void*, ...) argument lists
    raw = re.sub(r'\(.*\)', '', raw).strip()
    # Strip template angle brackets content
    raw = re.sub(r'<[^>]*>', '', raw).strip()
    # Strip leading namespace / path separators
    parts = re.split(r'::', raw)
    name = parts[-1].strip()
    # Truncate very long names
    if len(name) > 40:
        name = name[:37] + "…"
    return name or raw


def parse_ncu_csv(path: Path) -> pd.DataFrame:
    """Return a DataFrame with one row per (kernel_id, metric_name), or a wide
    DataFrame (one row per kernel launch, metrics as columns) with a
    `_wide_format` attr set to True -- see build_kernel_summary_wide()."""
    # ncu may prepend lines starting with "==" before the CSV header — skip them.
    lines = []
    with open(path, encoding="utf-8", errors="replace") as f:
        in_csv = False
        for line in f:
            if not in_csv:
                if line.startswith('"ID"') or line.startswith("ID,"):
                    in_csv = True
                    lines.append(line)
            else:
                lines.append(line)

    if not lines:
        print(f"WARNING: no CSV data found in {path}", file=sys.stderr)
        return pd.DataFrame()

    from io import StringIO

    # Wide-format export (`ncu --csv --page raw`): the row right after the
    # header is a units row (e.g. "%", "usecond") rather than kernel data --
    # detect it and drop it.
    header_cols = [c.strip().strip('"') for c in
                   pd.read_csv(StringIO(lines[0])).columns]
    is_wide = "Metric Name" not in header_cols and "Kernel Name" in header_cols

    if is_wide:
        data_lines = [lines[0]] + lines[2:] if len(lines) > 1 else lines
        df = pd.read_csv(StringIO("".join(data_lines)), dtype=str)
        df.columns = [c.strip().strip('"') for c in df.columns]
        df.attrs["wide_format"] = True
        df["kernel_short"] = df["Kernel Name"].apply(_clean_kernel_name)
        return df

    df = pd.read_csv(StringIO("".join(lines)), dtype=str)

    # Normalise column names: strip quotes and whitespace
    df.columns = [c.strip().strip('"') for c in df.columns]

    # Required columns. "Kernel Time" is an older-ncu-version column giving
    # duration directly; newer versions instead report a Metric Name="Duration"
    # row per kernel (handled in build_kernel_summary), so it's optional here.
    needed = {"ID", "Kernel Name", "Metric Name", "Metric Value"}
    missing = needed - set(df.columns)
    if missing:
        print(f"WARNING: ncu CSV missing columns: {missing}\n"
              f"  Found: {list(df.columns)}", file=sys.stderr)
        return pd.DataFrame()

    # Metric Value can contain thousands separators (e.g. "6,655" for cycle
    # counts) which pd.to_numeric chokes on -- strip them before conversion.
    df["Metric Value"] = pd.to_numeric(
        df["Metric Value"].str.replace(",", "", regex=False), errors="coerce")
    df["kernel_short"] = df["Kernel Name"].apply(_clean_kernel_name)
    df.attrs["wide_format"] = False

    return df


# ---------------------------------------------------------------------------
# Extract per-kernel summary
# ---------------------------------------------------------------------------

def build_kernel_summary_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Same output schema as build_kernel_summary(), but source is a wide
    ncu CSV (one row per kernel launch, metrics as columns)."""
    if df.empty:
        return pd.DataFrame()

    def col(name):
        if name not in df.columns:
            return pd.Series(float("nan"), index=df.index)
        return pd.to_numeric(df[name], errors="coerce")

    summary = pd.DataFrame({
        "kernel_id":       df["ID"],
        "kernel_name":     df["Kernel Name"],
        "kernel_short":    df["kernel_short"],
        "kernel_time":     None,
        "duration_us":     col(WIDE_DURATION_COL),
        "sm_pct":          col(WIDE_SM_PCT_COL),
        "dram_pct":        col(WIDE_DRAM_PCT_COL),
        "occ_achieved":    col(WIDE_OCC_ACHIEVED_COL),
        "occ_theoretical": col(WIDE_OCC_THEORETICAL_COL),
        "l1_pct":          col(WIDE_L1_PCT_COL),
        "l2_pct":          col(WIDE_L2_PCT_COL),
        "tensor_pct":      col(WIDE_TENSOR_PCT_COL),
        "hmma_pct":        col(WIDE_HMMA_PCT_COL),
        "imma_pct":        col(WIDE_IMMA_PCT_COL),
    })
    for reason in STALL_REASONS:
        summary[f"stall_{reason}"] = col(_wide_stall_col(reason))

    num_cols = ["duration_us", "sm_pct", "dram_pct", "occ_achieved",
                "occ_theoretical", "l1_pct", "l2_pct", "tensor_pct",
                "hmma_pct", "imma_pct"] + [f"stall_{r}" for r in STALL_REASONS]
    agg = (summary.groupby("kernel_short", sort=False)[num_cols]
           .mean()
           .reset_index())

    order = list(dict.fromkeys(summary["kernel_short"]))
    agg["_order"] = agg["kernel_short"].map({k: i for i, k in enumerate(order)})
    agg = agg.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    return agg


def build_kernel_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot metrics into one row per kernel, averaged across multiple captures."""
    if df.empty:
        return pd.DataFrame()

    if df.attrs.get("wide_format"):
        return build_kernel_summary_wide(df)

    has_kernel_time = "Kernel Time" in df.columns
    group_cols = ["ID", "Kernel Name", "kernel_short"]
    if has_kernel_time:
        group_cols.append("Kernel Time")

    records = []
    for key, grp in df.groupby(group_cols, sort=False):
        kid, kname, kshort = key[0], key[1], key[2]
        ktime = key[3] if has_kernel_time else None
        rec = {
            "kernel_id":    kid,
            "kernel_name":  kname,
            "kernel_short": kshort,
            "kernel_time":  ktime,
        }

        def first_match(candidates):
            sub = grp[grp["Metric Name"].apply(lambda m: _match(m, candidates))]
            if sub.empty:
                return float("nan")
            return sub["Metric Value"].mean()

        duration_us = float("nan")
        if has_kernel_time:
            # Older ncu: duration is the "Kernel Time" column (e.g. "1.23 ms").
            kt = str(ktime).strip()
            m = re.match(r'([\d.]+)\s*(ms|us|ns|s)', kt, re.IGNORECASE)
            if m:
                val, unit = float(m.group(1)), m.group(2).lower()
                duration_us = val * {"s": 1e6, "ms": 1e3, "us": 1.0, "ns": 1e-3}[unit]
        else:
            # Newer ncu: duration is a Metric Name="Duration" row, with its
            # own Metric Unit (e.g. "us", "usecond", "ms").
            dur_rows = grp[grp["Metric Name"] == "Duration"]
            if not dur_rows.empty:
                unit = str(dur_rows["Metric Unit"].iloc[0]).strip().lower()
                scale = _DURATION_UNIT_TO_US.get(unit, 1.0)
                duration_us = dur_rows["Metric Value"].mean() * scale
        rec["duration_us"] = duration_us

        rec["sm_pct"]          = first_match(SOL_SM_NAMES)
        rec["dram_pct"]        = first_match(SOL_MEM_NAMES)
        rec["occ_achieved"]    = first_match(OCC_ACHIEVED)
        rec["occ_theoretical"] = first_match(OCC_THEORETICAL)
        rec["l1_pct"]          = first_match(MEM_L1_NAMES)
        rec["l2_pct"]          = first_match(MEM_L2_NAMES)
        rec["tensor_pct"]      = first_match(TENSOR_PCT_NAMES)
        rec["hmma_pct"]        = first_match(TENSOR_HMMA_NAMES)
        rec["imma_pct"]        = first_match(TENSOR_IMMA_NAMES)
        for reason in STALL_REASONS:
            rec[f"stall_{reason}"] = first_match(_stall_names(reason))

        records.append(rec)

    summary = pd.DataFrame(records)

    # Average rows with the same kernel_short (multiple profiling iterations)
    num_cols = ["duration_us", "sm_pct", "dram_pct", "occ_achieved",
                "occ_theoretical", "l1_pct", "l2_pct", "tensor_pct",
                "hmma_pct", "imma_pct"] + [f"stall_{r}" for r in STALL_REASONS]
    agg = (summary.groupby("kernel_short", sort=False)[num_cols]
           .mean()
           .reset_index())

    # Preserve insertion order (first occurrence)
    order = list(dict.fromkeys(summary["kernel_short"]))
    agg["_order"] = agg["kernel_short"].map({k: i for i, k in enumerate(order)})
    agg = agg.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    return agg


# ---------------------------------------------------------------------------
# Kernel selection
# ---------------------------------------------------------------------------

def select_top_kernels(df: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """Restrict to the top-n kernels by execution time, so all charts
    (including occupancy/throughput, which have no natural cutoff of their
    own) share one readable, consistent kernel set.
    """
    if df.empty or len(df) <= n:
        return df

    keep = set(df["duration_us"].nlargest(n).index)

    return df.loc[sorted(keep)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
           "#bcbd22", "#17becf"]

def _bar(ax, labels, values, ylabel, title, color="#1f77b4", ylim=None):
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=color, width=0.6, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if ylim is not None:
        ax.set_ylim(ylim)
    # Value labels on bars
    for bar, v in zip(bars, values):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=6)


def save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------

def plot_duration(df: pd.DataFrame, out_dir: Path):
    labels = df["kernel_short"].tolist()
    vals   = df["duration_us"].tolist()
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    _bar(ax, labels, vals, "Duration (μs)", "Per-kernel GPU execution time", "#1f77b4")
    save(fig, out_dir / "profile_duration.png")


def plot_tensor_core(df: pd.DataFrame, out_dir: Path):
    labels = df["kernel_short"].tolist()
    vals   = df["tensor_pct"].tolist()
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    _bar(ax, labels, vals, "% of peak", "Tensor core utilization (SoL %)",
         "#17becf", ylim=(0, 110))
    save(fig, out_dir / "profile_tensor_core.png")


def plot_tensor_precision(df: pd.DataFrame, out_dir: Path):
    """HMMA (fp16/bf16) vs IMMA (int8/int4/binary) tensor-pipe duty cycle --
    splits the precision-agnostic profile_tensor_core.png number by op type."""
    labels = df["kernel_short"].tolist()
    hmma   = df["hmma_pct"].tolist()
    imma   = df["imma_pct"].tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    w = 0.35
    ax.bar(x - w/2, hmma, w, label="HMMA (fp16/bf16)", color="#17becf", edgecolor="white")
    ax.bar(x + w/2, imma, w, label="IMMA (int8/int4)", color="#bcbd22", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("% of peak", fontsize=8)
    ax.set_title("Tensor pipe utilization by precision", fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out_dir / "profile_tensor_precision.png")


def plot_stall_reasons(df: pd.DataFrame, out_dir: Path):
    """Stacked bar of % stall cycles by reason, per kernel -- identifies the
    bottleneck for kernels with low achieved occupancy/SM throughput."""
    labels = df["kernel_short"].tolist()
    x = np.arange(len(labels))
    stall_cols = [f"stall_{r}" for r in STALL_REASONS if f"stall_{r}" in df.columns
                  and df[f"stall_{r}"].notna().any()]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 6))
    bottom = np.zeros(len(labels))
    for i, col in enumerate(stall_cols):
        vals = df[col].fillna(0).to_numpy()
        reason = col.removeprefix("stall_").replace("_", " ")
        ax.bar(x, vals, bottom=bottom, label=reason,
               color=PALETTE[i % len(PALETTE)], edgecolor="white", linewidth=0.3)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("% of stall cycles", fontsize=8)
    ax.set_title("Warp stall-reason breakdown (bottleneck analysis)", fontsize=9)
    ax.legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out_dir / "profile_stall_reasons.png")


def plot_occupancy(df: pd.DataFrame, out_dir: Path):
    labels   = df["kernel_short"].tolist()
    achieved = df["occ_achieved"].tolist()
    theoret  = df["occ_theoretical"].tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    w = 0.35
    ax.bar(x - w/2, theoret,  w, label="Theoretical", color="#aec7e8", edgecolor="white")
    ax.bar(x + w/2, achieved, w, label="Achieved",    color="#1f77b4", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Occupancy (%)", fontsize=8)
    ax.set_title("Warp occupancy — achieved vs theoretical", fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out_dir / "profile_occupancy.png")


def plot_throughput(df: pd.DataFrame, out_dir: Path):
    labels = df["kernel_short"].tolist()
    sm     = df["sm_pct"].tolist()
    dram   = df["dram_pct"].tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    w = 0.35
    ax.bar(x - w/2, sm,   w, label="SM (compute)", color="#ff7f0e", edgecolor="white")
    ax.bar(x + w/2, dram, w, label="DRAM (memory)", color="#d62728", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("% of peak sustained", fontsize=8)
    ax.set_title("SM and DRAM throughput (Speed-of-Light %)", fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out_dir / "profile_throughput.png")


def plot_memory_bw(df: pd.DataFrame, out_dir: Path):
    labels = df["kernel_short"].tolist()
    l1     = df["l1_pct"].tolist()
    l2     = df["l2_pct"].tolist()
    dram   = df["dram_pct"].tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    w = 0.25
    ax.bar(x - w,   l1,   w, label="L1/TEX",  color="#2ca02c", edgecolor="white")
    ax.bar(x,       l2,   w, label="L2",       color="#98df8a", edgecolor="white")
    ax.bar(x + w,   dram, w, label="DRAM",     color="#d62728", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("% of peak bandwidth", fontsize=8)
    ax.set_title("Memory hierarchy bandwidth utilisation", fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out_dir / "profile_memory_bw.png")


def plot_overview(df: pd.DataFrame, out_dir: Path, gpu_label: str = "A100"):
    """2×2 summary sheet."""
    labels   = df["kernel_short"].tolist()
    n = len(labels)
    x = np.arange(n)
    fig, axes = plt.subplots(2, 2, figsize=(max(14, n * 1.0), 10))
    fig.suptitle(f"Kernel Profiles ({gpu_label})", fontsize=11, fontweight="bold")

    # Duration
    ax = axes[0, 0]
    _bar(ax, labels, df["duration_us"].tolist(), "μs", "Execution time", "#1f77b4")

    # Tensor core utilization
    ax = axes[0, 1]
    _bar(ax, labels, df["tensor_pct"].tolist(), "% of peak",
         "Tensor core utilization (SoL %)", "#17becf", ylim=(0, 110))

    # Occupancy
    ax = axes[1, 0]
    w = 0.35
    ax.bar(x - w/2, df["occ_theoretical"].tolist(), w,
           label="Theoretical", color="#aec7e8", edgecolor="white")
    ax.bar(x + w/2, df["occ_achieved"].tolist(), w,
           label="Achieved", color="#1f77b4", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Occupancy (%)", fontsize=8)
    ax.set_title("Warp occupancy", fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    # SM + DRAM throughput
    ax = axes[1, 1]
    ax.bar(x - w/2, df["sm_pct"].tolist(),   w,
           label="SM",   color="#ff7f0e", edgecolor="white")
    ax.bar(x + w/2, df["dram_pct"].tolist(), w,
           label="DRAM", color="#d62728", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("% of peak", fontsize=8)
    ax.set_title("Compute + DRAM throughput (SoL %)", fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    save(fig, out_dir / "profile_overview.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True,
                    help="Directory containing ncu_metrics.csv")
    ap.add_argument("--csv", default="ncu_metrics.csv",
                    help="ncu CSV filename within results-dir (default: ncu_metrics.csv)")
    ap.add_argument("--gpu-label", default="A100",
                    help="GPU this profile was captured on, shown in the overview "
                         "chart title (default: A100)")
    args = ap.parse_args()

    out_dir  = Path(args.results_dir)
    csv_path = out_dir / args.csv

    if not out_dir.is_dir():
        sys.exit(f"Not a directory: {out_dir}")
    if not csv_path.exists():
        sys.exit(f"ncu CSV not found: {csv_path}\n"
                 "  Run benchmarking/run_profile.sh first.")

    print(f"Parsing {csv_path}")
    df_raw = parse_ncu_csv(csv_path)
    if df_raw.empty:
        sys.exit("No parseable ncu data found.")

    print(f"  {len(df_raw)} metric rows across "
          f"{df_raw['Kernel Name'].nunique()} unique kernel names")

    df = build_kernel_summary(df_raw)
    if df.empty:
        sys.exit("Could not build kernel summary — check CSV format.")

    print(f"  {len(df)} distinct kernels after deduplication")

    df = select_top_kernels(df, n=6)
    df = df.sort_values("duration_us", ascending=False).reset_index(drop=True)
    print(f"  {len(df)} kernels kept (top 6 by execution time), "
          "ordered by descending execution time\n"
          "Generating figures:")

    # Only plot what we actually have data for (avoid all-NaN plots)
    has_duration  = df["duration_us"].notna().any()
    has_occupancy = (df["occ_achieved"].notna().any() or
                     df["occ_theoretical"].notna().any())
    has_throughput = (df["sm_pct"].notna().any() or df["dram_pct"].notna().any())
    has_mem_bw    = (df["l1_pct"].notna().any() or df["l2_pct"].notna().any())
    has_tensor    = df["tensor_pct"].notna().any()
    has_tensor_precision = (df["hmma_pct"].notna().any() or df["imma_pct"].notna().any())
    stall_cols    = [f"stall_{r}" for r in STALL_REASONS if f"stall_{r}" in df.columns]
    has_stalls    = any(df[c].notna().any() for c in stall_cols)

    if has_duration:
        plot_duration(df, out_dir)
    if has_tensor:
        plot_tensor_core(df, out_dir)
    if has_tensor_precision:
        plot_tensor_precision(df, out_dir)
    if has_occupancy:
        plot_occupancy(df, out_dir)
    if has_throughput:
        plot_throughput(df, out_dir)
    if has_mem_bw:
        plot_memory_bw(df, out_dir)
    if has_stalls:
        plot_stall_reasons(df, out_dir)

    # Overview always attempted (sub-plots just show zeros for missing metrics)
    plot_overview(df, out_dir, gpu_label=args.gpu_label)

    print(f"\nAll figures saved to {out_dir}")

    # Print a text summary table
    print("\nKernel summary:")
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.1f}".format)
    cols = ["kernel_short", "duration_us", "occ_achieved", "occ_theoretical",
            "sm_pct", "dram_pct", "tensor_pct", "hmma_pct", "imma_pct"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()

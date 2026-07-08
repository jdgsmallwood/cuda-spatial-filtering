"""Parse benchmark stdout files from run_benchmarks.sh and save figures.

Each benchmark binary writes one tagged line per config, e.g.:
  [CorrBeam ch=8 fpga=1 rx=10] elapsed=... GB/sec=0.123456
  [LambdaGPU ch=8 fpga=1 rx=10] avg_gpu_ms=5.2 gpu_util=85.3%
  [Processor ch=8 fpga=1 rx=10] ... GB/sec=0.456

Run with:
  python plot_benchmarks.py --results-dir benchmarking/20260703_120000
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
except ImportError:
    sys.exit("matplotlib not found — pip install matplotlib")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _kv(line: str) -> dict:
    """Extract all key=value pairs (and bare float-like values) from a line.

    Keys like `runs/sec`/`GB/sec`/`input_GB/sec` contain `/`, which `\\w`
    doesn't match -- so the key class must include it or these get truncated
    to just `sec`.
    """
    out = {}
    for m in re.finditer(r'([A-Za-z_][A-Za-z0-9_/]*)=([-+0-9.eE]+)%?', line):
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return out


def parse_file(path: Path) -> list[dict]:
    """Parse tagged benchmark lines, tagging each row with its sweep section.

    The corr-packet sweep (`corr=64..1024`, held at ch=8) reuses the same
    ch=8 configs as the channel sweep, so callers must filter on
    `sweep == "channel"` -- otherwise the corr sweep reappears as jittery
    duplicate points at ch=8. (No chart plots the corr sweep itself anymore,
    but the tag is kept so channel-sweep charts can still exclude it.)
    """
    rows = []
    sweep = "channel"
    with open(path) as f:
        for line in f:
            if line.startswith("==="):
                sweep = "corr" if "corr-packet sweep" in line else "channel"
                continue
            m = re.match(r'\[(\w[\w ]+) ch=(\d+) fpga=(\d+) rx=(\d+)\](.+)', line)
            if not m:
                continue
            tag, ch, fpga, rx, rest = m.groups()
            kv = _kv(rest)
            kv.update(tag=tag.strip(), ch=int(ch), fpga=int(fpga), rx=int(rx),
                      sweep=sweep)
            rows.append(kv)
    return rows


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS_4FPGA = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


# Real-time ingest rate: each (fpga, channel) pair delivers this many
# packets/sec, at this many bytes/packet.
REALTIME_PACKETS_PER_SEC_PER_FPGA_CHANNEL = 15500
REALTIME_BYTES_PER_PACKET = 2660


def realtime_gbps(nr_channels: int, nr_fpgas: int) -> float:
    """GB/s the pipeline must sustain to keep up with real-time ingest."""
    packets_per_sec = (REALTIME_PACKETS_PER_SEC_PER_FPGA_CHANNEL
                       * nr_channels * nr_fpgas)
    return packets_per_sec * REALTIME_BYTES_PER_PACKET / 1e9


def save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path}")


# ---------------------------------------------------------------------------
# Per-benchmark figure generators
# ---------------------------------------------------------------------------

def plot_corrbeam_channel_sweep(rows: list[dict], out_dir: Path, suffix: str = ""):
    tag = "CorrBeam"
    # channel sweep: corr=256 only (exclude the corr-packet sweep configs,
    # which reuse ch=8 and would otherwise show up as jitter there).
    # 1-FPGA runs are dropped -- only the 4-FPGA (real deployment) series
    # matters here.
    data = [r for r in rows
            if r["tag"] == tag and r.get("sweep") == "channel" and r["fpga"] != 1]
    # Group by (fpga) → sorted by ch
    groups = {}
    for r in data:
        groups.setdefault(r["fpga"], []).append(r)
    if not groups:
        return
    for g in groups.values():
        g.sort(key=lambda r: r["ch"])

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(f"CorrBeam+Only GPU Throughput — Channel Sweep{suffix}")

    for fpga, group in sorted(groups.items()):
        xs = [r["ch"] for r in group]
        ys = [r.get("GB/sec", 0) for r in group]
        ax.plot(xs, ys, "o-", label=f"{fpga} FPGA", color=COLORS_4FPGA[0])
        # Real-time data-rate requirement: 15500 packets/sec per
        # fpga/channel, 2660 bytes/packet.
        rt_ys = [realtime_gbps(ch, fpga) for ch in xs]
        ax.plot(xs, rt_ys, "x--", label=f"Real-time required ({fpga} FPGA)",
                color="black")
    ax.set_xlabel("NR_CHANNELS")
    ax.set_ylabel("GB/s (total I/O)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(8))

    save(fig, out_dir / f"corrbeam_channel_sweep{suffix}.png")


def plot_lambda_gpu(rows: list[dict], out_dir: Path, suffix: str = ""):
    tag = "LambdaGPU"
    # 1-FPGA runs are dropped -- only the 4-FPGA (real deployment) series matters here.
    data = [r for r in rows if r["tag"] == tag and r["fpga"] != 1]
    if not data:
        return
    groups = {}
    for r in data:
        groups.setdefault(r["fpga"], []).append(r)
    for g in groups.values():
        g.sort(key=lambda r: r["ch"])

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(f"RFIMit (corr+beam+eigen+fft) — Channel Sweep{suffix}")

    for fpga, group in sorted(groups.items()):
        xs = [r["ch"] for r in group]
        ys = [r.get("input_GB/sec", 0) for r in group]
        ax.plot(xs, ys, "o-", label=f"{fpga} FPGA", color=COLORS_4FPGA[0])
        rt_ys = [realtime_gbps(ch, fpga) for ch in xs]
        ax.plot(xs, rt_ys, "x--", label=f"Real-time required ({fpga} FPGA)",
                color="black")
    ax.set_xlabel("NR_CHANNELS")
    ax.set_ylabel("Input GB/s")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(8))

    save(fig, out_dir / f"lambda_gpu_channel_sweep{suffix}.png")


def plot_gpu_channel_sweep_combined(rows: list[dict], out_dir: Path, suffix: str = ""):
    """CorrBeam-only and RFIMit channel sweeps overlaid on one set of axes.

    Single-dataset counterpart to plot_gpu_comparison() (which overlays
    multiple machines) -- here it's one machine's two pipelines together, so
    the RFIMit overhead vs. the real-time line is visible at a glance.
    Only the 4-FPGA series is plotted, matching the other channel-sweep charts.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.suptitle(f"GPU Throughput — Channel Sweep (4 FPGA){suffix}")

    for tag, metric, style, marker, color, label in [
        ("CorrBeam", "GB/sec", "-", "o", COLORS_4FPGA[0], "CorrBeam-only"),
        ("LambdaGPU", "input_GB/sec", "-", "s", COLORS_4FPGA[1], "RFIMit"),
    ]:
        data = [r for r in rows if r["tag"] == tag and r["fpga"] == 4
                and (tag != "CorrBeam" or r.get("sweep") == "channel")]
        data.sort(key=lambda r: r["ch"])
        if not data:
            continue
        xs = [r["ch"] for r in data]
        ys = [r.get(metric, 0) for r in data]
        ax.plot(xs, ys, marker + style, label=label, color=color)

    all_ch = sorted({r["ch"] for r in rows if r["fpga"] == 4})
    if all_ch:
        rt_ys = [realtime_gbps(ch, 4) for ch in all_ch]
        ax.plot(all_ch, rt_ys, "x--", label="Real-time required (4 FPGA)",
                color="black")

    ax.set_xlabel("NR_CHANNELS")
    ax.set_ylabel("Input GB/s")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(8))

    save(fig, out_dir / f"gpu_channel_sweep_combined{suffix}.png")


COMPARISON_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]


def plot_gpu_comparison(datasets: list[tuple[str, list[dict]]], out_dir: Path):
    """Overlay the CorrBeam channel-sweep and LambdaGPU charts across machines.

    `datasets` is a list of (label, gpu_rows) pairs, e.g.
    [("A100", ozstar_rows), ("RTX 4000", blackmesa_rows)]. Only the 4-FPGA
    series is plotted per machine (matching plot_corrbeam_channel_sweep).
    """
    colors = {label: COMPARISON_COLORS[i % len(COMPARISON_COLORS)]
              for i, (label, _) in enumerate(datasets)}

    # --- CorrBeam (GB/s) + LambdaGPU (input GB/s) overlaid on one chart ---
    # Machine → color, pipeline → marker/linestyle, so both dimensions read
    # clearly on the same axes.
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.suptitle("GPU Throughput — Channel Sweep (4 FPGA)")

    for tag, metric, style, marker in [
        ("CorrBeam", "GB/sec", "-", "o"),
        ("LambdaGPU", "input_GB/sec", "-", "s"),
    ]:
        for label, rows in datasets:
            data = [r for r in rows if r["tag"] == tag and r["fpga"] == 4
                    and (tag != "CorrBeam" or r.get("sweep") == "channel")]
            data.sort(key=lambda r: r["ch"])
            if not data:
                continue
            xs = [r["ch"] for r in data]
            ys = [r.get(metric, 0) for r in data]
            pipeline_name = "CorrBeam-only" if tag == "CorrBeam" else "RFIMit"
            ax.plot(xs, ys, marker + style, label=f"{label} — {pipeline_name}",
                    color=colors[label])

    any_rows = next((rows for _, rows in datasets if rows), [])
    xs = sorted({r["ch"] for r in any_rows if r["fpga"] == 4}) or [8, 16, 24, 32]
    rt_ys = [realtime_gbps(ch, 4) for ch in xs]
    ax.plot(xs, rt_ys, "x--", label="Real-time required (4 FPGA)", color="black")

    ax.set_xlabel("NR_CHANNELS")
    ax.set_ylabel("Input GB/s")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(8))

    save(fig, out_dir / "comparison_gpu_channel_sweep.png")


def plot_processor(rows: list[dict], out_dir: Path):
    tag = "Processor"
    # bench_processor also runs a ch=1 baseline/warmup config outside the
    # 8/16/24/32 channel sweep -- drop it so it doesn't show up as a stray
    # point near the y-axis. 1-FPGA runs are dropped too -- only the 4-FPGA
    # (real deployment) series matters here.
    data = [r for r in rows if r["tag"] == tag and r["ch"] != 1 and r["fpga"] != 1]
    if not data:
        return
    groups = {}
    for r in data:
        groups.setdefault(r["fpga"], []).append(r)
    for g in groups.values():
        g.sort(key=lambda r: r["ch"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Processor (packet ring-buffer / reassembly) — Channel Sweep")

    for ax, metric, ylabel in [
        (axes[0], "GB/sec", "GB/s"),
        (axes[1], "packets/sec", "Packets/s"),
    ]:
        for fpga, group in sorted(groups.items()):
            xs = [r["ch"] for r in group]
            ys = [r.get(metric, 0) for r in group]
            ax.plot(xs, ys, "o-", label=f"{fpga} FPGA", color=COLORS_4FPGA[0])
        ax.set_xlabel("NR_CHANNELS")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(8))

    save(fig, out_dir / "processor_channel_sweep.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def gpu_rows_from_dir(results_dir: Path) -> list[dict]:
    rows = []
    for txt in sorted(results_dir.glob("*.txt")):
        if "bench_gpu" in txt.stem and "with_output" not in txt.stem:
            rows.extend(parse_file(txt))
    return rows


def processor_rows_from_dir(results_dir: Path) -> list[dict]:
    rows = []
    for txt in sorted(results_dir.glob("*.txt")):
        if "bench_processor" in txt.stem:
            rows.extend(parse_file(txt))
    return rows


def plot_processor_comparison(datasets: list[tuple[str, list[dict]]], out_dir: Path):
    """Overlay the Processor channel-sweep GB/s across machines/CPUs.

    `datasets` is a list of (label, processor_rows) pairs, e.g.
    [("Xeon 4210", blackmesa_rows), ("Ryzen 7 7700", ryzen_rows)]. Only the
    4-FPGA series is plotted, and the ch=1 baseline/warmup config is dropped
    (matching plot_processor).
    """
    colors = {label: COMPARISON_COLORS[i % len(COMPARISON_COLORS)]
              for i, (label, _) in enumerate(datasets)}

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.suptitle("Processor (packet ring-buffer / reassembly) — Channel Sweep")

    for label, rows in datasets:
        data = [r for r in rows if r["tag"] == "Processor" and r["ch"] != 1
                and r["fpga"] == 4]
        data.sort(key=lambda r: r["ch"])
        if not data:
            continue
        xs = [r["ch"] for r in data]
        ys = [r.get("GB/sec", 0) for r in data]
        ax.plot(xs, ys, "o-", label=label, color=colors[label])

    ax.set_xlabel("NR_CHANNELS")
    ax.set_ylabel("GB/s")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(8))

    save(fig, out_dir / "comparison_processor_channel_sweep.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir",
                    help="Directory containing benchmark .txt output files")
    ap.add_argument("--compare", nargs="+", metavar="DIR=LABEL",
                    help="Overlay GPU charts from multiple results dirs, e.g. "
                         "--compare ozstar_benchmarks=A100 "
                         "blackmesa_benchmarks/20260708_101738='RTX 4000'")
    ap.add_argument("--compare-processor", nargs="+", metavar="DIR=LABEL",
                    help="Overlay Processor charts from multiple results dirs, e.g. "
                         "--compare-processor "
                         "blackmesa_benchmarks/20260708_101738='Xeon 4210' "
                         "20260708_000926='Ryzen 7 7700'")
    ap.add_argument("--compare-out-dir", default="benchmarking",
                    help="Where to save --compare/--compare-processor figures "
                         "(default: benchmarking/)")
    args = ap.parse_args()

    def parse_dir_label_pairs(items, loader):
        datasets = []
        for item in items:
            dir_str, _, label = item.rpartition("=")
            if not dir_str:
                sys.exit(f"entries must be DIR=LABEL, got: {item!r}")
            d = Path(dir_str)
            if not d.is_dir():
                sys.exit(f"Not a directory: {d}")
            datasets.append((label, loader(d)))
        return datasets

    if args.compare or args.compare_processor:
        out_dir = Path(args.compare_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.compare:
            datasets = parse_dir_label_pairs(args.compare, gpu_rows_from_dir)
            print("Generating GPU comparison figures:")
            plot_gpu_comparison(datasets, out_dir)
        if args.compare_processor:
            datasets = parse_dir_label_pairs(args.compare_processor, processor_rows_from_dir)
            print("Generating Processor comparison figures:")
            plot_processor_comparison(datasets, out_dir)
        print(f"\nComparison figures saved to {out_dir}")
        return

    if not args.results_dir:
        sys.exit("Must pass --results-dir or --compare")

    out_dir = Path(args.results_dir)
    if not out_dir.is_dir():
        sys.exit(f"Not a directory: {out_dir}")

    print(f"Parsing results from {out_dir}")

    gpu_rows = []
    gpu_output_rows = []
    processor_rows = []

    for txt in sorted(out_dir.glob("*.txt")):
        rows = parse_file(txt)
        if not rows:
            continue
        print(f"  {txt.name}: {len(rows)} data rows")
        name = txt.stem
        if "bench_gpu_with_output" in name:
            gpu_output_rows.extend(rows)
        elif "bench_gpu" in name:
            gpu_rows.extend(rows)
        elif "bench_processor" in name:
            processor_rows.extend(rows)

    if not any([gpu_rows, gpu_output_rows, processor_rows]):
        sys.exit("No parseable benchmark data found in result files.")

    print("Generating figures:")

    if gpu_rows:
        plot_corrbeam_channel_sweep(gpu_rows, out_dir)
        plot_lambda_gpu(gpu_rows, out_dir)
        plot_gpu_channel_sweep_combined(gpu_rows, out_dir)

    if gpu_output_rows:
        plot_corrbeam_channel_sweep(gpu_output_rows, out_dir, suffix="_with_output")
        plot_lambda_gpu(gpu_output_rows, out_dir, suffix="_with_output")
        plot_gpu_channel_sweep_combined(gpu_output_rows, out_dir, suffix="_with_output")

    if processor_rows:
        plot_processor(processor_rows, out_dir)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()

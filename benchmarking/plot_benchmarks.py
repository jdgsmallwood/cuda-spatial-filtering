"""Parse benchmark stdout files from run_benchmarks.sh and save figures.

Each benchmark binary writes one tagged line per config, e.g.:
  [CorrBeam ch=8 fpga=1 rx=10] elapsed=... GB/sec=0.123456
  [LambdaGPU ch=8 fpga=1 rx=10] avg_gpu_ms=5.2 gpu_util=85.3%
  [Processor ch=8 fpga=1 rx=10] ... GB/sec=0.456
  [Vis Writer ch=8 fpga=1 rx=10] ... GB/sec=0.789
  [Eigen Writer ch=8 fpga=1 rx=10] ... GB/sec=0.789

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
    """Extract all key=value pairs (and bare float-like values) from a line."""
    out = {}
    for m in re.finditer(r'(\w+)=([\d.]+)%?', line):
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return out


def parse_file(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            m = re.match(r'\[(\w[\w ]+) ch=(\d+) fpga=(\d+) rx=(\d+)\](.+)', line)
            if not m:
                continue
            tag, ch, fpga, rx, rest = m.groups()
            kv = _kv(rest)
            kv.update(tag=tag.strip(), ch=int(ch), fpga=int(fpga), rx=int(rx))
            rows.append(kv)
    return rows


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS_1FPGA = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
COLORS_4FPGA = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def ch_label(r: dict) -> str:
    return f"{r['ch']}ch/{r['fpga']}fpga"


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
    # channel sweep: corr=256 only (exclude the corr-packet sweep configs)
    data = [r for r in rows if r["tag"] == tag]
    # Group by (fpga) → sorted by ch
    groups = {}
    for r in data:
        groups.setdefault(r["fpga"], []).append(r)
    if not groups:
        return
    for g in groups.values():
        g.sort(key=lambda r: r["ch"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"CorrBeam+Only GPU Throughput — Channel Sweep{suffix}")

    for ax, metric, ylabel in [
        (axes[0], "GB/sec", "GB/s (total I/O)"),
        (axes[1], "runs/sec", "Pipeline runs/s"),
    ]:
        for fpga, group in sorted(groups.items()):
            colors = COLORS_1FPGA if fpga == 1 else COLORS_4FPGA
            xs = [r["ch"] for r in group]
            ys = [r.get(metric, r.get("GB", 0)) for r in group]
            ax.plot(xs, ys, "o-", label=f"{fpga} FPGA", color=colors[0])
        ax.set_xlabel("NR_CHANNELS")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(8))

    save(fig, out_dir / f"corrbeam_channel_sweep{suffix}.png")


def plot_corrbeam_corr_sweep(rows: list[dict], out_dir: Path, suffix: str = ""):
    """Correlation-packet sweep: 8ch, corr=64..1024."""
    tag = "CorrBeam"
    # The corr-packet sweep: ch=8, corr varies — but we don't have corr in the
    # output directly. We use runs/sec as proxy (higher corr → fewer runs/sec
    # for same duration). The caller should separate the two sets by file.
    data = [r for r in rows if r["tag"] == tag and r["ch"] == 8]
    if not data:
        return
    # Sort by GB/sec descending as proxy for corr (smaller corr = more runs)
    data.sort(key=lambda r: r.get("runs/sec", 0), reverse=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"CorrBeam+Only — Corr-Packet Sweep (8ch){suffix}")
    for fpga in sorted({r["fpga"] for r in data}):
        sub = sorted([r for r in data if r["fpga"] == fpga],
                     key=lambda r: r.get("runs/sec", 0), reverse=True)
        ys = [r.get("GB/sec", 0) for r in sub]
        xs = range(len(ys))
        labels = [ch_label(r) for r in sub]
        ax.bar([x + fpga * 0.4 for x in xs], ys, width=0.4,
               label=f"{fpga} FPGA",
               color=COLORS_1FPGA[0] if fpga == 1 else COLORS_4FPGA[0])
    ax.set_ylabel("GB/s (total I/O)")
    ax.set_xlabel("Config (ordered by throughput, high→low corr)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    save(fig, out_dir / f"corrbeam_corr_sweep{suffix}.png")


def plot_lambda_gpu(rows: list[dict], out_dir: Path, suffix: str = ""):
    tag = "LambdaGPU"
    data = [r for r in rows if r["tag"] == tag]
    if not data:
        return
    groups = {}
    for r in data:
        groups.setdefault(r["fpga"], []).append(r)
    for g in groups.values():
        g.sort(key=lambda r: r["ch"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"LambdaGPU (corr+beam+eigen+fft) — Channel Sweep{suffix}")

    for ax, metric, ylabel in [
        (axes[0], "avg_gpu_ms", "Avg GPU kernel time (ms)"),
        (axes[1], "gpu_util", "GPU utilisation (%)"),
    ]:
        for fpga, group in sorted(groups.items()):
            xs = [r["ch"] for r in group]
            ys = [r.get(metric, 0) for r in group]
            if metric == "gpu_util":
                ys = [y * 100 if y <= 1.0 else y for y in ys]
            color = COLORS_1FPGA[0] if fpga == 1 else COLORS_4FPGA[0]
            ax.plot(xs, ys, "o-", label=f"{fpga} FPGA", color=color)
        ax.set_xlabel("NR_CHANNELS")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(8))

    save(fig, out_dir / f"lambda_gpu_channel_sweep{suffix}.png")


def plot_processor(rows: list[dict], out_dir: Path):
    tag = "Processor"
    data = [r for r in rows if r["tag"] == tag]
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
            color = COLORS_1FPGA[0] if fpga == 1 else COLORS_4FPGA[0]
            ax.plot(xs, ys, "o-", label=f"{fpga} FPGA", color=color)
        ax.set_xlabel("NR_CHANNELS")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(8))

    save(fig, out_dir / "processor_channel_sweep.png")


def plot_writers(rows: list[dict], out_dir: Path):
    for tag, label in [("Vis Writer", "Visibility"), ("Eigen Writer", "Eigendata")]:
        data = [r for r in rows if r["tag"] == tag]
        if not data:
            continue
        groups = {}
        for r in data:
            groups.setdefault(r["fpga"], []).append(r)
        for g in groups.values():
            g.sort(key=lambda r: r["ch"])

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(f"HDF5 {label} Writer Throughput — Channel Sweep")
        for fpga, group in sorted(groups.items()):
            xs = [r["ch"] for r in group]
            ys = [r.get("GB/sec", 0) for r in group]
            color = COLORS_1FPGA[0] if fpga == 1 else COLORS_4FPGA[0]
            ax.plot(xs, ys, "o-", label=f"{fpga} FPGA", color=color)
        ax.set_xlabel("NR_CHANNELS")
        ax.set_ylabel("GB/s")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(8))
        fname = tag.lower().replace(" ", "_")
        save(fig, out_dir / f"{fname}_channel_sweep.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True,
                    help="Directory containing benchmark .txt output files")
    args = ap.parse_args()

    out_dir = Path(args.results_dir)
    if not out_dir.is_dir():
        sys.exit(f"Not a directory: {out_dir}")

    print(f"Parsing results from {out_dir}")

    gpu_rows = []
    gpu_output_rows = []
    processor_rows = []
    writer_rows = []

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
        elif "bench_writers" in name:
            writer_rows.extend(rows)

    if not any([gpu_rows, gpu_output_rows, processor_rows, writer_rows]):
        sys.exit("No parseable benchmark data found in result files.")

    print("Generating figures:")

    if gpu_rows:
        plot_corrbeam_channel_sweep(gpu_rows, out_dir)
        plot_corrbeam_corr_sweep(gpu_rows, out_dir)
        plot_lambda_gpu(gpu_rows, out_dir)

    if gpu_output_rows:
        plot_corrbeam_channel_sweep(gpu_output_rows, out_dir, suffix="_with_output")
        plot_lambda_gpu(gpu_output_rows, out_dir, suffix="_with_output")

    if processor_rows:
        plot_processor(processor_rows, out_dir)

    if writer_rows:
        plot_writers(writer_rows, out_dir)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()

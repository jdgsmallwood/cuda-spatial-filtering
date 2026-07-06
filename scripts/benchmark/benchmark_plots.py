import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LAMBDA Pipeline Benchmark Results

    Throughput measurements for the CUDA spatial-filtering pipeline across four stages:

    | Stage | Binary | Primary metric |
    |-------|--------|----------------|
    | Packet capture | `bench_capture` | Mpkt/s, GB/s |
    | Ring-buffer processor | `bench_processor` | Mpkt/s, GB/s (8 configs) |
    | GPU pipeline (correlate + beamform) | `bench_gpu` | runs/s, GB/s in/out (8 configs) |
    | HDF5 writers | `bench_writers` | blocks/s, GB/s (8 configs × 2 writers) |

    ---

    **Usage**

    - Toggle *Run benchmarks* to execute `run_benchmarks.py` against a compiled build directory.
    - Or leave it off and point *Results JSON path* at an existing results file to regenerate
      plots without rerunning (useful once results are captured on a GPU node).
    - Enable *Save figures* to write PDF + PNG exports to the figures directory.
    """)
    return


@app.cell
def _():
    import json
    import subprocess
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    try:
        NOTEBOOK_DIR = Path(__file__).resolve().parent
    except NameError:
        NOTEBOOK_DIR = Path.cwd()

    return NOTEBOOK_DIR, Path, json, matplotlib, mo, np, pd, plt, subprocess, sys


@app.cell
def _(matplotlib):
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.framealpha": 0.85,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
    })
    return


@app.cell
def _(mo):
    run_switch = mo.ui.switch(value=False, label="Run benchmarks")
    json_path_input = mo.ui.text(
        value="results/benchmark_results.json",
        label="Results JSON path (relative to notebook or absolute)",
    )
    build_dir_input = mo.ui.text(
        value="../../build_apps",
        label="Build directory (used when Run benchmarks is on)",
    )
    duration_slider = mo.ui.slider(
        start=5, stop=120, value=10, show_value=True,
        label="Duration per stage (s)",
    )
    figures_dir_input = mo.ui.text(value="figures", label="Figures output directory")
    save_switch = mo.ui.switch(value=False, label="Save figures to disk")

    mo.vstack([
        mo.md("## Configuration"),
        run_switch,
        json_path_input,
        build_dir_input,
        duration_slider,
        mo.md("---"),
        figures_dir_input,
        save_switch,
    ])
    return (
        build_dir_input,
        duration_slider,
        figures_dir_input,
        json_path_input,
        run_switch,
        save_switch,
    )


@app.cell
def _(
    NOTEBOOK_DIR,
    Path,
    build_dir_input,
    duration_slider,
    json,
    json_path_input,
    mo,
    run_switch,
    subprocess,
    sys,
):
    _json_path = Path(json_path_input.value)
    if not _json_path.is_absolute():
        _json_path = NOTEBOOK_DIR / _json_path

    _build_dir = Path(build_dir_input.value)
    if not _build_dir.is_absolute():
        _build_dir = NOTEBOOK_DIR / _build_dir

    data = None
    _msg = ""

    if run_switch.value:
        _json_path.parent.mkdir(parents=True, exist_ok=True)
        _cmd = [
            sys.executable,
            str(NOTEBOOK_DIR / "run_benchmarks.py"),
            "--build-dir", str(_build_dir),
            "--duration", str(int(duration_slider.value)),
            "--json-out", str(_json_path),
        ]
        _proc = subprocess.run(_cmd, capture_output=True, text=True)
        if _proc.returncode == 0 and _json_path.exists():
            data = json.loads(_json_path.read_text())
            _msg = f"Benchmarks complete. Results saved to `{_json_path}`."
        else:
            _out = (_proc.stderr or _proc.stdout or "")[-3000:]
            _msg = f"**Benchmark failed** (exit {_proc.returncode}):\n\n```\n{_out}\n```"
    elif _json_path.exists():
        data = json.loads(_json_path.read_text())
        _meta = data.get("meta", {})
        _ts = _meta.get("timestamp", "")[:19].replace("T", " ")
        _msg = (
            f"Loaded `{_json_path.name}` — "
            f"commit `{_meta.get('git_commit', 'unknown')}` on "
            f"`{_meta.get('git_branch', 'unknown')}`, "
            f"host `{_meta.get('hostname', 'unknown')}`, "
            f"{_ts} UTC"
        )
    else:
        _msg = (
            f"No results file found at `{_json_path}`.  \n"
            "Toggle **Run benchmarks** above to generate one, or update the path."
        )

    mo.md(_msg)
    return (data,)


@app.cell
def _(data, pd):
    def _to_df(rows):
        if not rows or not isinstance(rows, list):
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if "nr_channels" not in df.columns or "nr_fpga_sources" not in df.columns:
            return df
        df = df.sort_values(["nr_channels", "nr_fpga_sources"]).reset_index(drop=True)
        df["config"] = df.apply(
            lambda r: f"{int(r['nr_channels'])}ch/{int(r['nr_fpga_sources'])}fpga", axis=1
        )
        return df

    proc_df = pd.DataFrame()
    gpu_df = pd.DataFrame()
    vis_df = pd.DataFrame()
    eigen_df = pd.DataFrame()
    capture_data = {}

    if data is not None:
        proc_rows = data.get("processor")
        gpu_rows = data.get("gpu_pipeline")
        proc_df = _to_df(proc_rows if isinstance(proc_rows, list) else [])
        gpu_df = _to_df(gpu_rows if isinstance(gpu_rows, list) else [])
        vis_df = _to_df(data.get("vis_writer") if isinstance(data.get("vis_writer"), list) else [])
        eigen_df = _to_df(data.get("eigen_writer") if isinstance(data.get("eigen_writer"), list) else [])
        if isinstance(data.get("capture"), dict):
            capture_data = data["capture"]

    return capture_data, eigen_df, gpu_df, proc_df, vis_df


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Figure 1 — CPU Ring-buffer Processor Throughput")
    return


@app.cell
def _(mo, plt, proc_df):
    _COLORS = {1: "#1f77b4", 4: "#ff7f0e"}
    _MARKERS = {1: "o", 4: "s"}

    fig1, _ax = plt.subplots(figsize=(3.5, 2.8))

    if not proc_df.empty and "packets/sec" in proc_df.columns:
        for _n_fpga, _grp in proc_df.groupby("nr_fpga_sources"):
            _n = int(_n_fpga)
            _ax.plot(
                _grp["nr_channels"],
                _grp["packets/sec"] / 1e6,
                color=_COLORS.get(_n, "#555"),
                marker=_MARKERS.get(_n, "x"),
                label=f"{_n} FPGA {'source' if _n == 1 else 'sources'}",
            )

        if "GB/sec" in proc_df.columns:
            _ax2 = _ax.twinx()
            _ax2.spines["top"].set_visible(False)
            for _n_fpga, _grp in proc_df.groupby("nr_fpga_sources"):
                _n = int(_n_fpga)
                _ax2.plot(
                    _grp["nr_channels"],
                    _grp["GB/sec"],
                    color=_COLORS.get(_n, "#555"),
                    marker=_MARKERS.get(_n, "x"),
                    linestyle="--",
                    alpha=0.5,
                )
            _ax2.set_ylabel("GB/s", color="#777")
            _ax2.tick_params(axis="y", colors="#777")

        _ax.set_xlabel("Number of channels")
        _ax.set_ylabel("Throughput (Mpkt/s)")
        _ax.set_title("Ring-buffer processor throughput")
        _ax.legend(loc="best")
    else:
        _ax.text(0.5, 0.5, "No processor data", ha="center", va="center",
                 transform=_ax.transAxes, color="#999")
        _ax.set_title("Ring-buffer processor throughput")
        mo.md("*No processor data available — load a results JSON or run benchmarks.*")

    plt.tight_layout()
    fig1
    return (fig1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Figure 2 — GPU Pipeline Throughput")
    return


@app.cell
def _(gpu_df, mo, plt):
    _COLORS = {1: "#1f77b4", 4: "#ff7f0e"}
    _MARKERS = {1: "o", 4: "s"}

    fig2, _ax = plt.subplots(figsize=(3.5, 2.8))

    if not gpu_df.empty and "runs/sec" in gpu_df.columns:
        for _n_fpga, _grp in gpu_df.groupby("nr_fpga_sources"):
            _n = int(_n_fpga)
            _ax.plot(
                _grp["nr_channels"],
                _grp["runs/sec"],
                color=_COLORS.get(_n, "#555"),
                marker=_MARKERS.get(_n, "x"),
                label=f"{_n} FPGA {'source' if _n == 1 else 'sources'}",
            )

        _ax.set_xlabel("Number of channels")
        _ax.set_ylabel("Throughput (runs/s)")
        _ax.set_title("GPU pipeline throughput (correlate + beamform)")
        _ax.legend(loc="best")

        if "input_GB/sec" in gpu_df.columns:
            _ax2 = _ax.twinx()
            _ax2.spines["top"].set_visible(False)
            for _n_fpga, _grp in gpu_df.groupby("nr_fpga_sources"):
                _n = int(_n_fpga)
                _ax2.plot(
                    _grp["nr_channels"],
                    _grp["input_GB/sec"],
                    color=_COLORS.get(_n, "#555"),
                    marker=_MARKERS.get(_n, "x"),
                    linestyle="--",
                    alpha=0.5,
                )
            _ax2.set_ylabel("Input GB/s", color="#777")
            _ax2.tick_params(axis="y", colors="#777")
    else:
        _ax.text(0.5, 0.5, "No GPU pipeline data", ha="center", va="center",
                 transform=_ax.transAxes, color="#999")
        _ax.set_title("GPU pipeline throughput (correlate + beamform)")
        mo.md("*No GPU pipeline data available.*")

    plt.tight_layout()
    fig2
    return (fig2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Figure 3 — HDF5 Writer Throughput")
    return


@app.cell
def _(eigen_df, mo, plt, vis_df):
    _COLORS = {"vis": "#2ca02c", "eigen": "#d62728"}
    _MARKERS = {"vis": "o", "eigen": "s"}
    _STYLES = {1: "-", 4: "--"}

    fig3, _ax = plt.subplots(figsize=(3.5, 2.8))

    _has_data = False
    for (_label, _df) in [("vis", vis_df), ("eigen", eigen_df)]:
        if _df.empty or "GB/sec" not in _df.columns:
            continue
        _has_data = True
        for _n_fpga, _grp in _df.groupby("nr_fpga_sources"):
            _n = int(_n_fpga)
            _lbl = f"{_label} writer, {_n} fpga"
            _ax.plot(
                _grp["nr_channels"],
                _grp["GB/sec"],
                color=_COLORS[_label],
                marker=_MARKERS[_label],
                linestyle=_STYLES.get(_n, ":"),
                label=_lbl,
            )

    if _has_data:
        _ax.set_xlabel("Number of channels")
        _ax.set_ylabel("Write throughput (GB/s)")
        _ax.set_title("HDF5 writer throughput")
        _ax.legend(loc="best", fontsize=7)
    else:
        _ax.text(0.5, 0.5, "No writer data", ha="center", va="center",
                 transform=_ax.transAxes, color="#999")
        _ax.set_title("HDF5 writer throughput")
        mo.md("*No writer data available.*")

    plt.tight_layout()
    fig3
    return (fig3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 4 — Pipeline Bottleneck Comparison

    All stages plotted on a common GB/s axis. The lowest curve at any configuration
    is the throughput bottleneck for that operating point.
    """)
    return


@app.cell
def _(eigen_df, gpu_df, mo, plt, proc_df, vis_df):
    fig4, _ax = plt.subplots(figsize=(5.5, 3.0))

    _STAGE_CFG = [
        ("Processor",    proc_df,  "GB/sec",       "#1f77b4", "o", "-"),
        ("GPU in",       gpu_df,   "input_GB/sec",  "#2ca02c", "s", "-"),
        ("GPU out",      gpu_df,   "output_GB/sec", "#9467bd", "^", "--"),
        ("Vis writer",   vis_df,   "GB/sec",        "#d62728", "D", "-"),
        ("Eigen writer", eigen_df, "GB/sec",        "#ff7f0e", "x", "-"),
    ]

    _has_data = False
    for (_stage_lbl, _df, _col, _color, _marker, _ls) in _STAGE_CFG:
        if _df.empty or _col not in _df.columns:
            continue
        _has_data = True
        _combined = (
            _df.groupby("nr_channels")[_col]
            .mean()
            .reset_index()
        )
        _ax.plot(
            _combined["nr_channels"],
            _combined[_col],
            color=_color,
            marker=_marker,
            linestyle=_ls,
            label=_stage_lbl,
        )

    if _has_data:
        _ax.set_xlabel("Number of channels")
        _ax.set_ylabel("Throughput (GB/s)")
        _ax.set_title("Pipeline stage throughput comparison (bottleneck identification)")
        _ax.legend(loc="best", ncol=2)
    else:
        _ax.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=_ax.transAxes, color="#999")
        _ax.set_title("Pipeline stage throughput comparison")
        mo.md("*No data available for bottleneck comparison.*")

    plt.tight_layout()
    fig4
    return (fig4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Figure 5 — Packet Capture Throughput")
    return


@app.cell
def _(capture_data, mo, plt):
    fig5, _ax = plt.subplots(figsize=(2.5, 2.2))

    if capture_data and "packets/sec" in capture_data:
        _pps = capture_data["packets/sec"] / 1e6
        _gbps = capture_data.get("GB/sec", 0.0)
        _bars = _ax.bar(
            ["Capture"],
            [_pps],
            color="#1f77b4",
            width=0.4,
            zorder=3,
        )
        _ax.bar_label(_bars, fmt="%.2f Mpkt/s", padding=3, fontsize=8)
        _ax.set_ylabel("Throughput (Mpkt/s)")
        _ax.set_title("Kernel-socket capture\n(recvmmsg, loopback)")
        _ax.set_ylim(0, _pps * 1.3)
        _ax.tick_params(axis="x", which="both", bottom=False)
        if _gbps:
            _ax.text(
                0, _pps * 0.5,
                f"{_gbps:.3f} GB/s",
                ha="center", va="center",
                color="white", fontsize=8, fontweight="bold",
            )
    else:
        _ax.text(0.5, 0.5, "No capture data", ha="center", va="center",
                 transform=_ax.transAxes, color="#999")
        _ax.set_title("Kernel-socket capture")
        mo.md("*No capture data available.*")

    plt.tight_layout()
    fig5
    return (fig5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Save Figures")
    return


@app.cell
def _(
    NOTEBOOK_DIR,
    Path,
    fig1,
    fig2,
    fig3,
    fig4,
    fig5,
    figures_dir_input,
    mo,
    save_switch,
):
    _figs_dir = Path(figures_dir_input.value)
    if not _figs_dir.is_absolute():
        _figs_dir = NOTEBOOK_DIR / _figs_dir

    if save_switch.value:
        _figs_dir.mkdir(parents=True, exist_ok=True)
        _saved = []
        _named_figs = [
            ("processor",    fig1),
            ("gpu_pipeline", fig2),
            ("writers",      fig3),
            ("bottleneck",   fig4),
            ("capture",      fig5),
        ]
        for _name, _fig in _named_figs:
            if _fig is None:
                continue
            for _ext in ("pdf", "png"):
                _p = _figs_dir / f"fig_{_name}.{_ext}"
                _fig.savefig(_p)
                _saved.append(_p.name)
        _list = "\n".join(f"- `{n}`" for n in _saved)
        mo.md(f"Saved {len(_saved)} files to `{_figs_dir}/`:\n\n{_list}")
    else:
        mo.md(
            f"Toggle **Save figures** above to export all figures to `{_figs_dir}/`  \n"
            f"(PDF at 300 DPI + PNG at 300 DPI)"
        )
    return


if __name__ == "__main__":
    app.run()

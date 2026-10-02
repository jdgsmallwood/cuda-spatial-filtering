#!/usr/bin/env python3
"""Build data-backed figures for docs/calibration-issues-report.tex.

Run with the lambda-commissioning virtualenv, which supplies python-casacore.
The large HDF5 capture is read only at five times and three fine channels.
"""

import json
import shutil
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from casacore.tables import table


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "build/calibration_runs/crab_240mhz_20260923"
OUT = ROOT / "docs/figures/calibration-report"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.2})


def save(name):
    plt.savefig(OUT / f"{name}.pdf")
    plt.close()


# A Measurement Set is a structured table; casacore stores it, CASA solves it.
fig, ax = plt.subplots(figsize=(8.1, 2.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis("off")


def box(x, y, w, h, heading, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                facecolor=color, edgecolor="0.25"))
    ax.text(x + w / 2, y + h * .7, heading, ha="center", va="center",
            weight="bold", fontsize=8)
    ax.text(x + w / 2, y + h * .32, body, ha="center", va="center", fontsize=6)


box(.1, 1.35, 2.15, 1.3, "HDF5 capture", "complex visibilities", "#dbeafe")
box(2.85, 1.15, 2.6, 1.7, "Measurement Set", "DATA / MODEL / FLAG\nUVW + subtables", "#e0f2fe")
box(6.15, 1.35, 2.15, 1.3, "CASA K/B/G", "delay / bandpass / gain", "#fef3c7")
box(8.95, 1.35, 2.5, 1.3, "Beam weights", "inverse gains + steering", "#dcfce7")
for x1, x2 in ((2.29, 2.75), (5.5, 6.05), (8.34, 8.85)):
    ax.add_patch(FancyArrowPatch((x1, 2), (x2, 2), arrowstyle="-|>",
                                 mutation_scale=15, linewidth=1.3, color="0.25"))
ax.text(4.15, .46, "casacore: table layout + sky/time/frequency reference frames",
        ha="center", fontsize=9, color="#075985")
ax.text(7.18, 3.42, "CASA tasks read DATA / MODEL_DATA and write calibration tables",
        ha="center", fontsize=8, color="#92400e")
save("measurement-set-flow")

# The exact projected uv lengths present in the exported 240 MHz MS.
with table(str(DATA / "crab_240mhz_cal_5min.ms"), ack=False) as ms:
    uvw = np.asarray(ms.getcol("UVW"), dtype=float)
uvmetres = np.linalg.norm(uvw[:, :2], axis=1)
cut = 7.0
fig, ax = plt.subplots(figsize=(7.3, 2.5))
ax.hist(uvmetres, bins=np.arange(0, 36, 1), color="#0e7490", edgecolor="white")
ax.axvspan(0, cut, color="#dc2626", alpha=.14, label="excluded from the >7 m solve")
ax.axvline(cut, color="#b91c1c", ls="--", lw=1.2)
ax.set(xlabel="Projected baseline length $\sqrt{u^2+v^2}$ (m)",
       ylabel="MS visibility rows", xlim=(0, 35),
       title="240 MHz capture: short-baseline selection")
ax.text(.98, .91, f"{np.mean(uvmetres < cut):.1%} of {len(uvmetres):,} rows below 7 m",
        ha="right", transform=ax.transAxes,
        bbox=dict(facecolor="white", edgecolor="0.8", alpha=.9))
ax.legend(loc="upper right", bbox_to_anchor=(1, .8), fontsize=8)
save("uv-baselines")

# Read actual CASA table flags rather than transcribing summary numbers.
trials = [("All-sky\nS/N $\geq 3$", "casa_cal"),
          ("All-sky\nS/N $\geq 1$", "casa_cal_minsnr1"),
          ("Clean / phase\nfinal", "casa_cal_clean31_phase"),
          ("Tau A point\n$uv>7$ m", "casa_cal_taua_7m")]
stages = [("K", "delay.K"), ("B", "bandpass.B"),
          ("phase G", "phase.G"), ("final G", "gain.G")]
coverage = np.zeros((len(trials), len(stages)))
for row, (_, directory) in enumerate(trials):
    for col, (_, name) in enumerate(stages):
        with table(str(DATA / directory / name), ack=False) as tab:
            flags = np.asarray(tab.getcol("FLAG"), dtype=bool)
        coverage[row, col] = 100 * np.mean(~flags)
fig, ax = plt.subplots(figsize=(7.3, 2.7))
x = np.arange(len(trials))
colors = ("#0284c7", "#7c3aed", "#d97706", "#059669")
for j, ((label, _), color) in enumerate(zip(stages, colors)):
    ax.bar(x + (j - 1.5) * .18, coverage[:, j], .17, label=label, color=color)
ax.set(xticks=x, xticklabels=[name for name, _ in trials], ylim=(0, 103),
       ylabel="Unflagged solution cells (%)", title="240 MHz CASA solution coverage")
ax.legend(ncol=4, fontsize=8, loc="upper center")
save("casa-coverage")

# Coherence samples follow the capture's embedded stream map, not a mutable map.
with h5py.File(DATA / "crab_240mhz_cal_5min.hdf5", "r") as capture:
    pairs = np.asarray(capture["baseline_antenna_ids"])
    manifest = json.loads(capture["audit/run_manifest_json"][()])
    fpga = {entry["antenna_id"]: entry["fpga_id"]
            for entry in manifest["forward_mapping"] if entry["antenna_id"] > 0}
    positions = {int(antenna): np.array((p["east"], p["north"]))
                 for group in manifest["normalized"]["config"]["antenna_positions"].values()
                 for antenna, p in group.items()}
    auto = {int(a): i for i, (a, b) in enumerate(pairs) if a == b}
    excluded = {1, 11, 14, 20, 36}
    chosen = []
    for index, (a, b) in enumerate(pairs):
        a, b = int(a), int(b)
        if (a == b or a in excluded or b in excluded or a not in auto or b not in auto
                or a not in positions or b not in positions):
            continue
        if 5 <= np.linalg.norm(positions[a] - positions[b]) <= 10:
            chosen.append((index, a, b, tuple(sorted((fpga[a], fpga[b])))))
    samples = {(hand, i, j): [] for hand in (0, 1) for i in range(4) for j in range(i, 4)}
    for time_index in np.linspace(0, capture["visibilities"].shape[0] - 1, 5, dtype=int):
        for channel in (13, 349, 657):
            v = capture["visibilities"][int(time_index), channel]
            for index, a, b, (i, j) in chosen:
                for hand in (0, 1):
                    denom = v[auto[a], hand, hand, 0] * v[auto[b], hand, hand, 0]
                    if denom <= 0:
                        continue
                    z = (v[index, hand, hand, 0] + 1j * v[index, hand, hand, 1]) / np.sqrt(denom)
                    if np.isfinite(z):
                        samples[(hand, i, j)].append(abs(z))
medians = np.full((2, 4, 4), np.nan)
for (hand, i, j), values in samples.items():
    if values:
        medians[hand, i, j] = medians[hand, j, i] = np.median(values)
fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.0), constrained_layout=True)
for hand, ax in enumerate(axes):
    im = ax.imshow(medians[hand], norm=LogNorm(vmin=.004, vmax=.16), cmap="viridis")
    ax.set(xticks=range(4), yticks=range(4), xlabel="FPGA", ylabel="FPGA",
           title=("raw hand 0 (YY)", "raw hand 1 (XX)")[hand])
    for i in range(4):
        for j in range(4):
            value = medians[hand, i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                        color="white" if value < .025 or value > .08 else "black", fontsize=7)
fig.colorbar(im, ax=axes, shrink=.8, label="Median $|V_{ij}|/\sqrt{V_{ii}V_{jj}}$")
save("cross-fpga-coherence")

# Illustrative Fourier response: point-like Crab versus large-scale emission.
frequency = 240.234375e6
wavelength = 299792458 / frequency
metres = np.linspace(0, 34.24, 500)
uv = metres / wavelength
fig, ax = plt.subplots(figsize=(7.3, 2.5))
ax.plot(metres, np.ones_like(metres), label="point source", color="#111827")
for fwhm_deg, label, color in ((5/60, "5 arcmin Gaussian", "#0e7490"),
                               (1, "1 degree Gaussian", "#d97706"),
                               (5, "5 degree Gaussian", "#7c3aed")):
    theta = np.deg2rad(fwhm_deg)
    response = np.exp(-((np.pi * theta * uv) ** 2) / (4 * np.log(2)))
    ax.plot(metres, response, label=label, color=color)
ax.axvline(7, color="#b91c1c", ls="--", lw=1, label="7 m solve cut")
ax.set(xlabel="Projected baseline (m) at 240.2 MHz", ylabel="Relative visibility amplitude",
       xlim=(0, 34.24), ylim=(0, 1.07), title="What a short array resolves: schematic Gaussian models")
ax.legend(ncol=2, fontsize=8)
save("model-angular-scale")

gallery = Path("/data/BLACKMESA_1/sma075/lambda-commissioning/runs/"
               "ibverbs-40ch-20260921/swap_fixed_gallery.png")
shutil.copyfile(gallery, OUT / "low-band-weighting-gallery.png")
print(f"Wrote five new PDF figures and one copied gallery to {OUT}")

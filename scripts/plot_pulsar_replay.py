#!/usr/bin/env python3
"""Produce the figures for docs/pulsar-replay-report.tex from replay products."""

import json
import pathlib
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "build/pulsar_replay_casacore"
OUT = ROOT / "docs/figures/pulsar-replay"
OUT.mkdir(parents=True, exist_ok=True)
TRUTH = json.loads((DATA / "truth.json").read_text())
plt.rcParams.update({"font.size": 9, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25})


def save(name):
    plt.savefig(OUT / f"{name}.pdf")
    plt.close()


def profile(path):
    result = subprocess.run(
        ["pdv", "-FTt", str(path)],
        check=True, capture_output=True, text=True)
    values = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "0" and parts[1] == "0":
            values.append(sum(map(float, parts[3:])))
    values = np.asarray(values)
    if values.size != 128:
        raise RuntimeError(f"Unexpected profile length {values.size}: {path}")
    return values


# Injection: use the exact envelope equation from create_pulsar_replay.py.
t = np.linspace(0, 1.5, 4000)
phase = np.remainder(t, TRUTH["period_seconds"]) / TRUTH["period_seconds"]
distance = np.minimum(phase, 1 - phase)
power = 1 + 12 * np.exp(-0.5 * (distance / 0.035) ** 2)
fig, ax = plt.subplots(figsize=(7.0, 2.0))
ax.plot(t, power, color="#155e75", lw=1.7)
ax.set(xlabel="Time from replay start (s)", ylabel="Relative source power",
       xlim=(0, 1.5), ylim=(0, 14), title="Injected 0.5 s pulse train")
save("injected-envelope")

# Geometry comes from the same casacore ENU direction as pipeline steering.
delays = np.asarray(TRUTH["delay_seconds_by_fpga_receiver"]) * 1e9
fig, ax = plt.subplots(figsize=(7.0, 2.2))
for fpga in range(delays.shape[0]):
    ax.scatter(np.arange(delays.shape[1]) + 10 * fpga, delays[fpga],
               s=22, label=f"FPGA {fpga}")
ax.axhline(0, color="0.25", lw=0.7)
ax.set(xlabel="FPGA receiver index", ylabel="Geometric delay (ns)",
       title=f"Casacore direction: azimuth {TRUTH['az_deg']:.2f}°, altitude {TRUTH['alt_deg']:.2f}°")
ax.legend(ncol=4, fontsize=8, loc="lower right")
save("geometric-delays")

gains = json.loads((DATA / "measured_fine_gains.json").read_text())["weights"]["176"]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
for pol, color in (("XX", "#155e75"), ("YY", "#d97706")):
    keys = sorted(gains[pol], key=int)
    z = np.array([complex(gains[pol][k]["real"], gains[pol][k]["imag"]) for k in keys])
    ants = np.array(list(map(int, keys)))
    axes[0].scatter(ants, abs(z), s=13, label=pol, color=color)
    axes[1].scatter(ants, np.angle(z), s=13, label=pol, color=color)
axes[0].set(xlabel="Antenna", ylabel="Measured gain amplitude")
axes[1].set(xlabel="Antenna", ylabel="Measured gain phase (rad)")
axes[0].legend(fontsize=8)
fig.suptitle("Injected antenna gains at coarse channel 176", y=1.04)
save("antenna-gains")

# Saved DADA voltage, folded directly as a cross-check on the input to DSPSR.
raw = next((DATA / "voltage_raw").glob("*.dada"))
array = np.memmap(raw, dtype="<f4", mode="r", offset=4096)
usable = array.size // (28 * 2 * 2)
array = array[:usable * 28 * 2 * 2].reshape(usable, 28, 2, 2)
sample_power = np.mean(array.astype("float64") ** 2, axis=(1, 2, 3))
phase = (np.arange(usable) * 34.56e-6 / TRUTH["period_seconds"]) % 1
edges = np.linspace(0, 1, 129)
bin_index = np.minimum(np.searchsorted(edges, phase, side="right") - 1, 127)
mean = np.bincount(bin_index, weights=sample_power, minlength=128) / np.bincount(bin_index, minlength=128)
fig, ax = plt.subplots(figsize=(7.0, 2.0))
ax.plot((edges[:-1] + edges[1:]) / 2, mean / np.median(mean), color="#155e75")
ax.set(xlabel="Injected pulse phase", ylabel="Power / median", xlim=(0, 1),
       title="Pulse present in the saved complex-voltage DADA stream")
save("dada-power-fold")

paths = [("GPU intensity", DATA / "intensity_final/pulsar.ar", "#d97706"),
         ("GPU voltage, PP+QQ", DATA / "voltage_sum_final/pulsar.ar", "#155e75"),
         ("CPU voltage check", DATA / "voltage_cpu_run2/pulsar.ar", "#059669")]
fig, ax = plt.subplots(figsize=(7.0, 2.3))
for label, path, color in paths:
    if not path.exists():
        continue
    y = profile(path)
    y = (y - np.median(y)) / (np.max(y) - np.median(y))
    y = np.roll(y, 32 - np.argmax(y))
    ax.plot(np.arange(128) / 128, y, label=label, color=color, lw=1.5)
ax.set(xlabel="Archive pulse phase", ylabel="Normalized profile", xlim=(0, 1),
       title="DSPSR recovered the periodic source (peaks aligned)")
ax.legend(fontsize=8)
save("dspsr-profiles")

# The fault isolation: both implementations agree before the CUDA detector.
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
for ax, stem in zip(axes, ("pre_Convolution", "pre_Detection")):
    cpu = np.fromfile(DATA / "dump_cpu" / f"{stem}.dump", dtype="<f4", offset=4096)
    gpu = np.fromfile(DATA / "dump_gpu" / f"{stem}.dump", dtype="<f4", offset=4096)
    ax.scatter(cpu[::100], gpu[::100], s=2, alpha=0.35)
    extent = max(abs(cpu).max(), abs(gpu).max())
    ax.plot([-extent, extent], [-extent, extent], color="black", lw=0.7)
    ax.set(xlabel="CPU sample", ylabel="GPU sample", title=stem.replace("pre_", "Before "))
fig.suptitle("GPU and CPU voltage streams agree before detection", y=1.04)
fig.subplots_adjust(top=0.77, wspace=0.3)
save("gpu-cpu-stage-comparison")

# The old in-place CUDA result versus the separate-buffer result.
bad = profile(DATA / "voltage_run3/pulsar.ar")
good = profile(DATA / "voltage_gpu_final/pulsar.ar")
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True)
for ax, y, title, color in ((axes[0], bad, "Original in-place GPU fold", "#b91c1c"),
                             (axes[1], good, "Separate-buffer GPU fold", "#155e75")):
    y = (y - np.median(y)) / (np.max(y) - np.median(y))
    ax.plot(np.arange(128) / 128, y, color=color, lw=1.3)
    ax.set(xlim=(0, 1), ylim=(-0.07, 1.07), xlabel="Archive phase", title=title)
axes[0].set_ylabel("Profile / peak")
fig.subplots_adjust(wspace=0.15)
save("gpu-before-after")

# Explain exactly why an in-place float2 -> float write is unsafe in CUDA.
fig, axes = plt.subplots(2, 1, figsize=(7.0, 2.4))
labels = ["Re 0", "Im 0", "Re 1", "Im 1", "Re 2", "Im 2", "Re 3", "Im 3"]
for ax, name in zip(axes, ("Original: aliased input and output", "Fix: separate output buffer")):
    ax.set(xlim=(-0.1, 8.1), ylim=(-0.2, 1.25), title=name)
    ax.axis("off")
    for i, label in enumerate(labels):
        ax.add_patch(Rectangle((i, 0.45), 0.95, 0.48, edgecolor="0.3", facecolor="#e0f2fe"))
        ax.text(i + .475, .69, label, ha="center", va="center", fontsize=8)
    for i in range(4):
        ax.add_patch(Rectangle((i, -0.08), 0.95, 0.42, edgecolor="0.3",
                               facecolor="#fee2e2" if ax is axes[0] and i == 2 else "#dcfce7"))
        ax.text(i + .475, .13, f"P {i}", ha="center", va="center", fontsize=8)
axes[0].annotate("P 2 write clobbers Re 1", xy=(2.48, .39), xytext=(4.2, .07),
                 arrowprops=dict(arrowstyle="->", color="#b91c1c"), color="#b91c1c", fontsize=8)
axes[1].text(4.3, .10, "Output writes cannot change unread input", fontsize=8, color="#166534")
fig.subplots_adjust(hspace=0.65)
save("cuda-buffer-overlap")

print(f"Wrote eight PDF figures to {OUT}")

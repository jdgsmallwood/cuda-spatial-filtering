#!/usr/bin/env python3
"""Compare raw FPGA PCAPs with the ADC-box PN sequence off and on."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, fftshift, next_fast_len

from lambda_commissioning.utils import _LAMBDA_PACKET_DTYPE, get_udp_payload_bytes

SAMPLE_SECONDS = 1.08e-6
CHANNEL = 404
LINE_HZ = -134277.34375


def read_channel(path):
    _, payloads = get_udp_payload_bytes(str(path), packets=20000)
    packets = np.frombuffer(b"".join(payloads), dtype=_LAMBDA_PACKET_DTYPE)
    return np.sort(packets[packets["freq_chan"] == CHANNEL], order="seq_no")


def voltages(packets):
    raw = packets["samples"].astype(np.float32)
    return (raw[..., 0] + 1j * raw[..., 1]) * packets["scales"][:, None, :, :]


def spectrum(packets, receiver=1, pol=0):
    n = (packets["seq_no"] - packets["seq_no"][0]).astype(np.int64)
    values = voltages(packets)[:, :, receiver, pol]
    samples = np.zeros(int(n.max()) + 64, np.complex64)
    samples[(n[:, None] + np.arange(64)).ravel()] = values.ravel()
    size = next_fast_len(len(samples))
    transformed = fftshift(fft(samples, size))
    frequency = fftshift(fftfreq(size, d=SAMPLE_SECONDS))
    return frequency, abs(transformed) ** 2 / values.size


def line_phasors(records, delays):
    counter = [r["seq_no"].astype(np.int64) - delays[i]
               for i, r in enumerate(records)]
    lo = max(n.min() for n in counter)
    hi = min(n.max() for n in counter)
    if hi - lo < 30000:
        raise ValueError("too little common counter time across boards")
    phasors = []
    for n, r in zip(counter, records):
        use = (n >= lo) & (n < hi)
        sample_index = (n[use, None] + np.arange(64) - lo).ravel()
        weight = np.exp(-2j * np.pi * LINE_HZ * SAMPLE_SECONDS * sample_index)
        phasors.append(weight @ voltages(r)[use].reshape(-1, 20))
    return np.array(phasors), int(hi - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcap_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--delays", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    off = read_channel(args.pcap_root / "fpga0_off/fpga0.pcapng")
    on = read_channel(args.pcap_root / "fpga0_on/fpga0.pcapng")
    f_off, p_off = spectrum(off)
    f_on, p_on = spectrum(on)
    targets = -329589.84375 + 195312.5 * np.arange(4)
    measured = []
    strengths = []
    for target in targets:
        near = np.flatnonzero(abs(f_on - target) < 5000)
        peak = near[np.argmax(p_on[near])]
        measured.append(float(f_on[peak]))
        strengths.append(float(p_on[peak] / np.median(p_on)))
    spacing = float(np.polyfit(np.arange(4), measured, 1)[0])
    delays = {int(k): int(v) for k, v in json.loads(args.delays.read_text()).items()}
    phase_sets = []
    durations = []
    for label in ("all_on", "all_on_repeat"):
        records = [read_channel(args.pcap_root / label / f"fpga{i}.pcapng")
                   for i in range(4)]
        phasor, span = line_phasors(records, delays)
        phase_sets.append(phasor / phasor[0, 2])
        durations.append(span * SAMPLE_SECONDS)
    phase_change = np.angle(phase_sets[1] / phase_sets[0])
    absolute = abs(phase_change)
    report = {
        "raw_pcap_root": str(args.pcap_root),
        "coarse_channel": CHANNEL,
        "centre_frequency_hz": CHANNEL * 781250,
        "input": "FPGA 0, local receiver 1, polarization 0 for spectrum",
        "comb_peak_offsets_hz": measured,
        "comb_spacing_hz": spacing,
        "comb_peak_to_median_power": strengths,
        "phase_line_offset_hz": LINE_HZ,
        "phase_reference": "FPGA 0, local receiver 1, polarization 0",
        "phase_repeatability_radians": {
            "median_absolute": float(np.median(absolute)),
            "p90_absolute": float(np.quantile(absolute, .9)),
            "max_absolute": float(absolute.max()),
            "median_by_fpga": [float(x) for x in np.median(absolute, axis=1)],
        },
        "common_counter_span_s": durations,
        "limitation": "ADC-box PN injection tests the signal chain from its injection point downstream; it does not measure antenna/sky/front-end gain. A one-line phase does not establish a wideband delay.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout="constrained")
    axes[0].plot(f_off / 1e3, 10 * np.log10(p_off / np.median(p_off)),
                 lw=.6, alpha=.8, label="PN off")
    axes[0].plot(f_on / 1e3, 10 * np.log10(p_on / np.median(p_on)),
                 lw=.6, alpha=.8, label="PN on")
    for peak in measured:
        axes[0].axvline(peak / 1e3, color="tab:red", alpha=.35, ls="--")
    axes[0].set(xlabel=f"Offset from coarse {CHANNEL} centre (kHz)",
                ylabel="Power / median (dB)",
                title=f"Raw packet spectrum; comb spacing {spacing / 1e3:.2f} kHz",
                xlim=(-390, 390), ylim=(-10, 40))
    axes[0].grid(alpha=.25)
    axes[0].legend()
    for board in range(4):
        axes[1].scatter(np.arange(20) + board * 20, phase_change[board],
                        s=18, label=f"FPGA {board}")
    axes[1].axhline(0, color="0.5", lw=1)
    axes[1].set(xlabel="Raw input/polarization index (20 per FPGA)",
                ylabel="Phase change between PN-on runs (rad)",
                title=f"Relative phase repeatability at {LINE_HZ / 1e3:.2f} kHz")
    axes[1].grid(alpha=.25)
    axes[1].legend(ncol=4)
    fig.savefig(args.output / "pn_comb_and_phase.png", dpi=170)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

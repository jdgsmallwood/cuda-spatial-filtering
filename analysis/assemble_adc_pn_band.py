#!/usr/bin/env python3
"""Assemble independent eight-channel ADC PN fits into a diagnostic band table."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fit_root", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    groups = []
    for start in (400, 408, 416, 424, 432):
        source = args.fit_root / ("long_compare" if start == 400 else f"full40_{start}")
        with np.load(source / "pn_multitone.npz") as z:
            ch = z["channels"].astype(int)
            phase = z["coarse_mean_phase_rad"]
        if not np.array_equal(ch, np.arange(start, start + 8)):
            raise ValueError(f"Unexpected channels in {source}")
        groups.append((ch, phase))
    channels = np.concatenate([g[0] for g in groups])
    runs = np.concatenate([g[1] for g in groups], axis=1)
    repeat_error = abs(np.angle(np.exp(1j * (runs[1] - runs[0]))))
    phase = np.angle(np.mean(np.exp(1j * runs), axis=0))
    # This is a quality flag, not an instruction to apply the table online.
    stable = repeat_error <= .5
    median = np.median(repeat_error, axis=1)
    p90 = np.quantile(repeat_error, .9, axis=1)
    report = {
        "channels": [int(channels[0]), int(channels[-1])],
        "phase_reference": "FPGA 0, receiver 1, raw polarization 0, separately in each coarse channel",
        "meaning": "Apparent ADC-box PN phase at four selected comb teeth, averaged within coarse channel and across two PN-on toggles; diagnostic only",
        "median_repeat_error_rad_all_channels_inputs": float(np.median(repeat_error)),
        "p90_repeat_error_rad_all_channels_inputs": float(np.quantile(repeat_error, .9)),
        "median_repeat_error_rad_by_eight_channel_group": [float(np.median(repeat_error[i:i+8])) for i in range(0, 40, 8)],
        "number_of_channel_input_pairs_over_half_rad": int(np.sum(~stable)),
        "total_channel_input_pairs": int(stable.size),
        "limitation": "Inputs on separate boards may see independent PN-generator phase. Coarse-centre phase does not imply the same fine-bin delay. Antenna/front-end is bypassed by this injection."
    }
    np.savez_compressed(args.output / "pn_full40.npz", channels=channels,
                        phase_rad=phase, repeat_error_rad=repeat_error,
                        stable_within_half_rad=stable,
                        median_repeat_error_rad=median,
                        p90_repeat_error_rad=p90)
    (args.output / "pn_full40_report.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), layout="constrained")
    axes[0].plot(channels, median, ".-", label="median over 80 raw streams")
    axes[0].plot(channels, p90, ".-", label="90th percentile")
    axes[0].axhline(.5, color="tab:red", ls="--", label="individual-stream quality threshold")
    axes[0].set(ylabel="PN phase change between toggles (rad)",
                title="Full 40-channel PN phase repeatability")
    axes[0].grid(alpha=.25)
    axes[0].legend()
    im = axes[1].imshow(repeat_error.T, origin="lower", aspect="auto",
                        extent=(channels[0] - .5, channels[-1] + .5, -.5, 79.5),
                        vmin=0, vmax=1, cmap="magma")
    axes[1].set(xlabel="Coarse channel", ylabel="Raw stream (20 per FPGA)")
    fig.colorbar(im, ax=axes[1], label="Absolute phase change (rad)")
    fig.savefig(args.output / "pn_full40_repeatability.png", dpi=170)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

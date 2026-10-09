#!/usr/bin/env python3
"""Compare raw 14-bit ADC occupancy before and after a controlled gain change.

The ADC capture stores signed 14-bit codes in bits 15..2 of int16 words. This
checks the analog capture itself; packet int8 values and packet scale factors
are a separate diagnostic.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_gain(directory):
    rows = []
    for board in range(4):
        for channel in range(20):
            matches = list(directory.glob(f"adc_raw_alveo{board}_chan{channel}_*.npy"))
            if len(matches) != 1:
                raise ValueError(f"Expected one raw capture for FPGA {board} ADC {channel} in {directory}; found {len(matches)}")
            samples = np.load(matches[0]).astype(np.int32) >> 2
            if len(samples) != 65536:
                raise ValueError(f"Unexpected raw capture length in {matches[0]}")
            absolute = abs(samples)
            rows.append({"fpga": board, "adc_channel": channel,
                         "rms_code": float(np.sqrt(np.mean(samples.astype(np.float64)**2))),
                         "median_abs_code": float(np.median(absolute)),
                         "p99_abs_code": float(np.quantile(absolute, .99)),
                         "p9999_abs_code": float(np.quantile(absolute, .9999)),
                         "max_abs_code": int(absolute.max()),
                         "fraction_near_full_scale": float(np.mean(absolute >= 8188)),
                         "fraction_zero": float(np.mean(samples == 0)),
                         "path": str(matches[0])})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("trial_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline-gain", type=int, default=20)
    parser.add_argument("--trial-gain", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    before = load_gain(args.baseline_dir)
    trial = load_gain(args.trial_dir)
    ratio = np.array([b["rms_code"] / a["rms_code"] if a["rms_code"] > 0 else np.nan
                      for a, b in zip(before, trial)])
    near = np.array([r["fraction_near_full_scale"] for r in trial])
    report = {
        "baseline_gain_setting": args.baseline_gain,
        "trial_gain_setting": args.trial_gain,
        "gain_direction": "0 is maximum gain; a lower setting increases analog gain",
        "adc_full_scale_code": [-8192, 8191],
        "per_channel": [{"fpga": a["fpga"], "adc_channel": a["adc_channel"],
                         "baseline": a, "trial": b, "rms_ratio": float(r)}
                        for a, b, r in zip(before, trial, ratio)],
        "median_rms_ratio": float(np.nanmedian(ratio)),
        "p10_rms_ratio": float(np.nanquantile(ratio, .1)),
        "p90_rms_ratio": float(np.nanquantile(ratio, .9)),
        "maximum_trial_near_full_scale_fraction": float(near.max()),
        "number_of_channels_with_any_near_full_scale_sample": int(np.count_nonzero(near)),
        "interpretation_limit": "One 65,536-sample snapshot per ADC/channel may miss rare clipping and does not establish improved sky SNR. Compare repeated raw captures, packet scales, and held-out sky coherence before keeping the setting.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout="constrained")
    for board in range(4):
        sl = slice(board * 20, (board + 1) * 20)
        axes[0].plot(np.arange(20) + board * 20, ratio[sl], ".", label=f"FPGA {board}")
        axes[1].plot(np.arange(20) + board * 20,
                     [r["max_abs_code"] for r in trial[sl]], ".")
    axes[0].set(ylabel="Trial / baseline raw ADC RMS", title="Measured analog gain change")
    axes[1].axhline(8188, color="tab:red", ls="--", label="near full scale")
    axes[1].set(xlabel="FPGA × 20 + ADC channel", ylabel="Trial maximum |ADC code|",
                title="Clipping headroom in 65,536-sample snapshots")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(args.output / "raw_adc_gain_comparison.png", dpi=170)
    print(json.dumps({key: report[key] for key in
                      ("median_rms_ratio", "p10_rms_ratio", "p90_rms_ratio",
                       "maximum_trial_near_full_scale_fraction")}, indent=2))


if __name__ == "__main__":
    main()

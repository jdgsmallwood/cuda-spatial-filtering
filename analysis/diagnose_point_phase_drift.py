#!/usr/bin/env python3
"""Test board-common versus per-antenna sky phase drift on held baselines.

Uses the existing point-source sufficient statistics. Early and middle thirds
set a phase-drift prediction for the late third; late training baselines are
used only for a common amplitude scale and the diagnostic same-time upper bound.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lambda_commissioning.point_calibration import solve
from lambda_commissioning.delay_bandpass import score


def fit_phase(h, k, train):
    weights = np.where(train, k, 0)
    data = np.divide(h, k, out=np.zeros_like(h), where=k > 0)
    return solve(data, np.ones_like(data), weights, phase_only=True)


def board_prediction(initial, current, boards, factor):
    change = np.angle(current * initial.conj())
    phase = np.zeros_like(change)
    for board in np.unique(boards):
        selected = boards == board
        phase[selected] = np.angle(np.mean(np.exp(1j * change[selected])))
    return initial * np.exp(1j * factor * phase)


def amplitude_for_training(h, k, phase, train):
    product = phase[:, None] * phase[None, :].conj()
    numerator = np.sum((h * product.conj()).real * train)
    denominator = np.sum(k * train)
    return np.sqrt(max(numerator / max(denominator, 1e-30), 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("statistics", type=Path)
    ap.add_argument("config", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.statistics) as z:
        frequency = z["frequency_hz"]
        ids = z["antenna_ids"].astype(int)
        h = z["h"]
        k = z["k"]
        dd = z["dd"]
        good = z["good"].astype(bool)
        held = z["held_baselines"].astype(bool)
        time = z["times_unix"]
        split = z["time_split"]
        retained = int(z["retained_fine_bins"])
    if retained != 27 or h.shape[0] != 3 or h.shape[1] != 2:
        raise ValueError("Expected three time thirds and production 27-bin output")
    if len(frequency) % retained:
        raise ValueError("Incomplete coarse-channel layout")
    conf = json.loads(args.config.read_text())
    board_of = {int(antenna): int(board)
                for board, antennas in conf["antenna_positions"].items()
                for antenna in antennas}
    boards = np.array([board_of[antenna] for antenna in ids])
    centre = np.arange(len(frequency) // retained) * retained + (16 - 3)
    centre = centre[good[centre]]
    if not len(centre):
        raise ValueError("No accepted centre frequencies")
    train = ~held & ~np.eye(len(ids), dtype=bool)
    test = held & ~np.eye(len(ids), dtype=bool)
    midpoint = [float(np.median(time[split == i])) for i in range(3)]
    factor = (midpoint[2] - midpoint[0]) / (midpoint[1] - midpoint[0])
    labels = ("static_early", "previous_middle", "board_linear_prediction",
              "antenna_linear_prediction", "same_time_per_antenna")
    metrics = {pol: {name: [] for name in labels} for pol in ("XX", "YY")}
    delta = np.zeros((2, len(centre), 2, len(ids)))
    for pi, pol in enumerate(("XX", "YY")):
        for fi, channel in enumerate(centre):
            gains = [fit_phase(h[t, pi, channel], k[t, pi, channel], train)
                     for t in range(3)]
            delta[pi, fi, 0] = np.angle(gains[1] * gains[0].conj())
            delta[pi, fi, 1] = np.angle(gains[2] * gains[0].conj())
            candidates = (gains[0], gains[1],
                          board_prediction(gains[0], gains[1], boards, factor),
                          gains[0] * np.exp(1j * factor * delta[pi, fi, 0]),
                          gains[2])
            for label, phase in zip(labels, candidates):
                amplitude = amplitude_for_training(h[2, pi, channel],
                                                    k[2, pi, channel],
                                                    phase, train)
                result = score(h[2, pi, channel][None] * test,
                               k[2, pi, channel][None] * test,
                               dd[2, pi, channel][None] * test,
                               (phase * amplitude)[None])
                metrics[pol][label].append(result["correlation"])
    summary = {pol: {name: float(np.median(metrics[pol][name])) for name in labels}
               for pol in ("XX", "YY")}
    board_scatter = {}
    for pi, pol in enumerate(("XX", "YY")):
        board_scatter[pol] = {}
        for t, name in enumerate(("early_to_middle", "early_to_late")):
            board_scatter[pol][name] = {}
            for board in range(4):
                values = delta[pi, :, t][:, boards == board]
                mean = np.angle(np.mean(np.exp(1j * values)))
                residual = np.angle(np.exp(1j * (values - mean)))
                board_scatter[pol][name][str(board)] = {
                    "mean_phase_rad": float(mean),
                    "median_abs_within_board_residual_rad": float(np.median(abs(residual))),
                    "p90_abs_within_board_residual_rad": float(np.quantile(abs(residual), .9)),
                }
    report = {
        "statistics": str(args.statistics),
        "selected_centre_frequencies_mhz": (frequency[centre] / 1e6).tolist(),
        "time_midpoints_unix": midpoint,
        "early_middle_to_late_extrapolation_factor": factor,
        "held_baseline_median_correlation_by_frequency": summary,
        "held_baseline_correlation_by_frequency": metrics,
        "board_phase_change": board_scatter,
        "method": "Point-model phase solved separately in three time thirds on 80% of >=7m baselines; early->middle predicts late; late training baselines only set a common positive amplitude for all candidates; 20% of baselines scored independently.",
        "limitations": ["A changing unmodelled sky or stationary term can mimic gain phase drift.",
                        "Per-antenna predictions require independent late-sky validation before installation."],
    }
    (args.output / "point_phase_drift.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "point_phase_drift.npz", antenna_ids=ids,
                        board_ids=boards, frequency_hz=frequency[centre],
                        phase_change_rad=delta)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for pi, pol in enumerate(("XX", "YY")):
        for label in labels:
            axes[pi, 0].plot(frequency[centre] / 1e6, metrics[pol][label], ".-",
                             label=label)
        axes[pi, 0].set(title=f"{pol} late held-baseline prediction",
                        xlabel="Frequency (MHz)", ylabel="Correlation", ylim=(-1, 1))
        axes[pi, 0].grid(alpha=.25)
        axes[pi, 0].legend(fontsize=7)
        for board in range(4):
            selected = boards == board
            axes[pi, 1].scatter(ids[selected],
                np.angle(np.mean(np.exp(1j * delta[pi, :, 1][:, selected]), axis=0)),
                s=25, label=f"FPGA {board}")
        axes[pi, 1].set(title=f"{pol} early-to-late phase by FPGA",
                        xlabel="Antenna ID", ylabel="Circular mean phase change (rad)",
                        ylim=(-np.pi, np.pi))
        axes[pi, 1].grid(alpha=.25)
        axes[pi, 1].legend(fontsize=7)
    fig.savefig(args.output / "point_phase_drift.png", dpi=170)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

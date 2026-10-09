#!/usr/bin/env python3
"""Fit a shared within-coarse delay to PN teeth across adjacent coarse channels.

Fit channels 0..3 and predict the fourth comb tooth in channels 4..7.
Independent PN captures are fitted separately, so their delay agreement tests
stability over the interval between them. Each coarse has a free phase offset;
the fit makes no claim that a pre-channelization delay is separately identified.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def wrap(value):
    return np.angle(np.exp(1j * value))


def fit_shared_delay(phase, offsets_hz, grid_ns):
    """Return the delay maximizing circular line coherence within each coarse."""
    angle = phase[:, :, None] - 2 * np.pi * offsets_hz[None, :, None] * grid_ns[None, None] * 1e-9
    score = np.abs(np.exp(1j * angle).mean(axis=1)).sum(axis=0)
    return float(grid_ns[np.argmax(score)])


def held_tooth_error(phase, offsets_hz, delay_ns, held=3):
    train = np.arange(len(offsets_hz)) != held
    intercept = np.angle(np.exp(1j * (phase[train] -
                              2 * np.pi * offsets_hz[train] * delay_ns * 1e-9)).mean())
    prediction = intercept + 2 * np.pi * offsets_hz[held] * delay_ns * 1e-9
    return float(abs(wrap(phase[held] - prediction)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pn_multitone", type=Path)
    p.add_argument("output_dir", type=Path)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(a.pn_multitone) as data:
        phase = data["relative_phase_rad"]
        offsets = data["line_offsets_hz"]
        channels = data["channels"]
    if phase.shape != (2, 8, 4, 80) or offsets.shape != (4,):
        raise ValueError("requires two PN runs, eight coarse channels, four teeth, 80 streams")
    grid = np.arange(-540., 541., 1.)
    delay = np.empty((2, 80))
    for run in range(2):
        for stream in range(80):
            delay[run, stream] = fit_shared_delay(phase[run, :4, :, stream], offsets, grid)
    held = np.empty((2, 4, 80))
    constant = np.empty_like(held)
    for run in range(2):
        for ci in range(4, 8):
            for stream in range(80):
                x = phase[run, ci, :, stream]
                held[run, ci - 4, stream] = held_tooth_error(x, offsets, delay[run, stream])
                constant[run, ci - 4, stream] = held_tooth_error(x, offsets, 0.)
    repeat = np.abs(delay[1] - delay[0])
    # The reference stream is exactly zero by definition; omit it from
    # population summaries because it is not an independent measurement.
    independent = np.arange(80) != 2
    boards = {}
    for board in range(4):
        sl = np.arange(board * 20, (board + 1) * 20)
        sl = sl[independent[sl]]
        boards[str(board)] = dict(
            pre_median_delay_ns=float(np.median(delay[0, sl])),
            post_median_delay_ns=float(np.median(delay[1, sl])),
            median_abs_repeat_difference_ns=float(np.median(repeat[sl])),
            p90_abs_repeat_difference_ns=float(np.percentile(repeat[sl], 90)),
            held_tooth_error_rad=float(np.median(held[:, :, sl])),
            constant_phase_error_rad=float(np.median(constant[:, :, sl])))
    report = dict(channels=channels.tolist(), line_offsets_hz=offsets.tolist(),
                  fit_coarse_channels=channels[:4].tolist(),
                  held_coarse_channels=channels[4:].tolist(),
                  delay_search_ns=[-540, 540],
                  median_abs_delay_repeat_difference_ns=float(np.median(repeat[independent])),
                  p90_abs_delay_repeat_difference_ns=float(np.percentile(repeat[independent], 90)),
                  held_tooth_error_rad_by_run=[float(np.median(held[r, :, independent])) for r in range(2)],
                  constant_phase_error_rad_by_run=[float(np.median(constant[r, :, independent])) for r in range(2)],
                  boards=boards,
                  limitation="The PN injection point, phase reset at toggle, and analog input path are not independently known. The fitted delay is the aggregate within-coarse response from injection to output, modulo the comb ambiguity; it is not a pre/post-channelization split or a sky gain.")
    (a.output_dir / "joint_local_delay_report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(a.output_dir / "joint_local_delay.npz", channels=channels,
                        line_offsets_hz=offsets, delay_ns=delay,
                        repeat_abs_ns=repeat, held_tooth_error_rad=held,
                        constant_phase_error_rad=constant)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for board in range(4):
        sl = slice(board * 20, (board + 1) * 20)
        axes[0].scatter(delay[0, sl], delay[1, sl], label=f"FPGA {board}", s=18)
    axes[0].plot([-540, 540], [-540, 540], color="0.5", ls="--")
    axes[0].set(xlabel="First PN run: fitted local delay (ns)",
                ylabel="Second PN run: fitted local delay (ns)")
    axes[0].legend()
    axes[0].grid(alpha=.25)
    for run in range(2):
        axes[1].scatter(np.full(4, run), np.median(held[run], axis=1),
                        label=f"run {run + 1} shared delay")
        axes[1].scatter(np.full(4, run) + .15, np.median(constant[run], axis=1),
                        marker="x", label=f"run {run + 1} constant phase")
    axes[1].set(xticks=[0, 1], xticklabels=["first", "second"],
                ylabel="Held fourth-tooth phase error (rad)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.25)
    fig.savefig(a.output_dir / "joint_local_delay.png", dpi=170)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

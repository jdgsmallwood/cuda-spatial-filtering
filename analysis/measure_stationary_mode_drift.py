#!/usr/bin/env python3
"""Bound phase drift in the strong fixed YY mode without calling it a clock solve."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("modes", type=Path)
    p.add_argument("stream_map", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    with np.load(a.modes) as z:
        v = z["modes"][1]  # [coarse, time, antenna], YY
        utc = z["utc"]
        ids = z["antenna_ids"].astype(int)
        freqs = z["frequency_hz"]
        strong = z["strong"][1]
    if not np.all(strong):
        raise ValueError("fixed YY mode was not selected at every sampled time")
    mapping = json.loads(a.stream_map.read_text())["fpgas"]
    board_of = {int(s["antenna_id"]): int(board)
                for board, value in mapping.items()
                for s in value["streams"] if s["antenna_id"] > 0}
    reference = int(np.flatnonzero(ids == 1)[0])
    # Eigenvectors carry an arbitrary phase at every time/frequency. Divide by
    # the reference input before unwrapping in time.
    relative = v * v[:, :, reference, None].conj()
    elapsed = (utc - utc[0]) / 60
    duration = elapsed[-1]
    records = []
    for ai, antenna in enumerate(ids):
        mode_amplitude = float(np.median(np.abs(v[:, :, ai])))
        reliable = mode_amplitude >= .02
        slopes = []
        residual_rms = []
        for ci in range(len(freqs)):
            phase = np.unwrap(np.angle(relative[ci, :, ai]))
            trend = np.polyfit(elapsed, phase, 1)
            slopes.append(trend[0] * duration * 180 / np.pi)
            residual_rms.append(np.std((phase - np.polyval(trend, elapsed)) * 180 / np.pi))
        slopes = np.asarray(slopes)
        records.append(dict(antenna_id=int(antenna), fpga=board_of[int(antenna)],
                            median_mode_amplitude=mode_amplitude,
                            reliable_phase=reliable,
                            drift_over_capture_deg=float(np.median(slopes)) if reliable else None,
                            between_channel_drift_mad_deg=float(
                                np.median(np.abs(slopes - np.median(slopes)))) if reliable else None,
                            detrended_phase_rms_deg=float(np.median(residual_rms)) if reliable else None))
    boards = {str(board): dict(
        median_drift_deg=float(np.median([r["drift_over_capture_deg"]
                                         for r in records if r["fpga"] == board and r["reliable_phase"]])),
        within_board_drift_range_deg=float(np.ptp([r["drift_over_capture_deg"]
                                                   for r in records if r["fpga"] == board and r["reliable_phase"]])))
        for board in range(4)}
    report = dict(reference_antenna=1, duration_minutes=float(duration),
                  frequencies_hz=freqs.tolist(), boards=boards, antennas=records,
                  median_abs_antenna_drift_deg=float(np.median(
                      [abs(r["drift_over_capture_deg"]) for r in records if r["reliable_phase"]])),
                  unresolved_antenna_ids=[r["antenna_id"] for r in records if not r["reliable_phase"]],
                  limitation="Tracks the dominant stationary antenna-side YY signal."
                             " Includes source/front-end phase changes and gives no unique"
                             " FPGA clock-rate estimate or sky gain solution.")
    (a.output / "stationary_mode_drift.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for board in range(4):
        rr = [r for r in records if r["fpga"] == board and r["reliable_phase"]]
        ax.scatter([r["antenna_id"] for r in rr],
                   [r["drift_over_capture_deg"] for r in rr], label=f"FPGA {board}")
    ax.axhline(0, color="0.5", lw=.8)
    ax.set(xlabel="Antenna", ylabel="Phase drift over observation (degrees)",
           title="Dominant stationary YY mode, relative to antenna 1")
    ax.grid(alpha=.25)
    ax.legend(ncol=4)
    fig.savefig(a.output / "stationary_mode_drift.png", dpi=170)
    print(json.dumps({"boards": boards,
                      "median_abs_antenna_drift_deg": report["median_abs_antenna_drift_deg"]},
                     indent=2))


if __name__ == "__main__":
    main()

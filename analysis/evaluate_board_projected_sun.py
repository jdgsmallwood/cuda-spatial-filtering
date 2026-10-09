#!/usr/bin/env python3
"""Project the PN-plus-Sun gain improvement onto four FPGA-common phases.

Both input antenna fits were trained on the first half of the Sun observation.
The projection is computed from those fits and the independent PN table only;
the second half of the Sun observation is used solely for scoring.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from analysis.fit_antenna_sun_residual import FREF, FINE, score, wrap
except ModuleNotFoundError:
    from fit_antenna_sun_residual import FREF, FINE, score, wrap


def project_board_phase(antenna, board, frequency, sky_phase, sky_delay,
                        pn_sky_phase, pn_sky_delay, pn_phase):
    """Return the circular-mean, board-common part of the gain difference."""
    offset = 2 * np.pi * (frequency - FREF) * 1e-9
    difference = np.array([
        pn_phase[i] + pn_sky_phase[i] - sky_phase[i] +
        offset * (pn_sky_delay[i] - sky_delay[i])
        for i in range(len(antenna))])
    board_phase = np.zeros((4,) + frequency.shape)
    coherence = np.zeros((4,) + frequency.shape)
    for b in range(4):
        member = board == b
        if not member.any():
            raise ValueError(f"No supported antennas on FPGA {b}")
        mean = np.exp(1j * difference[member]).mean(axis=0)
        board_phase[b] = np.angle(mean)
        coherence[b] = abs(mean)
    board_phase = wrap(board_phase - board_phase[0])
    return board_phase, coherence


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sun_coefficients", type=Path)
    ap.add_argument("pn_candidate", type=Path)
    ap.add_argument("antenna_fit", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--mapping", type=Path, default=Path("stream_antenna_map.json"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.sun_coefficients) as z:
        pair = z["pairs"].astype(int)
        frequency = z["frequency_hz"]
        coefficient = z["coefficient"]
    with np.load(args.pn_candidate) as z:
        pn_table = z["phase_rad"]
        pn_freq = z["fine_frequency_hz"]
    with np.load(args.antenna_fit) as z:
        antenna = z["antenna_ids"]
        sky_phase = z["phase_sky_only"]
        sky_delay = z["delay_ns_sky_only"]
        pn_sky_phase = z["phase_pn_then_sky"]
        pn_sky_delay = z["delay_ns_pn_then_sky"]
        good_test = z["good_test"]
        held_baseline = z["held_baseline"]
        reference_sky_error = z["error_sky_only"]
        reference_pn_error = z["error_pn_then_sky"]
    if not np.allclose(frequency, pn_freq[:8, FINE]):
        raise ValueError("Sun and PN selected frequency grids differ")
    mapping = json.loads(args.mapping.read_text())["fpgas"]
    raw = {int(stream["antenna_id"]): int(board) * 20 + i
           for board, fpga in mapping.items()
           for i, stream in enumerate(fpga["streams"])
           if stream["antenna_id"] > 0 and stream["polarization"] == 0}
    board = np.array([raw[int(a)] // 20 for a in antenna])
    pn_phase = np.array([pn_table[:8, FINE, raw[int(a)]] for a in antenna])
    board_phase, coherence = project_board_phase(
        antenna, board, frequency, sky_phase, sky_delay,
        pn_sky_phase, pn_sky_delay, pn_phase)
    index = {int(a): i for i, a in enumerate(antenna)}
    offset = 2 * np.pi * (frequency - FREF) * 1e-9
    error = np.full(coefficient.shape[1:], np.nan)
    same = np.array([raw[int(a)] // 20 == raw[int(b)] // 20 for a, b in pair])
    for bi, (a, b) in enumerate(pair):
        if int(a) not in index or int(b) not in index:
            continue
        ai, bj = index[int(a)], index[int(b)]
        sky_prediction = (sky_phase[ai] - sky_phase[bj] +
                          offset * (sky_delay[ai] - sky_delay[bj]))
        shift = board_phase[board[ai]] - board_phase[board[bj]]
        error[:, :, bi] = wrap(np.angle(coefficient[1, :, :, bi]) -
                               sky_prediction - shift)
    test_freq = np.zeros(frequency.shape, bool)
    test_freq[4:] = True
    test = good_test & test_freq[:, :, None]
    groups = {"all": np.ones(len(pair), bool),
              "held_baselines": held_baseline,
              "held_cross_fpga": held_baseline & ~same,
              "held_same_fpga": held_baseline & same,
              "cross_fpga": ~same,
              "same_fpga": same}
    report = {
        "method": "Circular projection of independently trained PN-plus-Sun minus sky-only antenna gains onto one phase per FPGA and frequency; sky-only per-antenna gains retained",
        "training": "Both antenna gain fits use only first-half Sun coefficients on coarse 400-403 and independent PN phases; no second-half Sun coefficient enters board projection",
        "testing": "Second-half Sun coefficients on coarse 404-407; same quality cells for all models",
        "median_board_projection_coherence": np.median(coherence, axis=(1, 2)).tolist(),
        "scores": {name: {group: score(values, test & selected[None, None])
                          for group, selected in groups.items()}
                   for name, values in (("sky_only", reference_sky_error),
                                        ("per_stream_pn_then_sun", reference_pn_error),
                                        ("board_projected_pn_then_sun", error))},
        "limitation": "Offline YY phase test on coarse 400-407 only; amplitudes and full-band transfer untested",
    }
    (args.output / "board_projected_sun_report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "board_projected_sun.npz",
                        frequency_hz=frequency, board_phase_rad=board_phase,
                        board_projection_coherence=coherence, pairs=pair,
                        held_baseline=held_baseline, good_test=good_test,
                        error_board_projected=error)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")
    for name, values in (("sky_only", reference_sky_error),
                         ("per_stream_pn_then_sun", reference_pn_error),
                         ("board_projected_pn_then_sun", error)):
        y = []
        for ci in range(4, 8):
            mask = test[ci] & (held_baseline & ~same)[None]
            y.append(score(values[ci], mask)["median_cosine"])
        axs[0].plot(np.arange(404, 408), y, ".-", label=name)
    axs[0].set(xlabel="Held coarse channel", ylabel="Median phase cosine on held cross-FPGA baselines",
               ylim=(-1, 1))
    axs[0].grid(alpha=.3)
    axs[0].legend()
    for b in range(1, 4):
        axs[1].plot(frequency.ravel() / 1e6, board_phase[b].ravel(), ".", ms=3,
                    label=f"FPGA {b} relative to FPGA 0")
    axs[1].set(xlabel="Frequency (MHz)", ylabel="Projected board phase (rad)")
    axs[1].grid(alpha=.3)
    axs[1].legend()
    fig.savefig(args.output / "board_projected_sun.png", dpi=170)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

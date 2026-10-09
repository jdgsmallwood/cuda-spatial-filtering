#!/usr/bin/env python3
"""Fit antenna phase/delay residuals to moving-Sun visibilities.

The Sun fringe and stationary term were separated by test_pn_fine_sun.py.
Only the first time half, four coarse channels and a subset of fine channels
and baselines are used to fit. All reported scores use the later time half.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


FREF = 312.5e6
FINE = np.array([2, 6, 10, 13, 16, 20, 24])


def wrap(x):
    return np.angle(np.exp(1j * x))


def initial_phases(pair, antenna, observed, quality, weight):
    """Angular synchronization separately at each training frequency."""
    index = {a: i for i, a in enumerate(antenna)}
    n = len(antenna)
    phases = np.full((observed.shape[0], n), np.nan)
    for fi in range(observed.shape[0]):
        matrix = np.zeros((n, n), complex)
        for bi, (a, b) in enumerate(pair):
            if not quality[fi, bi]:
                continue
            i, j = index[a], index[b]
            z = weight[fi, bi] * np.exp(1j * observed[fi, bi])
            matrix[i, j] += z
            matrix[j, i] += z.conjugate()
        if not np.any(matrix):
            continue
        _, vectors = np.linalg.eigh(matrix)
        v = vectors[:, -1]
        phases[fi] = wrap(np.angle(v) - np.angle(v[index[1]]))
    return phases


def initial_delay(phase, freq):
    valid = np.isfinite(phase)
    if valid.sum() < 3:
        return 0., 0.
    offsets = 2 * np.pi * (freq[valid] - FREF) * 1e-9
    trials = np.arange(-1000., 1001., 2.)
    z = np.exp(1j * (phase[valid, None] - offsets[:, None] * trials[None]))
    avg = z.mean(axis=0)
    best = np.argmax(abs(avg))
    return float(np.angle(avg[best])), float(trials[best])


def fit_gain(pair, freq, phase, pn_phase, quality, significance, pn_sign):
    """Fit phase_i + 2pi(f-FREF)delay_i after subtracting PN baseline phase."""
    supported = quality.any(axis=0)
    antenna = np.unique(pair[supported])
    if 1 not in antenna:
        raise ValueError("Reference antenna 1 absent")
    index = {a: i for i, a in enumerate(antenna)}
    free = np.array([a for a in antenna if a != 1])
    var = {a: i for i, a in enumerate(free)}
    fidx, bidx = np.where(quality)
    if len(fidx) < 2 * len(free):
        raise ValueError("Insufficient training observations")
    ab = pair[bidx]
    foffset = 2 * np.pi * (freq[fidx, bidx] - FREF) * 1e-9
    target = wrap(phase[fidx, bidx] - pn_sign * pn_phase[fidx, bidx])
    weights = np.sqrt(np.clip(significance[fidx, bidx], 3., 20.) / 10.)
    design = np.zeros((len(fidx), 2 * len(free)))
    for row, (a, b) in enumerate(ab):
        if a != 1:
            design[row, var[a]] = 1.
            design[row, len(free) + var[a]] = foffset[row]
        if b != 1:
            design[row, var[b]] = -1.
            design[row, len(free) + var[b]] = -foffset[row]
    corrected = wrap(phase - pn_sign * pn_phase)
    sync = initial_phases(pair[supported], antenna, corrected[:, supported],
                          quality[:, supported], significance[:, supported])
    x0 = np.zeros(2 * len(free))
    for a in free:
        x0[var[a]], x0[len(free) + var[a]] = initial_delay(sync[:, index[a]], freq[:, 0])

    def residual(x):
        return weights * wrap(target - design @ x)

    fit = least_squares(residual, x0, loss="soft_l1", f_scale=.7,
                        max_nfev=200, jac="2-point")
    gain_phase = np.zeros(len(antenna))
    gain_delay = np.zeros(len(antenna))
    for a in free:
        gain_phase[index[a]] = wrap(fit.x[var[a]])
        gain_delay[index[a]] = fit.x[len(free) + var[a]]
    prediction = np.full_like(phase, np.nan)
    for bi, (a, b) in enumerate(pair):
        if a not in index or b not in index:
            continue
        delta_phase = gain_phase[index[a]] - gain_phase[index[b]]
        delta_delay = gain_delay[index[a]] - gain_delay[index[b]]
        prediction[:, bi] = pn_sign * pn_phase[:, bi] + delta_phase + \
            2 * np.pi * (freq[:, bi] - FREF) * delta_delay * 1e-9
    return antenna, gain_phase, gain_delay, prediction, fit


def score(errors, mask):
    x = errors[mask & np.isfinite(errors)]
    return {"n": int(len(x)),
            "median_abs_error_rad": float(np.median(abs(x))) if len(x) else None,
            "median_cosine": float(np.median(np.cos(x))) if len(x) else None,
            "mean_cosine": float(np.mean(np.cos(x))) if len(x) else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sun_coefficients", type=Path)
    ap.add_argument("pn_candidate", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--mapping", type=Path, default=Path("stream_antenna_map.json"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.sun_coefficients) as z:
        pair = z["pairs"].astype(int)
        freq_grid = z["frequency_hz"]
        coeff = z["coefficient"]
        significance = z["significance"]
        modulation = z["modulation"]
    with np.load(args.pn_candidate) as z:
        pn_table = z["phase_rad"]
        stable = z["stable"]
        pn_freq = z["fine_frequency_hz"]
    if coeff.shape[1:3] != (8, 7) or not np.allclose(freq_grid, pn_freq[:8, FINE]):
        raise ValueError("Sun and PN frequency grids differ")
    mapping = json.loads(args.mapping.read_text())["fpgas"]
    raw = {int(stream["antenna_id"]): int(board) * 20 + i
           for board, fpga in mapping.items()
           for i, stream in enumerate(fpga["streams"])
           if stream["antenna_id"] > 0 and stream["polarization"] == 0}
    pn_phase = np.array([pn_table[:8, FINE, raw[a]] - pn_table[:8, FINE, raw[b]]
                         for a, b in pair]).transpose(1, 2, 0)
    pn_stable = np.array([stable[:8, raw[a]] & stable[:8, raw[b]]
                          for a, b in pair]).T[:, None, :]
    pn_stable = np.broadcast_to(pn_stable, coeff.shape[1:])
    good = (significance >= 3) & (modulation >= .05) & pn_stable[None]
    train_freq = np.zeros((8, 7), bool)
    train_freq[:4, [1, 3, 5]] = True
    test_freq = np.zeros((8, 7), bool)
    test_freq[4:] = True
    # Deterministic baseline holdout; the remaining antenna graph stays dense.
    held_baseline = np.array([(int(a) * 37 + int(b) * 17) % 5 == 0
                              for a, b in pair])
    train_mask = good[0] & train_freq[:, :, None] & ~held_baseline[None, None]
    full_freq = np.broadcast_to(freq_grid[:, :, None], good.shape[1:]).reshape(-1, len(pair))
    report = {"reference_antenna": 1,
              "training": "First 10 minutes; coarse 400-403, fine offsets -203, 0, +203 kHz; 80% of baselines",
              "testing": "Second 10 minutes; coarse 404-407, all seven selected fine bins; training and held-out baselines separately",
              "gain_model": "Per-antenna phase at 312.5 MHz plus phase slope / residual delay; no amplitude fit",
              "quality": "Sun significance >=3, modulation >=0.05, PN stable; identical quality cells across models",
              "n_training_cells": int(train_mask.sum()),
              "n_training_baselines": int((train_mask.reshape(-1, len(pair)).any(axis=0)).sum()),
              "n_held_baselines": int(held_baseline.sum()),
              "scores": {}}
    errors = {}
    gains = {}
    for name, pn_sign in (("sky_only", 0), ("pn_then_sky", 1),
                          ("opposite_pn_control", -1)):
        ant, phase, delay, prediction, fit = fit_gain(
            pair, full_freq, np.angle(coeff[0]).reshape(-1, len(pair)),
            pn_phase.reshape(-1, len(pair)), train_mask.reshape(-1, len(pair)),
            significance[0].reshape(-1, len(pair)), pn_sign)
        prediction = prediction.reshape(8, 7, len(pair))
        error = wrap(np.angle(coeff[1]) - prediction)
        errors[name] = error
        gains[name] = (phase, delay)
        groups = {}
        same = np.array([raw[a] // 20 == raw[b] // 20 for a, b in pair])
        for label, baseline_mask in (("all", np.ones(len(pair), bool)),
                                     ("fit_baselines", ~held_baseline),
                                     ("held_baselines", held_baseline),
                                     ("same_fpga", same),
                                     ("cross_fpga", ~same)):
            groups[label] = score(error, good[1] & test_freq[:, :, None] & baseline_mask[None, None])
        report["scores"][name] = {"groups": groups,
                                  "fit_success": bool(fit.success),
                                  "fit_cost": float(fit.cost),
                                  "median_abs_residual_delay_ns": float(np.median(abs(delay)))}
    report["antenna_ids"] = ant.tolist()
    (args.output / "antenna_sun_residual_report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "antenna_sun_residual.npz",
                        antenna_ids=ant, pairs=pair, frequency_hz=freq_grid,
                        train_mask=train_mask, good_test=good[1], held_baseline=held_baseline,
                        **{f"phase_{name}": gains[name][0] for name in gains},
                        **{f"delay_ns_{name}": gains[name][1] for name in gains},
                        **{f"error_{name}": errors[name] for name in errors})
    fig, axs = plt.subplots(2, 1, figsize=(11, 8), layout="constrained")
    for name in errors:
        ys = []
        for ci in range(4, 8):
            mask = good[1, ci] & held_baseline[None]
            ys.append(score(errors[name][ci], mask)["median_cosine"])
        axs[0].plot(np.arange(404, 408), ys, ".-", label=name)
    axs[0].set(xlabel="Held coarse channel", ylabel="Median phase cosine on held baselines",
               ylim=(-1, 1))
    axs[0].grid(alpha=.3)
    axs[0].legend()
    axs[1].scatter(ant, gains["pn_then_sky"][1], s=18)
    axs[1].set(xlabel="Antenna", ylabel="Fitted residual delay (ns)")
    axs[1].grid(alpha=.3)
    fig.savefig(args.output / "antenna_sun_residual.png", dpi=170)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

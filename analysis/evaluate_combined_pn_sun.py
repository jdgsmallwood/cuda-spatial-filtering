#!/usr/bin/env python3
"""Score combined PN phase + local delay against independent Sun coefficients.

Requires the per-baseline moving-Sun coefficients from test_pn_fine_sun.py.
Fit a constant and residual delay on the first four coarse channels/selected
fine bins in the first half of the observation. Score held-out frequencies in
the second half. Compare identical stage-two fits with no PN, PN subtraction,
and wrong-sign PN. This never changes the live correlator or science HDF5.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fit_phase(phase, freq, quality):
    if quality.sum() < 8:
        return None
    trial_ns = np.arange(-1000., 1000.01, 1.)
    offset = freq - 312.5e6
    rotated = np.exp(1j * (phase[quality, None] -
                           2 * np.pi * offset[quality, None] * trial_ns[None, :] * 1e-9))
    coherence = abs(rotated.mean(axis=0))
    best = int(np.argmax(coherence))
    return float(np.angle(rotated[:, best].mean())), float(trial_ns[best])


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
        freq = z["frequency_hz"]
        coeff = z["coefficient"]
        significance = z["significance"]
        modulation = z["modulation"]
    with np.load(args.pn_candidate) as z:
        phase_table = z["phase_rad"]
        stable = z["stable"]
        candidate_freq = z["fine_frequency_hz"]
    local_bins = np.array([2, 6, 10, 13, 16, 20, 24])
    if coeff.shape[1:3] != (8, 7) or not np.allclose(freq, candidate_freq[:8, local_bins]):
        raise ValueError("Sun and PN selected fine-frequency grids differ")
    mapping = json.loads(args.mapping.read_text())["fpgas"]
    raw = {int(stream["antenna_id"]): int(board) * 20 + i
           for board, fpga in mapping.items()
           for i, stream in enumerate(fpga["streams"])
           if stream["antenna_id"] > 0 and stream["polarization"] == 0}
    pn_phase = np.array([phase_table[:8, local_bins, raw[a]] -
                         phase_table[:8, local_bins, raw[b]] for a, b in pair])
    pn_phase = np.moveaxis(pn_phase, 0, -1)
    pn_stable = np.array([stable[:8, raw[a]] & stable[:8, raw[b]]
                          for a, b in pair]).T[:, None, :]
    pn_stable = np.broadcast_to(pn_stable, coeff.shape[1:])
    train_freq = np.zeros((8, 7), bool)
    train_freq[:4, [1, 3, 5]] = True
    test_freq = np.zeros((8, 7), bool)
    test_freq[4:, :] = True
    quality_train = (significance[0] >= 3) & (modulation[0] >= .05) & pn_stable
    quality_test = (significance[1] >= 3) & (modulation[1] >= .05) & pn_stable
    quality_train &= train_freq[:, :, None]
    quality_test &= test_freq[:, :, None]
    names = ("sky_only", "pn_then_sky", "opposite_pn_control")
    result = {}
    error_map = {}
    for name, sign in zip(names, (0, -1, +1)):
        errors = np.full(coeff.shape[1:], np.nan)
        fit_delay = np.full(len(pair), np.nan)
        for bi in range(len(pair)):
            train = quality_train[:, :, bi]
            test = quality_test[:, :, bi]
            if test.sum() < 8:
                continue
            phase = np.angle(coeff[0, :, :, bi]) + sign * pn_phase[:, :, bi]
            fit = fit_phase(phase.ravel(), freq.ravel(), train.ravel())
            if fit is None:
                continue
            intercept, tau_ns = fit
            fit_delay[bi] = tau_ns
            prediction_residual = intercept + 2 * np.pi * (freq - 312.5e6) * tau_ns * 1e-9
            prediction_observed = prediction_residual - sign * pn_phase[:, :, bi]
            errors[:, :, bi][test] = np.angle(
                coeff[1, :, :, bi][test] * np.exp(-1j * prediction_observed[test]))
        accepted = np.isfinite(errors)
        cosine = np.cos(errors)
        result[name] = {
            "n_predictions": int(accepted.sum()),
            "n_baselines": int(np.any(accepted, axis=(0, 1)).sum()),
            "median_abs_phase_error_rad": float(np.median(abs(errors[accepted]))) if accepted.any() else None,
            "median_phase_cosine": float(np.median(cosine[accepted])) if accepted.any() else None,
            "mean_phase_cosine": float(np.mean(cosine[accepted])) if accepted.any() else None,
            "median_abs_fitted_residual_delay_ns": float(np.nanmedian(abs(fit_delay))) if np.isfinite(fit_delay).any() else None,
            "median_cosine_by_held_coarse_channel": [
                float(np.median(cosine[ci][accepted[ci]])) if accepted[ci].any() else None
                for ci in range(4, 8)],
        }
        error_map[name] = errors
    report = {
        "sun_coefficients": str(args.sun_coefficients),
        "pn_candidate": str(args.pn_candidate),
        "train": "First time half; coarse 400-403, fine offsets approximately -203, 0, +203 kHz; per-baseline constant plus residual delay",
        "held_out": "Second time half; all seven selected fine bins in coarse 404-407",
        "quality": "Independent Sun fit significance >=3, fringe variance >=0.05, PN phase repeat <=0.5 rad, local PN delay repeat <=100 ns; same cells for all candidates",
        "scores": result,
        "limitation": "Baseline-based stage-two phase is only a diagnostic; antenna gains, amplitudes, images, and the entire 40-channel band are not yet validated."
    }
    (args.output / "pn_combined_sun_report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "pn_combined_sun.npz", pairs=pair,
                        frequency_hz=freq, quality_train=quality_train,
                        quality_test=quality_test,
                        **{f"error_{name}": value for name, value in error_map.items()})
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for name in names:
        y = result[name]["median_cosine_by_held_coarse_channel"]
        ax.plot(np.arange(404, 408), [np.nan if x is None else x for x in y],
                ".-", label=name)
    ax.set(xlabel="Held-out coarse channel", ylabel="Median Sun phase cosine",
           title="Combined PN phase and fine delay, then sky residual delay",
           ylim=(-1, 1))
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(args.output / "pn_combined_sun.png", dpi=170)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

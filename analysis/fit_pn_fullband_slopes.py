#!/usr/bin/env python3
"""Fit coarse-centre and within-coarse slopes in a same-tuning ADC PN capture.

The coarse-centre slope is identifiable modulo 1/781250 Hz, provided the PN
phase remains coherent between coarse channels. The within-coarse slope uses
the four comb teeth. Their difference is only an *effective* post-channel
term: analog bandpass phase, FPGA filterbank phase, and PN-generator timing
are not separated by this observation.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COARSE_HZ = 781250.0
TRIAL_NS = np.arange(-640.0, 640.01, 1.0)
LOCAL_TRIAL_NS = np.arange(-1000.0, 1000.01, 1.0)


def wrap(x):
    return np.angle(np.exp(1j * x))


def fit_delay(channels, phases, selected):
    freqs = (channels - channels[0]) * COARSE_HZ
    grid = 2 * np.pi * freqs[selected, None] * TRIAL_NS[None, :] * 1e-9
    phasors = np.exp(1j * (phases[selected, None] - grid))
    best = int(np.argmax(abs(phasors.mean(axis=0))))
    delay = float(TRIAL_NS[best])
    intercept = float(np.angle(phasors[:, best].mean()))
    residual = wrap(phases - intercept - 2 * np.pi * freqs * delay * 1e-9)
    # One robust refit excludes a bad comb/coarse channel without removing
    # broad smooth structure from the held-out odd channels.
    inlier = selected & (abs(residual) < 0.5)
    if np.count_nonzero(inlier) >= 10 and np.count_nonzero(inlier) < np.count_nonzero(selected):
        grid = 2 * np.pi * freqs[inlier, None] * TRIAL_NS[None, :] * 1e-9
        phasors = np.exp(1j * (phases[inlier, None] - grid))
        best = int(np.argmax(abs(phasors.mean(axis=0))))
        delay = float(TRIAL_NS[best])
        intercept = float(np.angle(phasors[:, best].mean()))
        residual = wrap(phases - intercept - 2 * np.pi * freqs * delay * 1e-9)
    return delay, intercept, residual, inlier


def fit_four_tooth_delay(phases, offsets_hz):
    grid = 2 * np.pi * offsets_hz[:, None] * LOCAL_TRIAL_NS[None, :] * 1e-9
    phasors = np.exp(1j * (phases[:, None] - grid))
    best = int(np.argmax(abs(phasors.mean(axis=0))))
    delay = float(LOCAL_TRIAL_NS[best])
    intercept = float(np.angle(phasors[:, best].mean()))
    residual = wrap(phases - intercept - 2 * np.pi * offsets_hz * delay * 1e-9)
    return delay, float(np.sqrt(np.mean(residual**2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pn_npz", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.pn_npz) as z:
        channels = z["channels"].astype(int)
        phases = z["coarse_mean_phase_rad"]
        repeat = np.median(z["repeat_error_rad"], axis=1)
        local_phase = z["relative_phase_rad"]
        line_amplitude = z["line_amplitude"]
        offsets_hz = z["line_offsets_hz"]
    if len(channels) != 40 or not np.array_equal(channels, np.arange(400, 440)):
        raise ValueError("Expected a single same-tuning 400–439 PN analysis")
    centre_delay = np.full((2, 80), np.nan)
    centre_intercept = np.full((2, 80), np.nan)
    centre_residual = np.full((2, 40, 80), np.nan)
    centre_inlier = np.zeros((2, 40, 80), bool)
    local_delay = np.full((2, 40, 80), np.nan)
    local_residual = np.full((2, 40, 80), np.nan)
    for run in range(2):
        for channel in range(40):
            for stream in range(80):
                if stream == 2 or np.min(line_amplitude[run, channel, :, stream]) <= 0:
                    continue
                local_delay[run, channel, stream], local_residual[run, channel, stream] = (
                    fit_four_tooth_delay(local_phase[run, channel, :, stream], offsets_hz))
    # Include adjacent coarse channels in the fit. Selecting only even
    # channels halves the frequency spacing and creates a 640 ns alias that
    # flips every held odd channel by pi.
    held_channels = (channels - channels[0]) % 3 == 0
    for run in range(2):
        for stream in range(80):
            if stream == 2:
                continue
            training = (~held_channels) & (repeat[:, stream] < 0.5)
            if np.count_nonzero(training) < 12:
                continue
            d, a, residual, inlier = fit_delay(channels, phases[run, :, stream], training)
            centre_delay[run, stream] = d
            centre_intercept[run, stream] = a
            centre_residual[run, :, stream] = residual
            centre_inlier[run, :, stream] = inlier
    good = np.isfinite(centre_delay).all(axis=0)
    held = abs(centre_residual[:, held_channels, :])
    centre_repeat = abs(wrap(2 * np.pi * (channels[-1] - channels[0]) * COARSE_HZ *
                             (centre_delay[1] - centre_delay[0]) * 1e-9))
    # The centre estimate aliases every 1280 ns. Report only the principal
    # difference; do not assign it to a physical FPGA pipeline stage.
    good_local = (local_residual < 0.4) & (repeat[None, :, :] < 0.5)
    local_median = np.full((2, 80), np.nan)
    for run in range(2):
        for stream in range(80):
            selected = good_local[run, :, stream]
            if np.count_nonzero(selected) >= 12:
                local_median[run, stream] = float(np.median(local_delay[run, selected, stream]))
    effective_remainder = (local_median - centre_delay + 640) % 1280 - 640
    report = {
        "channels": [400, 439],
        "input": str(args.pn_npz),
        "fit_selection": "coarse channels with index modulo 3 in {1,2} and phase repeatability <0.5 rad; index modulo 3 == 0 held out",
        "usable_streams": int(good.sum()),
        "coarse_centre_delay_ns_median_by_board_run0": [
            float(np.nanmedian(centre_delay[0, b*20:(b+1)*20])) for b in range(4)],
        "within_coarse_delay_ns_median_by_board_run0": [
            float(np.nanmedian(local_median[0, b*20:(b+1)*20])) for b in range(4)],
        "effective_remainder_ns_median_by_board_run0": [
            float(np.nanmedian(effective_remainder[0, b*20:(b+1)*20])) for b in range(4)],
        "within_coarse_four_tooth_residual_rad_median": float(np.nanmedian(local_residual)),
        "within_coarse_delay_repeatability_ns_median": float(np.nanmedian(abs(local_median[1] - local_median[0]))),
        "within_coarse_delay_repeatability_ns_p90": float(np.nanquantile(abs(local_median[1] - local_median[0]), .9)),
        "centre_delay_repeatability_ns_median": float(np.nanmedian(abs(centre_delay[1] - centre_delay[0]))),
        "centre_delay_repeatability_ns_p90": float(np.nanquantile(abs(centre_delay[1] - centre_delay[0]), .9)),
        "held_centre_phase_error_rad_median_by_run": [float(np.nanmedian(x)) for x in held],
        "held_centre_phase_error_rad_p90_by_run": [float(np.nanquantile(x, .9)) for x in held],
        "held_centre_phase_error_rad_by_channel": np.nanmedian(held, axis=(0, 2)).tolist(),
        "caveat": "The difference between slopes is an effective spectral response, modulo 1280 ns. It does not uniquely isolate a physical post-channelization delay. PN injection bypasses the antenna/front-end path.",
    }
    np.savez_compressed(args.output / "pn_fullband_slopes.npz", channels=channels,
                        coarse_centre_delay_ns=centre_delay,
                        coarse_centre_intercept_rad=centre_intercept,
                        coarse_centre_residual_rad=centre_residual,
                        coarse_centre_fit_inlier=centre_inlier,
                        within_coarse_delay_ns=local_delay,
                        within_coarse_fit_residual_rad=local_residual,
                        within_coarse_median_delay_ns=local_median,
                        effective_remainder_ns=effective_remainder,
                        centre_repeat_phase_span_rad=centre_repeat)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    axes[0].plot(channels, np.nanmedian(abs(centre_residual[0]), axis=1), ".-", label="first PN toggle")
    axes[0].plot(channels, np.nanmedian(abs(centre_residual[1]), axis=1), ".-", label="second PN toggle")
    axes[0].set(ylabel="Median absolute residual (rad)",
                title="Single coarse-centre slope across 400–439; every third channel held out")
    axes[0].grid(alpha=.25)
    axes[0].legend()
    for board in range(4):
        sl = slice(board*20, (board+1)*20)
        axes[1].scatter(centre_delay[0, sl], local_median[0, sl],
                        s=22, label=f"FPGA {board}")
    axes[1].set(xlabel="Slope across coarse centres (ns, modulo 1280 ns)",
                ylabel="Slope within coarse channels (ns)",
                title="Distinct effective slopes in the ADC-box PN signal")
    axes[1].grid(alpha=.25)
    axes[1].legend()
    fig.savefig(args.output / "pn_fullband_slopes.png", dpi=170)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

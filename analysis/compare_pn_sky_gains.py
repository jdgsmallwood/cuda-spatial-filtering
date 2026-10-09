#!/usr/bin/env python3
"""Test whether independent ADC PN phases predict held-out sky gain phases.

The existing sky gains are solved independently per fine bin. Train one smooth
constant-plus-delay residual on even coarse centres; score odd centres. The PN
table comes from different raw-packet captures. This is an offline frequency
transfer check, not a visibility-level or image-level calibration result.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COARSE_HZ = 781250.0


def circular_fit(phase, keep, channel):
    # The training set contains adjacent channels. An even/odd split would
    # double frequency spacing and introduce a 640 ns alias that flips all
    # held-out odd-channel predictions by pi without affecting the fit.
    train = keep & (channel % 4 != 3)
    held = keep & (channel % 4 == 3)
    if train.sum() < 20 or held.sum() < 7:
        return np.nan, np.nan, np.full(len(channel), np.nan)
    trial_ns = np.arange(-640., 640.01, 1.)
    offset = (channel - channel[0]) * COARSE_HZ
    residual = np.exp(1j * (phase[train, None] -
                            2 * np.pi * offset[train, None] * trial_ns[None, :] * 1e-9))
    best = int(np.argmax(abs(residual.mean(axis=0))))
    delay = trial_ns[best]
    intercept = np.angle(residual[:, best].mean())
    model = intercept + 2 * np.pi * offset * delay * 1e-9
    error = abs(np.angle(np.exp(1j * (phase - model))))
    return delay, float(np.median(error[held])), error


def constant_fit(phase, keep, channel):
    train = keep & (channel % 4 != 3)
    held = keep & (channel % 4 == 3)
    if train.sum() < 20 or held.sum() < 7:
        return np.nan
    intercept = np.angle(np.mean(np.exp(1j * phase[train])))
    error = abs(np.angle(np.exp(1j * (phase[held] - intercept))))
    return float(np.median(error))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sky_candidates", type=Path)
    ap.add_argument("pn_table", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--mapping", type=Path, default=Path("stream_antenna_map.json"))
    ap.add_argument("--max-pn-repeat-rad", type=float, default=.5)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.sky_candidates) as z:
        ids = z["antenna_ids"].astype(int)
        gains = z["gains"][np.arange(40) * 27 + 13]
        freq = z["frequency_hz"][np.arange(40) * 27 + 13]
    with np.load(args.pn_table) as z:
        channel = z["channels"].astype(int)
        pn = z["phase_rad"]
        stable = z["repeat_error_rad"] <= args.max_pn_repeat_rad
    if not np.array_equal(channel, np.arange(400, 440)) or not np.allclose(freq, channel * COARSE_HZ):
        raise ValueError("Expected aligned coarse-centre frequency axes 400-439")
    mapping = json.loads(args.mapping.read_text())["fpgas"]
    raw = {int(s["antenna_id"]): board * 20 + i
           for key, streams in mapping.items() for board in [int(key)]
           for i, s in enumerate(streams["streams"])
           if s["antenna_id"] > 0 and s["polarization"] == 0}
    ref_id = 1
    ref_sky = int(np.flatnonzero(ids == ref_id)[0])
    ref_raw = raw[ref_id]
    records = []
    errors = {name: [] for name in ("sky_only", "pn_then_sky", "opposite_pn_control")}
    constant_errors = {name: [] for name in errors}
    examples = {}
    for si, antenna in enumerate(ids):
        ant = int(antenna)
        if ant in (ref_id, 8, 12, 14) or ant not in raw:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            sky = np.angle(gains[:, si] / gains[:, ref_sky])
        pn_relative = pn[:, raw[ant]] - pn[:, ref_raw]
        keep = (np.isfinite(sky) & (abs(gains[:, si]) > 0) &
                (abs(gains[:, ref_sky]) > 0) &
                stable[:, raw[ant]] & stable[:, ref_raw])
        row = {"antenna_id": ant, "usable_centres": int(keep.sum())}
        for name, sign in (("sky_only", 0), ("pn_then_sky", -1),
                           ("opposite_pn_control", +1)):
            phase = np.angle(np.exp(1j * (sky + sign * pn_relative)))
            delay, median, error = circular_fit(phase, keep, channel)
            row[name] = {"fitted_residual_delay_ns": float(delay),
                         "held_out_median_abs_phase_rad": float(median),
                         "constant_phase_only_median_abs_rad": constant_fit(phase, keep, channel)}
            if np.isfinite(median):
                errors[name].extend(error[keep & (channel % 4 == 3)].tolist())
                constant_errors[name].append(row[name]["constant_phase_only_median_abs_rad"])
            if ant in (16, 4, 21):
                examples[(ant, name)] = phase
        records.append(row)
    summary = {name: {"held_out_median_abs_phase_rad": float(np.median(value)),
                      "held_out_p90_abs_phase_rad": float(np.quantile(value, .9)),
                      "constant_phase_only_median_of_antenna_medians_rad": float(np.median(constant_errors[name])),
                      "median_abs_fitted_residual_delay_ns": float(np.nanmedian(abs(np.array([row[name]["fitted_residual_delay_ns"] for row in records])))),
                      "n_channel_antenna_predictions": len(value)}
               for name, value in errors.items()}
    report = {
        "sky_candidates": str(args.sky_candidates), "pn_table": str(args.pn_table),
        "reference_antenna": ref_id,
        "max_pn_repeat_rad": args.max_pn_repeat_rad,
        "train": "All centres except 403,407,...,439; fit one per-antenna constant phase and one residual delay in [-640,640] ns",
        "held_out": "Every fourth coarse centre 403,407,...,439; same quality mask for all three candidates",
        "excluded_antenna_ids": [1, 8, 12, 14],
        "summary": summary, "antenna_results": records,
        "limitation": "The sky gains were inferred from a nighttime point-source model with known RFI/model uncertainty. This compares frequency transfer of independently fitted sky gain phases; it does not prove improved held-out visibilities or an on-sky antenna calibration."
    }
    (args.output / "pn_sky_gain_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), layout="constrained")
    for axis, ant in zip(axes, (16, 4, 21)):
        for name in ("sky_only", "pn_then_sky", "opposite_pn_control"):
            if (ant, name) in examples:
                phase = examples[(ant, name)]
                finite = np.isfinite(phase)
                axis.plot(channel[finite], np.unwrap(phase[finite]),
                          ".-", ms=3, label=name)
        axis.set(title=f"Antenna {ant} relative to antenna 1; phase at coarse centres",
                 xlabel="Coarse channel", ylabel="Unwrapped phase (rad)")
        axis.grid(alpha=.25)
        axis.legend()
    fig.savefig(args.output / "pn_sky_gain_phases.png", dpi=170)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

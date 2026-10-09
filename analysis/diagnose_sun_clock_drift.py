#!/usr/bin/env python3
"""Test for time-varying FPGA-common delay using the Sun fringe spectrum.

The early/late phase ratio cancels the static per-frequency calibration.
Board-pair slopes should be coherent and close around board triangles if they
represent a differential clock delay. This is a diagnostic, not a solve.
"""

import argparse
import json
from pathlib import Path

from casacore.tables import table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def window_phasor(data, model, begin, end):
    numerator = None
    count = None
    for ti in range(begin, end):
        vis = data.getcol("DATA", ti * 630, 630)
        pred = model.getcol("MODEL_DATA", ti * 630, 630)
        flag = data.getcol("FLAG", ti * 630, 630)
        ratio = vis * pred.conj()
        amp = np.abs(ratio)
        valid = (~flag) & (amp > 0) & (abs(pred) > .5)
        unit = np.divide(ratio, amp, out=np.zeros_like(ratio), where=valid)
        if numerator is None:
            numerator = unit.astype(np.complex64)
            count = valid.astype(np.int16)
        else:
            numerator += unit
            count += valid
    mean = np.divide(numerator, count, out=np.zeros_like(numerator),
                     where=count > 0)
    return mean, count


def fit_slope(freq, phasor, counts, accepted):
    valid = accepted & (counts >= 5) & (abs(phasor) > .55)
    if valid.sum() < 80:
        return None
    phase = np.unwrap(np.angle(phasor[valid]))
    x = (freq[valid] - np.mean(freq[valid])) / 1e6
    keep = np.ones(len(x), bool)
    for _ in range(3):
        slope, intercept = np.polyfit(x[keep], phase[keep], 1)
        residual = phase - (slope * x + intercept)
        scale = 1.4826 * np.median(abs(residual[keep] - np.median(residual[keep])))
        keep = abs(residual) < max(3 * scale, np.deg2rad(1.0))
    slope, intercept = np.polyfit(x[keep], phase[keep], 1)
    residual = phase[keep] - (slope * x[keep] + intercept)
    local_x = []
    local_y = []
    coarse = np.arange(len(freq)) // 27
    for coarse_id in np.unique(coarse[valid]):
        indices = np.flatnonzero(valid & (coarse == coarse_id))
        if len(indices) < 10:
            continue
        local_phase = np.unwrap(np.angle(phasor[indices]))
        local_x.extend((freq[indices] - np.mean(freq[indices])) / 1e6)
        local_y.extend(local_phase - np.mean(local_phase))
    local_x = np.asarray(local_x)
    local_y = np.asarray(local_y)
    if len(local_x) < 80:
        return None
    local_keep = np.ones(len(local_x), bool)
    for _ in range(3):
        local_slope = np.dot(local_x[local_keep], local_y[local_keep]) / \
                      np.dot(local_x[local_keep], local_x[local_keep])
        local_residual = local_y - local_slope * local_x
        local_scale = 1.4826 * np.median(abs(local_residual[local_keep]))
        local_keep = abs(local_residual) < max(3 * local_scale, np.deg2rad(.5))
    local_slope = np.dot(local_x[local_keep], local_y[local_keep]) / \
                  np.dot(local_x[local_keep], local_x[local_keep])
    local_residual = local_y[local_keep] - local_slope * local_x[local_keep]
    return {"global_linear_delay_ns": float(slope / (2 * np.pi) * 1000),
            "local_delay_ns": float(local_slope / (2 * np.pi) * 1000),
            "local_rms_phase_residual_deg": float(np.rad2deg(
                np.sqrt(np.mean(local_residual**2)))),
            "phase_deg_at_mean_frequency": float(np.rad2deg(intercept)),
            "rms_phase_residual_deg": float(np.rad2deg(np.sqrt(np.mean(residual**2)))),
            "frequency_bins": int(keep.sum())}


def main(source: Path, model_path: Path, config_path: Path, output: Path):
    image_path = output.with_suffix(".png")
    if output.exists() or image_path.exists():
        raise FileExistsError(output)
    config = json.loads(config_path.read_text())
    board_map = {int(a): int(b) for b, antennas in
                 config["antenna_positions"].items() for a in antennas}
    with table(str(source / "SPECTRAL_WINDOW"), ack=False) as spectral:
        freq = spectral.getcell("CHAN_FREQ", 0)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    report = {"source_ms": str(source), "model_ms": str(model_path),
              "windows": {"early": [0, 10], "middle": [55, 65],
                          "late": [100, 110]},
              "interpretation": "Change in Sun-model phase by FPGA pair; "
              "a clock-delay hypothesis requires coherent frequency slopes "
              "and pair closure, not just a changing broadband phase.",
              "results": {}}
    with table(str(source), ack=False) as data, table(str(model_path), ack=False) as model:
        if data.nrows() != model.nrows() or data.nrows() % 630:
            raise ValueError("measurement-set rows differ")
        a = data.getcol("ANTENNA1", 0, 630)
        b = data.getcol("ANTENNA2", 0, 630)
        ba = np.array([board_map[int(x) + 1] for x in a])
        bb = np.array([board_map[int(x) + 1] for x in b])
        early, count_early = window_phasor(data, model, 0, 10)
        for ci, (label, window) in enumerate((("middle", (55, 65)),
                                               ("late", (100, 110)))):
            recent, count_recent = window_phasor(data, model, *window)
            change = recent * early.conj()
            valid = (count_early >= 5) & (count_recent >= 5) & \
                    (abs(early) > .55) & (abs(recent) > .55)
            change = np.divide(change, abs(change), out=np.zeros_like(change),
                               where=valid & (abs(change) > 0))
            report["results"][label] = {}
            for pi, pol in enumerate(("XX", "YY")):
                axis = axes[pi, ci]
                accepted = ((freq >= 312e6) &
                            (freq < (334e6 if pol == "XX" else 324e6)) &
                            ~((freq >= 318.4e6) & (freq < 319.7e6)))
                report["results"][label][pol] = {}
                for pair_no, (i, j) in enumerate(((0, 1), (0, 2), (0, 3),
                                                   (1, 2), (1, 3), (2, 3))):
                    forward = (ba == i) & (bb == j)
                    reverse = (ba == j) & (bb == i)
                    chosen = forward | reverse
                    if not chosen.any():
                        continue
                    z = change[chosen, :, pi].copy()
                    z[reverse[chosen]] = z[reverse[chosen]].conj()
                    sample_count = np.count_nonzero(valid[chosen, :, pi], axis=0)
                    mean = np.divide(z.sum(axis=0), sample_count,
                                     out=np.zeros(len(freq), complex),
                                     where=sample_count > 0)
                    fit = fit_slope(freq, mean, sample_count, accepted)
                    if fit is None:
                        continue
                    name = f"{i}–{j}"
                    report["results"][label][pol][name] = fit
                    shown = accepted & (sample_count >= 5) & (abs(mean) > .55)
                    positions = np.flatnonzero(shown)
                    breaks = np.flatnonzero(np.diff(freq[positions]) > 1.5 * np.median(np.diff(freq)))
                    for segment_no, segment in enumerate(np.split(positions, breaks + 1)):
                        if not len(segment):
                            continue
                        axis.plot(freq[segment] / 1e6,
                                  np.rad2deg(np.angle(mean[segment])), lw=1,
                                  color=f"C{pair_no}",
                                  label=(f"{name}: local {fit['local_delay_ns']:+.1f} ns"
                                         if segment_no == 0 else None))
                axis.set_title(f"{pol}: early → {label} Sun phase, by FPGA pair")
                axis.set(xlabel="Frequency (MHz)", ylabel="Phase change (degrees)")
                axis.grid(alpha=.2)
                axis.legend(loc="best", fontsize=8, ncol=2)
    fig.suptitle("Inter-FPGA delay-drift diagnostic; static calibration cancels",
                 fontsize=16)
    fig.savefig(image_path, dpi=150)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("sun_model", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.source, args.sun_model, args.config, args.output)

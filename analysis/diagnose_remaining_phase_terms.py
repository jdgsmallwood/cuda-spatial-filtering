#!/usr/bin/env python3
"""Summarize phase drift and spectral structure in a full-band Sun gain fit."""

import argparse
import json
from pathlib import Path

import numpy as np


def median_abs_deg(x):
    return float(np.rad2deg(np.median(abs(np.asarray(x)))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("validation", type=Path)
    ap.add_argument("calibration_report", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--mapping", type=Path, default=Path("stream_antenna_map.json"))
    ap.add_argument("--point-control", type=Path)
    args = ap.parse_args()
    with np.load(args.validation) as z:
        frequency = z["frequencies_hz"]
        gain = z["gains"]
        good = z["good"]
        antenna = z["antenna_order"]
        change = z["phase_change"]
    report = json.loads(args.calibration_report.read_text())
    mapping = json.loads(args.mapping.read_text())["fpgas"]
    board = {int(s["antenna_id"]): int(k) for k, v in mapping.items()
             for s in v["streams"] if s["antenna_id"] > 0 and s["polarization"] == 0}
    result = {"fit": str(args.validation),
              "window_start_utc": report["utc"],
              "meaning": "First-window gains; phase drift compares independently fitted late-window gains after removing one global phase",
              "polarizations": {}}
    reference = int(np.flatnonzero(antenna == 1)[0])
    for pi, pol in enumerate(("XX", "YY")):
        excluded = set(report["excluded_antennas"]) | {
            int(v["antenna_id"]) for v in report["excluded_receptors"]
            if v["polarization"] == pol}
        accepted = good[:, pi]
        board_mean = []
        board_result = {}
        for bi in range(4):
            indices = [i for i, a in enumerate(antenna)
                       if board[int(a)] == bi and int(a) not in excluded]
            values = change[accepted, pi][:, indices]
            centre = np.angle(np.mean(np.exp(1j * values), axis=1))
            residual = np.angle(np.exp(1j * (values - centre[:, None])))
            board_mean.append(centre)
            board_result[str(bi)] = {
                "n_receptors": len(indices),
                "median_abs_board_drift_deg": median_abs_deg(centre),
                "median_abs_within_board_drift_deg": median_abs_deg(residual),
                "p90_abs_within_board_drift_deg": float(
                    np.rad2deg(np.percentile(abs(residual), 90)))}
        for bi in range(1, 4):
            diff = np.angle(np.exp(1j * (board_mean[bi] - board_mean[0])))
            board_result[str(bi)]["median_abs_drift_relative_fpga0_deg"] = median_abs_deg(diff)

        linear_residual = []
        constant_residual = []
        local_delay = []
        coarse_jump = []
        for ai, a in enumerate(antenna):
            if a == 1 or int(a) in excluded:
                continue
            per_coarse = {}
            for ci in range(len(frequency) // 27):
                ids = np.arange(ci * 27, (ci + 1) * 27)
                ids = ids[accepted[ids]]
                if len(ids) < 12:
                    continue
                phase = np.unwrap(np.angle(gain[ids, pi, ai] /
                                            gain[ids, pi, reference]))
                x = (frequency[ids] - frequency[ids].mean()) * 1e-6
                slope, intercept = np.polyfit(x, phase, 1)
                linear_residual.extend(abs(phase - (slope * x + intercept)))
                constant_residual.extend(abs(phase - np.median(phase)))
                local_delay.append(slope / (2 * np.pi) * 1000.)
                per_coarse[ci] = (ids, slope, intercept)
            for ci in range(len(frequency) // 27 - 1):
                if ci not in per_coarse or ci + 1 not in per_coarse:
                    continue
                ids0, slope0, intercept0 = per_coarse[ci]
                ids1, slope1, intercept1 = per_coarse[ci + 1]
                centre0 = frequency[ci * 27 + 13]
                centre1 = frequency[(ci + 1) * 27 + 13]
                phase0 = slope0 * (centre0 - frequency[ids0].mean()) * 1e-6 + intercept0
                phase1 = slope1 * (centre1 - frequency[ids1].mean()) * 1e-6 + intercept1
                predicted = phase0 + slope0 * (centre1 - centre0) * 1e-6
                coarse_jump.append(abs(np.angle(np.exp(1j * (phase1 - predicted)))))

        result["polarizations"][pol] = {
            "accepted_fine_channels": int(accepted.sum()),
            "median_abs_linear_fit_residual_rad": float(np.median(linear_residual)),
            "p90_abs_linear_fit_residual_rad": float(np.percentile(linear_residual, 90)),
            "median_abs_constant_fit_residual_rad": float(np.median(constant_residual)),
            "median_abs_local_delay_ns": float(np.median(abs(np.asarray(local_delay)))),
            "median_abs_coarse_boundary_jump_rad": float(np.median(coarse_jump)),
            "board_drift": board_result,
        }
    if args.point_control:
        point = json.loads(args.point_control.read_text())
        result["sun_point_control"] = {
            pol: {"point_late_validation": point["median_validation"][pol][3],
                  "disk_late_validation": report["median_validation"][pol][3],
                  "point_accepted_channels": point["accepted_channels"][pol],
                  "disk_accepted_channels": report["accepted_channels"][pol]}
            for pol in ("XX", "YY")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

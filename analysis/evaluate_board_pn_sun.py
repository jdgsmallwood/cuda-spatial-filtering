#!/usr/bin/env python3
"""Test a board-common PN phase prior on held-out moving-Sun coefficients.

Uses precisely the train/test cells already selected by
evaluate_combined_pn_sun.py, so within-FPGA and cross-FPGA comparisons are
paired with its sky-only and per-stream PN results. This is diagnostic and
does not modify the capture, calibration table, or live pipeline.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_combined_pn_sun import fit_phase


def summarize(error, selected):
    valid = np.isfinite(error) & selected[None, None, :]
    values = abs(error[valid])
    return {
        "n_predictions": int(valid.sum()),
        "n_baselines": int(np.any(valid, axis=(0, 1)).sum()),
        "median_abs_phase_error_rad": float(np.median(values)) if len(values) else None,
        "median_phase_cosine": float(np.median(np.cos(values))) if len(values) else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sun_coefficients", type=Path)
    ap.add_argument("pn_candidate", type=Path)
    ap.add_argument("combined_result", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--mapping", type=Path, default=Path("stream_antenna_map.json"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with np.load(args.sun_coefficients) as z:
        pair = z["pairs"].astype(int)
        freq = z["frequency_hz"]
        coefficient = z["coefficient"]
    with np.load(args.pn_candidate) as z:
        phase = z["phase_rad"]
        stable = z["stable"]
    with np.load(args.combined_result) as z:
        train = z["quality_train"]
        test = z["quality_test"]
        other = {name: z[f"error_{name}"]
                 for name in ("sky_only", "pn_then_sky", "opposite_pn_control")}

    mapping = json.loads(args.mapping.read_text())["fpgas"]
    board_of = {}
    board_phase = np.empty((4, 8, 7))
    board_coherence = np.empty(4)
    bins = np.array([2, 6, 10, 13, 16, 20, 24])
    for board, fpga in mapping.items():
        bi = int(board)
        raw = []
        for i, stream in enumerate(fpga["streams"]):
            if stream["antenna_id"] > 0 and stream["polarization"] == 0:
                board_of[int(stream["antenna_id"])] = bi
                if stream["antenna_id"] not in (8, 12, 14):
                    raw.append(bi * 20 + i)
        selected_phase = phase[:8][:, bins][:, :, raw]
        selected_stable = stable[:8][:, raw][:, None, :]
        phasor = np.where(selected_stable, np.exp(1j * selected_phase), np.nan + 0j)
        board_mean = np.nanmean(phasor, axis=-1)
        board_phase[bi] = np.angle(board_mean)
        board_coherence[bi] = np.nanmedian(abs(board_mean))

    baseline_board_phase = np.moveaxis(
        np.array([board_phase[board_of[a]] - board_phase[board_of[b]]
                  for a, b in pair]), 0, -1)
    board_error = np.full(coefficient.shape[1:], np.nan)
    for bi in range(len(pair)):
        tr = train[:, :, bi]
        te = test[:, :, bi]
        if te.sum() < 8:
            continue
        first_phase = np.angle(coefficient[0, :, :, bi]) - baseline_board_phase[:, :, bi]
        fit = fit_phase(first_phase.ravel(), freq.ravel(), tr.ravel())
        if fit is None:
            continue
        intercept, delay_ns = fit
        prediction = (intercept + 2 * np.pi * (freq - 312.5e6) * delay_ns * 1e-9
                      + baseline_board_phase[:, :, bi])
        board_error[:, :, bi][te] = np.angle(
            coefficient[1, :, :, bi][te] * np.exp(-1j * prediction[te]))

    same = np.array([board_of[a] == board_of[b] for a, b in pair])
    scores = {}
    for category, selected in (("same_fpga", same), ("cross_fpga", ~same)):
        scores[category] = {name: summarize(error, selected) for name, error in
                            (("sky_only", other["sky_only"]),
                             ("per_stream_pn", other["pn_then_sky"]),
                             ("board_common_pn", board_error),
                             ("opposite_pn", other["opposite_pn_control"]))}
    report = {
        "sun_coefficients": str(args.sun_coefficients),
        "pn_candidate": str(args.pn_candidate),
        "quality": "Identical train and held-out cells to pn_combined_sun_report.json",
        "board_phase": "Circular mean of stable connected YY PN streams, excluding mapped antennas 8, 12, and 14",
        "board_phase_resultant_median_by_fpga": board_coherence.tolist(),
        "scores": scores,
        "limitation": "Board-common stage one is a diagnostic; it is not an antenna-based sky gain solve or validated science correction.",
    }
    (args.output / "pn_board_common_sun_report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "pn_board_common_sun.npz", pairs=pair,
                        board_phase_rad=board_phase, error_board_common_pn=board_error)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

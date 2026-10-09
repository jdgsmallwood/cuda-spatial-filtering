#!/usr/bin/env python3
"""Compare nominal and X/Y-swapped receptor coherence on the Sun capture."""

import argparse
import json
from pathlib import Path

from casacore.tables import table
import h5py
import numpy as np


def covariance_window(capture: h5py.File, start: int, stop: int,
                      channels: int) -> np.ndarray:
    # One HDF5 time chunk at a time keeps peak memory bounded and uses a
    # coherent time average before measuring visibility amplitudes.
    block = capture["visibilities"][start:stop, :channels]
    complex_block = block[..., 0] + 1j * block[..., 1]
    values = complex_block.mean(axis=0)
    pairs = capture["baseline_antenna_ids"][:]
    a, b = pairs.T - 1
    keep = (a >= 0) & (b >= 0) & (a < 36) & (b < 36)
    a, b = a[keep], b[keep]
    values = values[:, keep]
    cov = np.zeros((channels, 72, 72), np.complex64)
    for p in range(2):
        for q in range(2):
            i, j = 2*a+p, 2*b+q
            cov[:, i, j] = values[:, :, p, q]
            cov[:, j, i] = values[:, :, p, q].conj()
    return cov


def evaluate(capture_path: Path, ms_path: Path, validation: Path,
             output: Path, samples: int) -> None:
    if output.exists():
        raise FileExistsError(output)
    with np.load(validation) as fit:
        joint = fit["joint"]
    nfreq = min(len(joint), 435)
    accepted = np.flatnonzero(joint[:nfreq])
    with table(str(ms_path), ack=False) as ms:
        a = ms.getcol("ANTENNA1", 0, 630)
        b = ms.getcol("ANTENNA2", 0, 630)
        lengths = np.linalg.norm(ms.getcol("UVW", 0, 630), axis=1)
    selected = np.zeros((36, 36), bool)
    selected[a, b] = lengths > 7
    selected[b, a] = lengths > 7
    # Known weak/unstable receptors are excluded only as references; targets
    # themselves remain testable. 8 and 12 are not removed from either side.
    references = np.ones(36, bool)
    references[np.array([5, 14, 29]) - 1] = False
    results = {}
    with h5py.File(capture_path) as capture:
        nt = capture["visibilities"].shape[0]
        for window, start in (("early", 3), ("late", nt-samples-3)):
            cov = covariance_window(capture, start, start+samples, nfreq)
            auto = np.maximum(cov.diagonal(axis1=-2, axis2=-1).real, 1e-30)
            window_rows = []
            for ant in range(36):
                other = np.flatnonzero(selected[ant] & references)
                other = other[other != ant]
                if not len(other):
                    continue
                hand_rows = {}
                for pol, hand in (("XX", 1), ("YY", 0)):
                    target_nom = 2*ant+hand
                    target_swap = 2*ant+(1-hand)
                    refs = 2*other+hand
                    normal = (np.abs(cov[accepted, target_nom][:, refs])
                              / np.sqrt(auto[accepted, target_nom, None]
                                        * auto[accepted][:, refs]))
                    swapped = (np.abs(cov[accepted, target_swap][:, refs])
                               / np.sqrt(auto[accepted, target_swap, None]
                                         * auto[accepted][:, refs]))
                    usable = np.isfinite(normal) & np.isfinite(swapped)
                    n = normal[usable]
                    s = swapped[usable]
                    hand_rows[pol] = {
                        "nominal_median_coherence": float(np.median(n)),
                        "swapped_median_coherence": float(np.median(s)),
                        "swapped_over_nominal": float(np.median(s) / max(np.median(n), 1e-30)),
                        "fraction_swapped_stronger": float(np.mean(s > n)),
                        "samples": int(len(n)),
                    }
                window_rows.append({"antenna_id": ant+1, "reference_baselines": len(other),
                                    **hand_rows})
            results[window] = window_rows
            print(f"read {window}: {start}:{start+samples}", flush=True)
    report = {"capture": str(capture_path), "accepted_shared_bins": int(len(accepted)),
              "baseline_min_m": 7, "windows": results}
    output.write_text(json.dumps(report, indent=2) + "\n")
    for window in ("early", "late"):
        print(window)
        for row in sorted(results[window], key=lambda row: -min(
                row["XX"]["swapped_over_nominal"],
                row["YY"]["swapped_over_nominal"]))[:10]:
            print(row["antenna_id"],
                  round(row["XX"]["swapped_over_nominal"], 2),
                  round(row["YY"]["swapped_over_nominal"], 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("ms", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=11)
    args = parser.parse_args()
    evaluate(args.capture, args.ms, args.validation, args.output, args.samples)

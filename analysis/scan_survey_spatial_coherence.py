#!/usr/bin/env python3
"""Rank surveyed coarse channels by same-FPGA normalized sky/RFI coherence.

The power-only RFI survey can miss a broadband coherent signal. This reads a
few complete integrations per 40-channel HDF5 file and reports each board and
polarization separately. It is a frequency-selection diagnostic, not a sky
calibration or an RFI classification.
"""
import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def summarize(path, antenna_board, fractions):
    with h5py.File(path) as h:
        ids = h["antenna_ids"][:].astype(int)
        pairs = h["baseline_antenna_ids"][:].astype(int)
        low = int(h.attrs["min_channel"])
        high = int(h.attrs["max_channel"])
        count = high - low + 1
        fine = h["visibilities"].shape[1] // count
        first_raw = int(h.attrs.get("first_raw_fine_bin", 2 if fine == 28 else 3))
        centre = 16 - first_raw
        if not 0 <= centre < fine:
            raise ValueError(f"cannot locate raw fine bin 16 in {path}")
        indices = np.arange(count) * fine + centre
        autos = {int(a): int(np.flatnonzero((pairs[:, 0] == a) &
                                           (pairs[:, 1] == a))[0])
                 for a in ids if a > 0}
        mapping = {}
        for board in range(4):
            ant = [int(a) for a in ids if a > 0 and antenna_board.get(int(a)) == board
                   and a != 14]
            mapping[board] = [int(i) for i, (a, b) in enumerate(pairs)
                              if a in ant and b in ant and a != b]
        missing = h["vis_missing_nums"][:]
        valid = np.flatnonzero(missing[:, 0] == 0)
        valid = valid[(valid >= 3) & (valid < len(missing) - 1)]
        if len(valid) < len(fractions):
            raise ValueError(f"too few complete integrations in {path}")
        times = valid[np.minimum(np.floor(np.asarray(fractions) *
                                          (len(valid) - 1)).astype(int), len(valid) - 1)]
        values = np.empty((len(times), count, 4, 2), float)
        for ti, t in enumerate(times):
            block = h["visibilities"][int(t), indices]
            for board in range(4):
                pair_indices = mapping[board]
                a, b = pairs[pair_indices].T
                for hand in range(2):
                    real = block[:, :, hand, hand, 0]
                    imag = block[:, :, hand, hand, 1]
                    cross = real[:, pair_indices] + 1j * imag[:, pair_indices]
                    norm = np.sqrt(np.maximum(
                        real[:, [autos[int(v)] for v in a]] *
                        real[:, [autos[int(v)] for v in b]], 1e-30))
                    values[ti, :, board, hand] = np.median(np.abs(cross) / norm, axis=1)
        return dict(path=str(path), min_channel=low, max_channel=high,
                    fine_channels_per_coarse=fine, first_raw_fine_bin=first_raw,
                    sampled_integrations=times.tolist(),
                    sampled_coherence=values.tolist(),
                    median_coherence=np.median(values, axis=0).tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("survey_manifest", type=Path)
    p.add_argument("stream_map", type=Path)
    p.add_argument("output_dir", type=Path)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(a.survey_manifest.read_text())
    smap = json.loads(a.stream_map.read_text())["fpgas"]
    antenna_board = {int(stream["antenna_id"]): int(board)
                     for board, detail in smap.items()
                     for stream in detail["streams"] if stream["antenna_id"] > 0}
    runs = []
    for key, window in sorted(manifest["windows"].items()):
        if not window["accepted"]:
            continue
        attempt = next(x for x in window["attempts"] if x["accepted"])
        print("Scanning", key, attempt["output"], flush=True)
        runs.append(summarize(Path(attempt["output"]), antenna_board,
                              [0.15, 0.40, 0.65, 0.85]))
    rows = []
    for run in runs:
        for ci, channel in enumerate(range(run["min_channel"], run["max_channel"] + 1)):
            row = dict(coarse_channel=channel, frequency_mhz=channel * .78125,
                       survey_file=run["path"])
            for board in range(4):
                for hand, name in enumerate(("YY", "XX")):
                    row[f"fpga{board}_{name}"] = run["median_coherence"][ci][board][hand]
            row["median_YY"] = float(np.median([row[f"fpga{b}_YY"] for b in range(4)]))
            row["median_XX"] = float(np.median([row[f"fpga{b}_XX"] for b in range(4)]))
            rows.append(row)
    with (a.output_dir / "spatial_coherence.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (a.output_dir / "spatial_coherence.json").write_text(
        json.dumps(dict(method=__doc__,runs=runs), indent=2) + "\n")
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, layout="constrained")
    for axis, hand in zip(axes, ("XX", "YY")):
        for run in runs:
            channel = np.arange(run["min_channel"], run["max_channel"] + 1)
            by_board = np.asarray(run["median_coherence"])[:, :, 1 if hand == "XX" else 0]
            axis.plot(channel * .78125, np.median(by_board, axis=1), color="tab:blue", lw=.8)
            axis.scatter(np.repeat(channel[:, None], 4, axis=1).ravel() * .78125,
                         by_board.ravel(), s=2, color="0.45", alpha=.5)
        axis.set(ylabel=f"{hand} normalized cross coherence", yscale="log")
        axis.grid(alpha=.25)
    axes[-1].set_xlabel("Coarse centre (MHz)")
    fig.savefig(a.output_dir / "spatial_coherence.png", dpi=170)
    print("Wrote", a.output_dir, flush=True)


if __name__ == "__main__":
    main()

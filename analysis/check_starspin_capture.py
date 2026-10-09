#!/usr/bin/env python3
"""Check packet loss, time continuity and finite visibilities in one capture."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    with h5py.File(a.capture) as h:
        seq = h["vis_seq_nums"][:].astype(np.int64)
        missing = h["vis_missing_nums"][:].astype(np.int64)
        ids = h["antenna_ids"][:].astype(int)
        pairs = h["baseline_antenna_ids"][:].astype(int)
        if missing.shape[0] != len(seq) or h["visibilities"].shape[0] != len(seq):
            raise ValueError("visibility, sequence and missing-record counts differ")
        sample = np.unique(np.r_[0, np.linspace(0, len(seq) - 1, 12, dtype=int)])
        finite = []
        nonzero_live_autos = []
        live = np.flatnonzero((pairs[:, 0] == pairs[:, 1]) & (pairs[:, 0] > 0))
        for i in sample:
            vis = h["visibilities"][int(i)]
            finite.append(float(np.mean(np.isfinite(vis))))
            nonzero_live_autos.append(bool(np.all(np.any(vis[:, live, :, :, 0] != 0, axis=(0, 2, 3)))))
        sequence_contiguous = bool(np.all(seq[1:, 0] == seq[:-1, 1] + 64))
        expected = int(missing[:, 1].sum())
        lost = int(missing[:, 0].sum())
        report = dict(capture=str(a.capture), shape=list(h["visibilities"].shape),
                      antenna_ids=ids.tolist(), min_channel=int(h.attrs["min_channel"]),
                      max_channel=int(h.attrs["max_channel"]),
                      integrations=len(seq), duration_s=float((seq[-1, 1] - seq[0, 0] + 64) * 1.08e-6),
                      missing_packets=lost, expected_packet_slots=expected,
                      missing_fraction=lost / expected,
                      missing_percent=100 * lost / expected,
                      lossy_integrations=int(np.count_nonzero(missing[:, 0])),
                      sequence_contiguous=sequence_contiguous,
                      sampled_integrations=sample.tolist(),
                      sampled_finite_fraction=finite,
                      sampled_live_autos_nonzero=nonzero_live_autos)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("integrations", "duration_s", "missing_packets",
                       "expected_packet_slots", "missing_percent",
                       "lossy_integrations", "sequence_contiguous")}, indent=2))


if __name__ == "__main__":
    main()

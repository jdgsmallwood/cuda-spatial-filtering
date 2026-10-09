#!/usr/bin/env python3
"""Validate Sun amplitude gains on time bins that were not used to fit them."""

import argparse
import json
from pathlib import Path

from casacore.tables import table
import numpy as np

if __package__:
    from .fit_sun_amplitude_selfcal import score
else:
    from fit_sun_amplitude_selfcal import score


def main(source: Path, model_path: Path, fit_path: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    with np.load(fit_path) as fit:
        gain = fit["gain"]
        a = fit["antenna_a"]
        b = fit["antenna_b"]
        cuts = fit["cuts_m"]
    results = {pol: {"before": [], "previous_only": [],
                     "interpolated": [], "same_time_fit": []}
               for pol in ("XX", "YY")}
    with table(str(source), ack=False) as data, table(str(model_path), ack=False) as model:
        if data.nrows() != model.nrows() or data.nrows() != len(gain) * 630:
            raise ValueError("MS and gain dimensions differ")
        if not np.array_equal(data.getcol("ANTENNA1", 0, 630), a):
            raise ValueError("baseline ordering differs")
        length = np.linalg.norm(data.getcol("UVW", 0, 630), axis=1)
        for pi, pol in enumerate(("XX", "YY")):
            long = np.flatnonzero(length > cuts[pi])
            train = long[np.arange(len(long)) % 5 != 0]
            held = long[np.arange(len(long)) % 5 == 0]
            # Stop at 119: 121 is a fully flagged, truncated final MS bin.
            for ti in range(1, len(gain)-2, 2):
                start = ti * 630
                vis = data.getcol("DATA", start, 630)[:, :, pi:pi+1]
                pred = model.getcol("MODEL_DATA", start, 630)[:, :, pi:pi+1]
                flag = data.getcol("FLAG", start, 630)[:, :, pi:pi+1]
                if np.all(flag):
                    continue
                variants = {
                    "before": np.ones(36),
                    "previous_only": gain[ti-1, pi],
                    "interpolated": np.sqrt(gain[ti-1, pi] * gain[ti+1, pi]),
                    "same_time_fit": gain[ti, pi],
                }
                for label, local in variants.items():
                    adjusted = vis / (local[a] * local[b])[:, None, None]
                    results[pol][label].append(score(adjusted, pred, flag,
                                                     train, held)[0])
    summary = {"source": str(source), "model": str(model_path),
               "fit": str(fit_path), "odd_10s_bins": len(results["XX"]["before"]),
               "held_baselines": "every fifth baseline above the hand-specific cut",
               "median_held_complex_residual": {
                   pol: {kind: float(np.median(values)) for kind, values in scores.items()}
                   for pol, scores in results.items()}}
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("model", type=Path)
    p.add_argument("fit", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    main(a.source, a.model, a.fit, a.output)

#!/usr/bin/env python3
"""Flag high Sun-model residual time-frequency cells in a copied solar MS.

The diagnostic scalar was fitted on separate baselines from those used for
flag decisions. This remains an exploratory, capture-specific mask.
"""

import argparse
import json
from pathlib import Path

from casacore.tables import table
import numpy as np


def main(source: Path, diagnostic: Path, output: Path,
         xx_threshold: float, yy_threshold: float, flag_last: bool):
    manifest = Path(str(output) + ".flagging.json")
    if output.exists() or manifest.exists():
        raise FileExistsError(output)
    with np.load(diagnostic) as saved:
        residual = saved["residual_fraction"]
        freq = saved["frequency_hz"]
    threshold = np.array([xx_threshold, yy_threshold])
    bad = np.isfinite(residual) & (residual > threshold[None, None, :])
    if flag_last:
        bad[-1, :, :] = True
    with table(str(source), ack=False) as src:
        nrows = src.nrows()
        if nrows != residual.shape[0] * 630:
            raise ValueError("diagnostic time bins do not match MS")
        with table(str(source / "SPECTRAL_WINDOW"), ack=False) as spectral:
            if not np.array_equal(spectral.getcell("CHAN_FREQ", 0), freq):
                raise ValueError("diagnostic frequency axis does not match MS")
        copied = src.copy(str(output), deep=True, valuecopy=True)
        copied.close()
    existing = np.zeros(2, np.int64)
    added = np.zeros(2, np.int64)
    with table(str(output), readonly=False, ack=False) as dest:
        for ti in range(residual.shape[0]):
            start = ti * 630
            flag = dest.getcol("FLAG", start, 630)
            existing += flag.sum(axis=(0, 1))
            before = flag.copy()
            flag |= bad[ti][None, :, :]
            added += (flag & ~before).sum(axis=(0, 1))
            dest.putcol("FLAG", flag, start, 630)
            dest.putcol("FLAG_ROW", flag.all(axis=(1, 2)), start, 630)
    report = {"source": str(source), "diagnostic": str(diagnostic),
              "output": str(output), "threshold": {"XX": xx_threshold,
                                                      "YY": yy_threshold},
              "flag_last_partial_integration": flag_last,
              "new_bad_time_frequency_cells": bad.sum(axis=(0, 1)).tolist(),
              "already_flagged_visibility_cells": existing.tolist(),
              "additional_flagged_visibility_cells": added.tolist(),
              "total_possible_visibility_cells_per_hand": int(nrows * len(freq))}
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("diagnostic", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--xx-threshold", type=float, default=0.4)
    p.add_argument("--yy-threshold", type=float, default=0.3)
    p.add_argument("--keep-last", action="store_true")
    a = p.parse_args()
    main(a.source, a.diagnostic, a.output, a.xx_threshold,
         a.yy_threshold, not a.keep_last)

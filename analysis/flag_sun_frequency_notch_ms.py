#!/usr/bin/env python3
"""Copy a solar measurement set and flag a measured narrow-band defect."""

import argparse
import json
from pathlib import Path

from casacore.tables import table
import numpy as np


def main(source: Path, output: Path, low_mhz: float, high_mhz: float):
    manifest = Path(str(output) + ".frequency_notch.json")
    if output.exists() or manifest.exists():
        raise FileExistsError(output)
    if not low_mhz < high_mhz:
        raise ValueError("notch endpoints must increase")
    with table(str(source / "SPECTRAL_WINDOW"), ack=False) as spectral:
        frequency = spectral.getcell("CHAN_FREQ", 0) / 1e6
    bad = (frequency >= low_mhz) & (frequency < high_mhz)
    if not bad.any() or bad.all():
        raise ValueError("frequency notch must flag a strict subset of channels")
    with table(str(source), ack=False) as src:
        copy = src.copy(str(output), deep=True, valuecopy=True)
        copy.close()
    new_flags = 0
    with table(str(output), readonly=False, ack=False) as dst:
        for start in range(0, dst.nrows(), 630):
            count = min(630, dst.nrows() - start)
            flags = dst.getcol("FLAG", start, count)
            new_flags += int(np.count_nonzero(~flags[:, bad, :]))
            flags[:, bad, :] = True
            dst.putcol("FLAG", flags, start, count)
            dst.putcol("FLAG_ROW", flags.all(axis=(1, 2)), start, count)
    result = {"source": str(source), "output": str(output),
              "notch_mhz": [low_mhz, high_mhz],
              "first_flagged_mhz": float(frequency[bad][0]),
              "last_flagged_mhz": float(frequency[bad][-1]),
              "flagged_fine_bins": int(bad.sum()),
              "newly_flagged_visibility_cells": new_flags,
              "diagnostic_only": True}
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--low-mhz", type=float, default=318.4)
    parser.add_argument("--high-mhz", type=float, default=319.7)
    args = parser.parse_args()
    main(args.source, args.output, args.low_mhz, args.high_mhz)

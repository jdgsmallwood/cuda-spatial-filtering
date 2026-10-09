#!/usr/bin/env python3
"""Apply a first-to-last-window linear phase drift to a calibrated Sun MS.

This is a retrospective diagnostic: it uses a gain fit at the end of the same
capture, so it is not an operational calibration recipe.
"""

import argparse
import json
from pathlib import Path

from astropy.time import Time
from casacore.tables import table
import numpy as np


def main(source: Path, validation: Path, output: Path) -> None:
    manifest = Path(str(output) + ".json")
    if output.exists() or manifest.exists():
        raise FileExistsError(output)
    with np.load(validation) as saved:
        if not np.array_equal(saved["antenna_order"], np.arange(1, 37)):
            raise ValueError("MS antenna IDs and fit order disagree")
        first = saved["gains"]
        last = saved["later_gains"]
        times_sec = Time(saved["times_jd"], format="jd").mjd * 86400
    phase_change = np.angle(last * first.conj())
    if phase_change.shape != (1080, 2, 36):
        raise ValueError("unexpected gain shape")
    with table(str(source), ack=False) as src:
        nrows = src.nrows()
        if nrows % 630:
            raise ValueError("expected 630 baseline rows per time bin")
        a = src.getcol("ANTENNA1", 0, 630)
        b = src.getcol("ANTENNA2", 0, 630)
        if (src.getcol("DATA", 0, 1).shape != (1, 1080, 2)
                or not np.array_equal(a, src.getcol("ANTENNA1", nrows-630, 630))
                or not np.array_equal(b, src.getcol("ANTENNA2", nrows-630, 630))):
            raise ValueError("unexpected MS layout")
        copied = src.copy(str(output), deep=True, valuecopy=True)
        copied.close()
    phase_a = np.take(phase_change, a, axis=2).transpose(2, 0, 1)
    phase_b = np.take(phase_change, b, axis=2).transpose(2, 0, 1)
    delta = phase_a - phase_b
    alphas = []
    with table(str(output), readonly=False, ack=False) as dest:
        for bin_index in range(nrows // 630):
            row = bin_index * 630
            timestamp = float(dest.getcol("TIME", row, 1)[0])
            alpha = float(np.clip((timestamp - times_sec[0])
                                  / (times_sec[2] - times_sec[0]), 0, 1))
            values = dest.getcol("DATA", row, 630)
            # The MS DATA column is the conjugate of the native covariance.
            # A native gain change g_i -> g_i exp(i phi_i) therefore adds
            # exp(+i(phi_i - phi_j)) to the already-corrected MS visibility.
            values *= np.exp(1j * alpha * delta)
            dest.putcol("DATA", values, row, 630)
            alphas.append(alpha)
    manifest.write_text(json.dumps({"source": str(source.resolve()),
                                    "validation": str(validation.resolve()),
                                    "method": "linear phase interpolation between "
                                    "first and last independently fitted windows",
                                    "retrospective_only": True,
                                    "alpha_first": alphas[0],
                                    "alpha_last": alphas[-1],
                                    "bins": len(alphas)}, indent=2) + "\n")
    print(f"wrote {output}; phase interpolation alpha {alphas[0]:.3f}..{alphas[-1]:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.source, args.validation, args.output)

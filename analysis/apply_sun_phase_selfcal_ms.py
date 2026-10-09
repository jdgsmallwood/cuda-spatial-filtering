#!/usr/bin/env python3
"""Apply prior-bin Sun phase selfcal to an offline calibrated MS copy."""

import argparse
import json
from pathlib import Path

from casacore.tables import table
import numpy as np


def main(source: Path, fit_path: Path, output: Path):
    manifest = Path(str(output) + ".json")
    if output.exists() or manifest.exists():
        raise FileExistsError(output)
    with np.load(fit_path) as fit:
        phase = fit["phase"]
        a_expected = fit["antenna_a"]
        b_expected = fit["antenna_b"]
        scores = fit["scores"]
    if not np.isfinite(phase).all():
        raise ValueError("phase gains must be finite")
    with table(str(source), ack=False) as src:
        if src.nrows() != len(phase) * 630:
            raise ValueError("phase time bins do not match MS")
        a = src.getcol("ANTENNA1", 0, 630)
        b = src.getcol("ANTENNA2", 0, 630)
        if not np.array_equal(a, a_expected) or not np.array_equal(b, b_expected):
            raise ValueError("baseline mapping differs from fit")
        copied = src.copy(str(output), deep=True, valuecopy=True)
        copied.close()
    with table(str(output), readonly=False, ack=False) as dest:
        for ti in range(len(phase)):
            if ti == 0:
                continue
            start = ti * 630
            local = phase[ti-1]
            correction = np.exp(1j * (local[:, a] - local[:, b])).T
            values = dest.getcol("DATA", start, 630)
            values /= correction[:, None, :]
            dest.putcol("DATA", values, start, 630)
            if ti % 20 == 0:
                print(f"applied {ti+1}/{len(phase)} time bins", flush=True)
    report = {"source": str(source), "fit": str(fit_path),
              "output": str(output), "time_bins": len(phase),
              "method": "previous-bin broadband antenna phase; no amplitude change",
              "held_out_odd_bin_complex_residual": {
                  pol: {"before": float(np.nanmedian(scores[1:-1:2, pi, 0])),
                        "after": float(np.nanmedian(scores[1:-1:2, pi, 2]))}
                  for pi, pol in enumerate(("XX", "YY"))},
              "diagnostic_only": True}
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("fit", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    main(a.source, a.fit, a.output)

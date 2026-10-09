#!/usr/bin/env python3
"""Apply validated per-time antenna amplitude gains to an offline MS copy."""

import argparse
import json
from pathlib import Path

from casacore.tables import table
import numpy as np


def main(source: Path, fit_path: Path, output: Path, lag_one: bool = False):
    manifest = Path(str(output) + ".json")
    if output.exists() or manifest.exists():
        raise FileExistsError(output)
    with np.load(fit_path) as fit:
        gain = fit["gain"]
        a_expected = fit["antenna_a"]
        b_expected = fit["antenna_b"]
        validation = fit["validation"]
    if not np.isfinite(gain).all() or (gain <= 0).any():
        raise ValueError("amplitude gains must be finite and positive")
    with table(str(source), ack=False) as src:
        if src.nrows() != len(gain) * 630:
            raise ValueError("gain time bins do not match MS")
        a = src.getcol("ANTENNA1", 0, 630)
        b = src.getcol("ANTENNA2", 0, 630)
        if not np.array_equal(a, a_expected) or not np.array_equal(b, b_expected):
            raise ValueError("baseline mapping differs from fit")
        copied = src.copy(str(output), deep=True, valuecopy=True)
        copied.close()
    with table(str(output), readonly=False, ack=False) as dest:
        for ti in range(len(gain)):
            start = ti * 630
            values = dest.getcol("DATA", start, 630)
            local_gain = (gain[ti-1] if ti > 0 else np.ones_like(gain[0])) \
                if lag_one else gain[ti]
            denominator = (local_gain[:, a] * local_gain[:, b]).T
            values /= denominator[:, None, :]
            dest.putcol("DATA", values, start, 630)
            if ti % 20 == 0:
                print(f"applied {ti+1}/{len(gain)} time bins", flush=True)
    report = {"source": str(source), "fit": str(fit_path),
              "output": str(output), "time_bins": len(gain),
              "same_bin_fit_held_out_complex_residual_median_before_after": {
                  pol: [float(np.nanmedian(validation[:-1, pi, 0])),
                        float(np.nanmedian(validation[:-1, pi, 2]))]
                  for pi, pol in enumerate(("XX", "YY"))},
              "method": ("previous-bin real antenna amplitude selfcal; no phase change"
                         if lag_one else "same-bin real antenna amplitude selfcal; no phase change"),
              "causal_previous_bin": lag_one,
              "temporal_validation": ("sun_amplitude_time_transfer.json"
                                      if lag_one else None),
              "diagnostic_only": True}
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("fit", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--lag-one", action="store_true",
                   help="Apply the previous 10-second bin's amplitude gains")
    a = p.parse_args()
    main(a.source, a.fit, a.output, a.lag_one)

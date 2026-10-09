#!/usr/bin/env python3
"""Subtract a time/frequency dependent moving Sun disk from corrected visibilities.

The disk phase comes from an independent geometric MODEL_DATA measurement set;
one robust *real* apparent flux is fitted per integration, fine channel and
hand. This is an offline diagnostic, not an absolute-flux calibration.
"""

import argparse
import json
from pathlib import Path

from casacore.tables import table
import numpy as np

from diagnose_sun_model_residuals import fit_real_amplitude


def main(source: Path, model_path: Path, output: Path, min_baseline: float):
    manifest = Path(str(output) + ".subtraction.json")
    if output.exists() or manifest.exists():
        raise FileExistsError(output)
    with table(str(source), ack=False) as src, table(str(model_path), ack=False) as model:
        if src.nrows() != model.nrows() or src.nrows() % 630:
            raise ValueError("measurement set row counts differ")
        for name in ("TIME", "ANTENNA1", "ANTENNA2"):
            if not np.array_equal(src.getcol(name), model.getcol(name)):
                raise ValueError(f"{name} columns differ")
        copied = src.copy(str(output), deep=True, valuecopy=True)
        copied.close()
    fluxes = []
    with table(str(output), readonly=False, ack=False) as dest, table(
            str(model_path), ack=False) as model:
        uvw = dest.getcol("UVW", 0, 630)
        train = np.flatnonzero(np.linalg.norm(uvw, axis=1) > min_baseline)
        for ti in range(dest.nrows() // 630):
            start = ti * 630
            vis = dest.getcol("DATA", start, 630)
            pred = model.getcol("MODEL_DATA", start, 630)
            flag = dest.getcol("FLAG", start, 630)
            flux = fit_real_amplitude(vis, pred, flag, train)
            valid = np.isfinite(flux)
            # An unmodelled cell stays flagged rather than silently receiving
            # a zero model. Normally only pre-existing bad cells are invalid.
            flag |= ~valid[None, :, :]
            vis -= np.where(valid, flux, 0)[None, :, :] * pred
            dest.putcol("DATA", vis, start, 630)
            dest.putcol("FLAG", flag, start, 630)
            dest.putcol("FLAG_ROW", flag.all(axis=(1, 2)), start, 630)
            fluxes.append(np.nanmedian(flux, axis=0).tolist())
            if ti % 20 == 0:
                print(f"subtracted {ti+1}/{dest.nrows()//630} integrations", flush=True)
    report = {"source": str(source), "model": str(model_path),
              "output": str(output), "fit_min_baseline_m": min_baseline,
              "method": "independent moving Sun disk phase, robust real apparent flux "
                        "per time/fine channel/hand across long baselines",
              "apparent_flux_median_by_integration": fluxes,
              "diagnostic_only": True}
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("model", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--min-baseline", type=float, default=7)
    a = p.parse_args()
    main(a.source, a.model, a.output, a.min_baseline)

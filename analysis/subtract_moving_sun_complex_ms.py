#!/usr/bin/env python3
"""Subtract a held-baseline-validated complex moving-Sun amplitude.

The imaginary scalar can absorb a residual common Sun phase. This is an
offline diagnostic, and its ability to preserve off-Sun sources must be
checked separately.
"""

import argparse
import json
from pathlib import Path
import warnings

from casacore.tables import table
import numpy as np


def complex_flux(vis, pred, flag, train):
    usable = (~flag[train]) & (abs(pred[train]) > .5)
    ratio = np.divide(vis[train], pred[train],
                      out=np.zeros_like(vis[train]), where=abs(pred[train]) > .5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        real = np.nanmedian(np.where(usable, ratio.real, np.nan), axis=0)
        imag = np.nanmedian(np.where(usable, ratio.imag, np.nan), axis=0)
    valid = (np.sum(usable, axis=0) >= 20) & np.isfinite(real) & np.isfinite(imag) & (real > 0)
    return np.where(valid, real + 1j * imag, np.nan + 0j)


def held_error(vis, pred, flag, held, flux, real_scale):
    difference = vis[held] - flux[None, :, :] * pred[held]
    denominator = np.abs(real_scale[None, :, :] * pred[held])
    error = np.divide(abs(difference), denominator,
                      out=np.full_like(denominator, np.nan, dtype=float),
                      where=denominator > 0)
    error[flag[held]] = np.nan
    return error


def main(source: Path, model_path: Path, output: Path):
    manifest = Path(str(output) + ".complex_subtraction.json")
    if output.exists() or manifest.exists():
        raise FileExistsError(output)
    with table(str(source), ack=False) as src, table(str(model_path), ack=False) as model:
        if src.nrows() != model.nrows() or src.nrows() % 630:
            raise ValueError("measurement-set rows differ")
        for name in ("TIME", "ANTENNA1", "ANTENNA2"):
            if not np.array_equal(src.getcol(name), model.getcol(name)):
                raise ValueError(f"{name} differs")
        copied = src.copy(str(output), deep=True, valuecopy=True)
        copied.close()
    scores = []
    imag_fraction = []
    with table(str(output), readonly=False, ack=False) as dst, table(
            str(model_path), ack=False) as model:
        length = np.linalg.norm(dst.getcol("UVW", 0, 630), axis=1)
        long = np.flatnonzero(length > 7)
        train = long[np.arange(len(long)) % 5 != 0]
        held = long[np.arange(len(long)) % 5 == 0]
        for ti in range(dst.nrows() // 630):
            start = ti * 630
            vis = dst.getcol("DATA", start, 630)
            pred = model.getcol("MODEL_DATA", start, 630)
            flag = dst.getcol("FLAG", start, 630)
            flux = complex_flux(vis, pred, flag, train)
            valid = np.isfinite(flux)
            real_scale = np.where(valid, flux.real, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                before = np.nanmedian(held_error(vis, pred, flag, held,
                                                  np.zeros_like(flux), real_scale), axis=0)
                real_error = np.nanmedian(held_error(vis, pred, flag, held,
                                                     real_scale, real_scale), axis=0)
                complex_error = np.nanmedian(held_error(vis, pred, flag, held,
                                                        flux, real_scale), axis=0)
                imag_fraction.append(np.nanmedian(abs(flux.imag) / real_scale, axis=0))
            scores.append(np.stack([before, real_error, complex_error], axis=-1))
            flag |= ~valid[None, :, :]
            vis -= np.where(valid, flux, 0)[None, :, :] * pred
            dst.putcol("DATA", vis, start, 630)
            dst.putcol("FLAG", flag, start, 630)
            dst.putcol("FLAG_ROW", flag.all(axis=(1, 2)), start, 630)
            if ti % 20 == 0:
                print(f"subtracted {ti+1}/{dst.nrows()//630} bins", flush=True)
    scores = np.asarray(scores)
    imag_fraction = np.asarray(imag_fraction)
    summary = {"source": str(source), "model": str(model_path),
               "output": str(output), "training_baselines": len(train),
               "held_baselines": len(held),
               "held_median_fraction": {
                   pol: {name: float(np.nanmedian(scores[:-1, :, pi, ki]))
                         for ki, name in enumerate(("before", "real", "complex"))}
                   for pi, pol in enumerate(("XX", "YY"))},
               "median_abs_imag_over_real": {
                   pol: float(np.nanmedian(imag_fraction[:-1, pi]))
                   for pi, pol in enumerate(("XX", "YY"))},
               "method": "real/imag median Sun coefficient per time, fine "
                         "channel and hand on 4/5 of baselines >7 m; score "
                         "remaining fifth; subtract on copied MS",
               "diagnostic_only": True}
    manifest.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("sun_model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.source, args.sun_model, args.output)

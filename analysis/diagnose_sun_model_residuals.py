#!/usr/bin/env python3
"""Score a moving Sun model on held-out baselines of a corrected solar MS.

The fitted scalar is real and independent for each time, frequency, and hand.
It only removes the apparent Sun amplitude; it cannot absorb baseline phases.
The fit uses four fifths of long baselines and reports the remaining fifth.
"""

import argparse
import json
from pathlib import Path

from casacore.tables import table
import numpy as np


def fit_real_amplitude(vis, model, flag, train, min_model=0.5):
    """Robust apparent flux per channel and hand from selected baselines."""
    usable = (~flag[train]) & (np.abs(model[train]) > min_model)
    ratio = np.divide(vis[train], model[train], out=np.zeros_like(vis[train]),
                      where=np.abs(model[train]) > min_model)
    with np.errstate(all="ignore"):
        flux = np.nanmedian(np.where(usable, ratio.real, np.nan), axis=0)
    return np.where(np.isfinite(flux) & (flux > 0), flux, np.nan)


def main(corrected: Path, model_ms: Path, output: Path, min_baseline: float):
    if output.exists():
        raise FileExistsError(output)
    with table(str(corrected), ack=False) as data, table(str(model_ms), ack=False) as model:
        if data.nrows() != model.nrows() or data.nrows() % 630:
            raise ValueError("different or unexpected measurement-set row counts")
        for name in ("TIME", "ANTENNA1", "ANTENNA2"):
            if not np.array_equal(data.getcol(name), model.getcol(name)):
                raise ValueError(f"{name} columns do not match")
        uvw = data.getcol("UVW", 0, 630)
        long = np.linalg.norm(uvw, axis=1) > min_baseline
        indices = np.flatnonzero(long)
        train = indices[np.arange(len(indices)) % 5 != 0]
        held = indices[np.arange(len(indices)) % 5 == 0]
        nt = data.nrows() // 630
        nf = data.getcell("DATA", 0).shape[0]
        residual = np.full((nt, nf, 2), np.nan, np.float32)
        phase_residual = np.full_like(residual, np.nan)
        amp = np.full_like(residual, np.nan)
        full_rms = np.full((nt, 2), np.nan, np.float32)
        per_baseline = np.zeros((nt, 630, 2), np.float32)
        valid_count = np.zeros((nt, 630, 2), np.int32)
        for ti in range(nt):
            start = ti * 630
            vis = data.getcol("DATA", start, 630)
            pred = model.getcol("MODEL_DATA", start, 630)
            flag = data.getcol("FLAG", start, 630)
            flux = fit_real_amplitude(vis, pred, flag, train)
            amp[ti] = flux
            difference = vis - flux[None, :, :] * pred
            denom = np.abs(flux[None, :, :] * pred)
            error = np.divide(np.abs(difference), denom,
                              out=np.full_like(denom, np.nan, dtype=float),
                              where=denom > 0)
            error[flag] = np.nan
            model_phase_error = np.angle(vis * pred.conj())
            model_phase_error[flag] = np.nan
            with np.errstate(all="ignore"):
                residual[ti] = np.nanmedian(error[held], axis=0)
                phase_residual[ti] = np.nanmedian(
                    np.abs(model_phase_error[held]), axis=0)
                full_rms[ti] = np.sqrt(np.nanmean(error[held] ** 2, axis=(0, 1)))
                per_baseline[ti] = np.nanmedian(error, axis=1)
            valid_count[ti] = np.sum(~flag & np.isfinite(error), axis=1)
            if ti % 20 == 0:
                print(f"scored {ti+1}/{nt} integrations", flush=True)
    with table(str(corrected / "SPECTRAL_WINDOW"), ack=False) as spectral:
        freq = spectral.getcell("CHAN_FREQ", 0)
    np.savez_compressed(output, residual_fraction=residual,
                        phase_residual_rad=phase_residual,
                        apparent_flux=amp, full_rms=full_rms,
                        per_baseline_residual=per_baseline,
                        valid_count=valid_count, frequency_hz=freq,
                        baseline_indices=indices, held_indices=held)
    summary = {"corrected_ms": str(corrected), "model_ms": str(model_ms),
               "min_baseline_m": min_baseline, "long_baselines": len(indices),
               "held_baselines": len(held), "times": nt,
               "median_held_fraction": np.nanmedian(residual, axis=(0, 1)).tolist(),
               "p95_held_fraction": np.nanpercentile(residual, 95, axis=(0, 1)).tolist(),
               "worst_times": {pol: sorted(range(nt),
                   key=lambda i: -float(np.nanmedian(residual[i, :, pi])))[:10]
                   for pi, pol in enumerate(("XX", "YY"))}}
    Path(str(output) + ".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corrected", type=Path)
    parser.add_argument("model_ms", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-baseline", type=float, default=7)
    args = parser.parse_args()
    main(args.corrected, args.model_ms, args.output, args.min_baseline)

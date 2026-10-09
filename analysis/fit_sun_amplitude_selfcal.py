#!/usr/bin/env python3
"""Fit time-dependent antenna amplitudes from the moving Sun, with holdouts.

This is an offline imaging experiment. One real positive antenna gain is
estimated per 10 s integration and hand from long-baseline Sun amplitudes,
averaged over accepted frequencies. Every fifth long baseline is withheld
from the gain fit and used for a complex-visibility prediction score.
"""

import argparse
import json
from pathlib import Path
import warnings

from casacore.tables import table
import numpy as np
from scipy.optimize import least_squares

if __package__:
    from .diagnose_sun_model_residuals import fit_real_amplitude
else:
    from diagnose_sun_model_residuals import fit_real_amplitude


def fit_antenna_amplitudes(pair_amplitude, antenna_a, antenna_b,
                           train_indices, nant=36):
    """Robust log-amplitude solve, normalized to median live gain one."""
    usable = train_indices[np.isfinite(pair_amplitude[train_indices])
                           & (pair_amplitude[train_indices] > 0)]
    involved = np.bincount(np.r_[antenna_a[usable], antenna_b[usable]],
                           minlength=nant)
    active = np.flatnonzero(involved >= 5)
    usable = usable[np.isin(antenna_a[usable], active)
                    & np.isin(antenna_b[usable], active)]
    if len(active) < 10 or len(usable) < 3 * len(active):
        raise ValueError("too few supported antennas or baselines")
    reference = int(active[0])
    variable = active[1:]
    lookup = {int(ant): col for col, ant in enumerate(variable)}
    design = np.zeros((len(usable), len(active)), dtype=float)
    for row, baseline in enumerate(usable):
        for ant in (antenna_a[baseline], antenna_b[baseline]):
            if ant != reference:
                design[row, lookup[int(ant)]] = 1
        design[row, -1] = 1  # arbitrary apparent Sun amplitude
    if np.linalg.matrix_rank(design) != len(active):
        raise ValueError("antenna amplitude graph is rank deficient")
    observed = np.log(pair_amplitude[usable])
    initial = np.linalg.lstsq(design, observed, rcond=None)[0]
    fit = least_squares(lambda params: design @ params - observed, initial,
                        loss="soft_l1", f_scale=0.05, max_nfev=25)
    log_gains = np.zeros(nant)
    log_gains[variable] = fit.x[:-1]
    log_gains[active] -= np.median(log_gains[active])
    gains = np.ones(nant)
    gains[active] = np.exp(log_gains[active])
    return gains, active


def score(vis, model, flag, train, held):
    scalar = fit_real_amplitude(vis, model, flag, train)
    denominator = np.abs(scalar[None, :, :] * model[held])
    difference = np.abs(vis[held] - scalar[None, :, :] * model[held])
    score_array = np.divide(difference, denominator,
                            out=np.full_like(denominator, np.nan, dtype=float),
                            where=denominator > 0)
    score_array[flag[held]] = np.nan
    return float(np.nanmedian(score_array)), float(np.nanpercentile(score_array, 90))


def main(corrected: Path, model_path: Path, output: Path):
    if output.exists() or Path(str(output) + ".json").exists():
        raise FileExistsError(output)
    cuts = [3, 7]
    with table(str(corrected), ack=False) as data, table(str(model_path), ack=False) as model:
        if data.nrows() != model.nrows() or data.nrows() % 630:
            raise ValueError("MS row counts differ")
        for name in ("TIME", "ANTENNA1", "ANTENNA2"):
            if not np.array_equal(data.getcol(name), model.getcol(name)):
                raise ValueError(f"{name} differs between MS files")
        a = data.getcol("ANTENNA1", 0, 630)
        b = data.getcol("ANTENNA2", 0, 630)
        length = np.linalg.norm(data.getcol("UVW", 0, 630), axis=1)
        ntime = data.nrows() // 630
        gain = np.ones((ntime, 2, 36), np.float32)
        supported = np.zeros((ntime, 2, 36), bool)
        validation = np.full((ntime, 2, 4), np.nan, np.float32)
        train_lists, held_lists = [], []
        for cut in cuts:
            long = np.flatnonzero(length > cut)
            train_lists.append(long[np.arange(len(long)) % 5 != 0])
            held_lists.append(long[np.arange(len(long)) % 5 == 0])
        for ti in range(ntime):
            start = ti * 630
            vis = data.getcol("DATA", start, 630)
            pred = model.getcol("MODEL_DATA", start, 630)
            flag = data.getcol("FLAG", start, 630)
            for pi in range(2):
                train, held = train_lists[pi], held_lists[pi]
                if np.all(flag[:, :, pi]):
                    continue
                ratio = np.divide(vis[:, :, pi], pred[:, :, pi],
                                  out=np.full_like(vis[:, :, pi], np.nan),
                                  where=np.abs(pred[:, :, pi]) > .5)
                ratio[flag[:, :, pi]] = np.nan
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    pair_amp = np.nanmedian(ratio.real, axis=1)
                gain[ti, pi], active = fit_antenna_amplitudes(pair_amp, a, b, train)
                supported[ti, pi, active] = True
                before = score(vis[:, :, pi:pi+1], pred[:, :, pi:pi+1],
                               flag[:, :, pi:pi+1], train, held)
                adjusted = vis[:, :, pi:pi+1] / (
                    gain[ti, pi, a] * gain[ti, pi, b])[:, None, None]
                after = score(adjusted, pred[:, :, pi:pi+1],
                              flag[:, :, pi:pi+1], train, held)
                validation[ti, pi] = [*before, *after]
            if ti % 20 == 0:
                print(f"fitted {ti+1}/{ntime} time bins", flush=True)
    np.savez_compressed(output, gain=gain, supported=supported,
                        validation=validation, antenna_a=a, antenna_b=b,
                        cuts_m=np.array(cuts))
    summary = {"corrected_ms": str(corrected), "model_ms": str(model_path),
               "times": ntime, "cuts_m": {"XX": 3, "YY": 7},
               "method": "robust per-time per-hand log-amplitude selfcal; "
                         "one in five long baselines withheld",
               "scores": {pol: {"before_median": float(np.nanmedian(validation[:-1, pi, 0])),
                                "after_median": float(np.nanmedian(validation[:-1, pi, 2])),
                                "before_p90": float(np.nanmedian(validation[:-1, pi, 1])),
                                "after_p90": float(np.nanmedian(validation[:-1, pi, 3])),
                                "gain_p05_p50_p95": np.percentile(
                                    gain[:-1, pi][supported[:-1, pi]], [5, 50, 95]).tolist()}
                          for pi, pol in enumerate(("XX", "YY"))}}
    Path(str(output) + ".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("corrected", type=Path)
    p.add_argument("model_ms", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    main(a.corrected, a.model_ms, a.output)

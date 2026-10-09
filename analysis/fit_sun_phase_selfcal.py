#!/usr/bin/env python3
"""Test short-timescale Sun phase selfcal with baseline and time holdouts.

Fits one broadband antenna phase per 10 s bin and hand from masked long
baselines. Every fifth baseline is held out. The fit does not change the
existing fine-frequency phase table or infer a new delay slope.
"""

import argparse
import json
from pathlib import Path

from casacore.tables import table
import numpy as np
from scipy.optimize import least_squares

if __package__:
    from .fit_sun_amplitude_selfcal import score
else:
    from fit_sun_amplitude_selfcal import score


def fit_antenna_phases(pair_phasor, counts, a, b, train, nant=36):
    usable = train[(counts[train] > 50) & np.isfinite(pair_phasor[train])]
    involved = np.bincount(np.r_[a[usable], b[usable]], minlength=nant)
    active = np.flatnonzero(involved >= 5)
    usable = usable[np.isin(a[usable], active) & np.isin(b[usable], active)]
    if len(active) < 10 or len(usable) < 3 * len(active):
        raise ValueError("too few phase-supported antennas")
    reference = active[0]
    columns = {int(ant): col for col, ant in enumerate(active[1:])}
    incidence = np.zeros((len(usable), len(active)-1))
    for row, baseline in enumerate(usable):
        if a[baseline] != reference:
            incidence[row, columns[int(a[baseline])]] = 1
        if b[baseline] != reference:
            incidence[row, columns[int(b[baseline])]] = -1
    if np.linalg.matrix_rank(incidence) < len(active)-1:
        raise ValueError("antenna phase graph is rank deficient")
    observed = np.angle(pair_phasor[usable])
    result = least_squares(
        lambda x: np.angle(np.exp(1j * (incidence @ x - observed))),
        np.zeros(len(active)-1), loss="soft_l1", f_scale=.1, max_nfev=30)
    phases = np.zeros(nant)
    phases[active[1:]] = result.x
    phases[active] -= np.angle(np.mean(np.exp(1j*phases[active])))
    return phases, active


def main(corrected: Path, model_path: Path, output: Path):
    if output.exists() or Path(str(output) + ".json").exists():
        raise FileExistsError(output)
    with table(str(corrected), ack=False) as data, table(str(model_path), ack=False) as model:
        if data.nrows() != model.nrows() or data.nrows() % 630:
            raise ValueError("measurement-set row counts differ")
        for name in ("TIME", "ANTENNA1", "ANTENNA2"):
            if not np.array_equal(data.getcol(name), model.getcol(name)):
                raise ValueError(f"{name} differs")
        a = data.getcol("ANTENNA1", 0, 630)
        b = data.getcol("ANTENNA2", 0, 630)
        length = np.linalg.norm(data.getcol("UVW", 0, 630), axis=1)
        ntime = data.nrows() // 630
        phase = np.zeros((ntime, 2, 36), np.float32)
        supported = np.zeros((ntime, 2, 36), bool)
        scores = np.full((ntime, 2, 3), np.nan, np.float32)
        baseline_groups = []
        for cut in (3, 7):
            long = np.flatnonzero(length > cut)
            baseline_groups.append((long[np.arange(len(long)) % 5 != 0],
                                    long[np.arange(len(long)) % 5 == 0]))
        for ti in range(ntime):
            start = ti * 630
            vis = data.getcol("DATA", start, 630)
            pred = model.getcol("MODEL_DATA", start, 630)
            flag = data.getcol("FLAG", start, 630)
            for pi in range(2):
                if np.all(flag[:, :, pi]):
                    continue
                train, held = baseline_groups[pi]
                z = np.divide(vis[:, :, pi], pred[:, :, pi],
                              out=np.zeros_like(vis[:, :, pi]),
                              where=np.abs(pred[:, :, pi]) > .5)
                z[flag[:, :, pi]] = 0
                magnitude = np.abs(z)
                z = np.divide(z, magnitude, out=np.zeros_like(z),
                              where=magnitude > 0)
                counts = np.count_nonzero(z, axis=1)
                pair_phasor = np.sum(z, axis=1)
                phase[ti, pi], active = fit_antenna_phases(
                    pair_phasor, counts, a, b, train)
                supported[ti, pi, active] = True
                before = score(vis[:, :, pi:pi+1], pred[:, :, pi:pi+1],
                               flag[:, :, pi:pi+1], train, held)[0]
                correction = np.exp(1j*(phase[ti, pi, a]-phase[ti, pi, b]))
                adjusted = vis[:, :, pi:pi+1] / correction[:, None, None]
                same = score(adjusted, pred[:, :, pi:pi+1],
                             flag[:, :, pi:pi+1], train, held)[0]
                scores[ti, pi, :2] = (before, same)
            if ti % 20 == 0:
                print(f"fitted {ti+1}/{ntime} time bins", flush=True)
        for ti in range(1, ntime-1, 2):
            start = ti * 630
            vis = data.getcol("DATA", start, 630)
            pred = model.getcol("MODEL_DATA", start, 630)
            flag = data.getcol("FLAG", start, 630)
            for pi in range(2):
                if np.all(flag[:, :, pi]):
                    continue
                train, held = baseline_groups[pi]
                correction = np.exp(1j*(phase[ti-1, pi, a]-phase[ti-1, pi, b]))
                adjusted = vis[:, :, pi:pi+1] / correction[:, None, None]
                scores[ti, pi, 2] = score(adjusted, pred[:, :, pi:pi+1],
                                          flag[:, :, pi:pi+1], train, held)[0]
    np.savez_compressed(output, phase=phase, supported=supported,
                        scores=scores, antenna_a=a, antenna_b=b)
    summary = {"corrected_ms": str(corrected), "model_ms": str(model_path),
               "cuts_m": {"XX": 3, "YY": 7},
               "method": "broadband phase per antenna/hand/10-second bin; "
                         "one in five long baselines withheld",
               "held_baseline_median": {pol: {
                   "before": float(np.nanmedian(scores[:-1, pi, 0])),
                   "same_bin": float(np.nanmedian(scores[:-1, pi, 1])),
                   "odd_bins_before": float(np.nanmedian(scores[1:-1:2, pi, 0])),
                   "odd_bins_previous_bin": float(np.nanmedian(scores[1:-1:2, pi, 2]))}
                   for pi, pol in enumerate(("XX", "YY"))}}
    Path(str(output) + ".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("corrected", type=Path)
    p.add_argument("model_ms", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    main(a.corrected, a.model_ms, a.output)

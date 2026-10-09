#!/usr/bin/env python3
"""Test fresh PN phase transfer into independent sky times/frequencies.

Use the capture's embedded input map. Fit a constant plus residual delay on
coarse 400--403 in the first time half. Predict 404--414 in the second half.
All baseline selection comes from early training cells, geometry and PN
repeatability; late sky significance never selects the reported results.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument("capture", type=Path)
    p.add_argument("pn", type=Path)
    p.add_argument("slopes", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    with np.load(a.cache) as z:
        f = z["frequency_hz"]
        use = f < 324e6
        f = f[use]
        raw = z["raw"][:, use, 1].astype(complex)
        ids, pos = z["antenna_ids"], z["positions_m"]
        training, direction = z["training"].astype(bool), z["Sun_direction"]
    with h5py.File(a.capture) as h:
        manifest = json.loads(h["audit/run_manifest_json"][()])
    raw_for = {r["antenna_id"]: r["fpga_id"] * 20 + r["fpga_stream_id"]
        for r in manifest["forward_mapping"] if r["canonical_polarization"] == 0 and r["antenna_id"] > 0}
    ri = np.array([raw_for[int(ant)] for ant in ids])
    with np.load(a.pn) as z:
        phase = z["relative_phase_rad"]
        line = z["line_offsets_hz"]
    with np.load(a.slopes) as z:
        delay = z["within_coarse_median_delay_ns"]
    delay[:, 2] = 0  # PN reference input has phase zero by definition.
    intercept = np.angle(np.mean(np.exp(1j * (phase -
        2 * np.pi * line[None, None, :, None] * delay[:, None, None, :] * 1e-9)), axis=2))
    ci = np.rint(f / 781250.).astype(int) - 400
    off = f - (ci + 400) * 781250.
    predicted = intercept[:, ci][:, :, ri] + 2 * np.pi * off[None, :, None] * delay[:, None, ri] * 1e-9
    pn_phase = np.angle(np.mean(np.exp(1j * predicted), axis=0))
    stable = (abs(np.angle(np.exp(1j * (predicted[0] - predicted[1])))) < .5)
    stable &= abs(delay[0, ri] - delay[1, ri])[None] < 100
    lengths = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    ia, ib = np.where(np.triu(lengths >= 5, 1))
    geo = direction @ pos.T
    fringe = np.exp(2j * np.pi * f[None, :, None] / 299792458. *
        (geo[:, ia] - geo[:, ib])[:, None])
    data = raw[:, :, ia, ib]
    coefficients, snrs, modulation = [], [], []
    for take in (training, ~training):
        ph, d = fringe[take], data[take]
        ph = ph - ph.mean(axis=0)
        var = np.sum(abs(ph) ** 2, axis=0)
        coefficient = np.divide(np.sum(ph.conj() * (d - d.mean(axis=0)), axis=0),
            var, out=np.zeros_like(var, dtype=complex), where=var > 0)
        residual = d - d.mean(axis=0) - coefficient[None] * ph
        sigma = np.sqrt(np.mean(abs(residual) ** 2, axis=0) / np.maximum(var, 1e-30))
        coefficients.append(coefficient)
        snrs.append(abs(coefficient) / np.maximum(sigma, 1e-30))
        modulation.append(var / take.sum())
    coefficients, snrs, modulation = np.array(coefficients), np.array(snrs), np.array(modulation)
    pn = pn_phase[:, ia] - pn_phase[:, ib]
    good_pn = stable[:, ia] & stable[:, ib]
    train_freq = ci < 4
    held_freq = (ci >= 4) & (ci < 15)
    train_ok = train_freq[:, None] & good_pn & (snrs[0] >= 3) & (modulation[0] >= .05)
    baseline_ok = train_ok.sum(axis=0) >= 9
    held_ok = held_freq[:, None] & good_pn & baseline_ok[None] & (modulation[1] >= .05)
    trial = np.arange(-1000., 1000.01, 1.)
    freq_off = f - 312.5e6
    grid = 2 * np.pi * freq_off[:, None] * trial[None] * 1e-9
    results, errors, fitted_delays = {}, {}, {}
    for name, sign in (("sky_only", 0), ("PN_then_sky", 1), ("reversed_PN_control", -1)):
        error = np.full(pn.shape, np.nan)
        delay_fit = np.full(len(ia), np.nan)
        for b in np.flatnonzero(baseline_ok):
            take = train_ok[:, b]
            phase_fit = np.angle(coefficients[0, :, b]) - sign * pn[:, b]
            fit = np.mean(np.exp(1j * (phase_fit[take, None] - grid[take])), axis=0)
            best = np.argmax(abs(fit))
            delay_fit[b] = trial[best]
            prediction = np.angle(fit[best]) + grid[:, best] + sign * pn[:, b]
            error[held_ok[:, b], b] = np.angle(coefficients[1, held_ok[:, b], b] *
                                             np.exp(-1j * prediction[held_ok[:, b]]))
        good = np.isfinite(error)
        cosine = np.cos(error)
        same = (ri[ia] // 20) == (ri[ib] // 20)
        results[name] = dict(predictions=int(good.sum()), baselines=int(baseline_ok.sum()),
            median_abs_phase_error_deg=float(np.rad2deg(np.median(abs(error[good])))),
            median_cosine=float(np.median(cosine[good])), mean_cosine=float(np.mean(cosine[good])),
            same_fpga_median_cosine=float(np.nanmedian(cosine[:, same])),
            cross_fpga_median_cosine=float(np.nanmedian(cosine[:, ~same])),
            median_abs_residual_delay_ns=float(np.nanmedian(abs(delay_fit))))
        errors[name] = error
        fitted_delays[name] = delay_fit
    report = dict(capture=str(a.capture), pn=str(a.pn), slopes=str(a.slopes),
        train="coarse 400--403, first time half; one baseline phase and residual delay",
        held="coarse 404--414, second time half; selection excludes all held sky data",
        scores=results,
        limitations=["Per-baseline transfer diagnostic, not final antenna calibration.",
            "PN noise calibration was captured after the sky observation in the same verified FPGA boot and tuning.",
            "The input map is the saved capture map, not the current runtime map.",
            "PN local slope averaged across the full band; per-coarse phase intercept retained.",
            "PN injection bypasses the antenna/front-end and cannot alone establish sky gains.",
            "The reversed-PN control is essential: a flexible residual delay can absorb much of a sign reversal. Similar held scores do not independently validate the physical PN sign or delay decomposition."])
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(a.output / "transfer.npz", frequency_hz=f,
        pairs=np.column_stack((ids[ia], ids[ib])), coefficients=coefficients,
        significance=snrs, train_ok=train_ok, held_ok=held_ok, pn_phase=pn,
        **{name + "_phase_error": e for name, e in errors.items()},
        **{name + "_delay_ns": d for name, d in fitted_delays.items()})
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    for name, err in errors.items():
        ax.plot(f[held_freq] / 1e6, np.nanmedian(np.cos(err[held_freq]), axis=1), ".-", label=name)
    ax.set(xlabel="Held frequency (MHz)", ylabel="Median held Sun phase cosine", ylim=(-1, 1),
        title="Fresh PN calibration + early sky residual: later times and other frequencies")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(a.output / "pn_sky_transfer.png", dpi=160)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

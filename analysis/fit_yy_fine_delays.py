#!/usr/bin/env python3
"""Fit diagnostic YY gains on all 27 retained fine bins and test delay slopes.

This is an offline candidate fit. No FPGA delays or observing gains are changed.
The sky, baseline taper, capture time, and held-out split match the corrected
centre-bin comparison. Fine-bin masks are applied before fitting.
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils import iers
from lambda_commissioning.interferometers import RadioArray
from lambda_commissioning.modelling import _aep_interp, calc_lmn
from lambda_commissioning.point_calibration import metrics, solve
from scipy.optimize import least_squares

C = 299792458.0


def phase_slope(freq, gains, valid, reference=0):
    """Robust phase slope within one coarse channel; return seconds/antenna."""
    nfreq, nant = gains.shape
    out = np.full(nant, np.nan)
    x = (freq - np.mean(freq)) * 1e-6
    for ant in range(nant):
        if ant == reference:
            out[ant] = 0.0
            continue
        keep = valid & np.isfinite(gains[:, ant]) & (np.abs(gains[:, ant]) > 0)
        if keep.sum() < 12:
            continue
        phase = np.unwrap(np.angle(gains[keep, ant] / gains[keep, reference]))
        design = np.stack([np.ones(keep.sum()), x[keep]], axis=1)
        start = np.linalg.lstsq(design, phase, rcond=None)[0]
        fit = least_squares(lambda p: np.angle(np.exp(1j * (phase - design @ p))),
                            start, loss="soft_l1", f_scale=.4, max_nfev=50)
        out[ant] = fit.x[1] / (2 * np.pi * 1e6)
    return out


def make_geometry(config, catalogue, ids, times_mjd):
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    loc = config["array_location"]
    site = EarthLocation.from_geodetic(loc["longitude_deg"] * u.deg,
                                       loc["latitude_deg"] * u.deg,
                                       loc["height_m"] * u.m)
    pos = np.array([[v[axis] for axis in ("east", "north", "up")]
                    for ant in ids
                    for fpga in config["antenna_positions"].values()
                    for key, v in fpga.items() if int(key) == ant])
    array = RadioArray(eastNorthHeight=pos, arrayLat=loc["latitude_deg"],
                       arrayLon=loc["longitude_deg"], antIDs=ids)
    array.enh2xyz()
    array.calc_baseline_matrix()
    components = [c for s in catalogue["sources"] for c in s["components"]]
    sky = SkyCoord([c["ra_deg"] for c in components],
                   [c["dec_deg"] for c in components], unit="deg")
    aa = sky[None, :].transform_to(AltAz(obstime=Time(times_mjd, format="mjd")[:, None],
                                       location=site))
    l, m, n = calc_lmn(aa.alt.deg, aa.az.deg)
    geo = (l[:, :, None] * array.uu_m[0] +
           m[:, :, None] * array.vv_m[0] +
           (n[:, :, None] - 1) * array.ww_m[0])
    return pos, components, aa, l, m, geo


def point_model(freq, coarse, components, aa, l, m, geo):
    logf = np.log(freq / 200e6)
    flux = np.array([c["s200_jy"] * np.exp(c["alpha"] * logf +
                   c["beta"] * logf**2) for c in components]).T
    beam = _aep_interp(coarse, "YY")(np.stack([m, l], axis=-1))
    apparent = beam[:, None, :] * flux[None, :, :] * (aa.alt.deg[:, None, :] > 0)
    steering = np.exp(2j * np.pi * freq[None, :, None, None] / C * geo[:, None])
    return np.einsum("tfs,tfsi,tfsj->tfij", apparent, steering,
                     steering.conj(), optimize=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--train-times", type=int, default=32)
    p.add_argument("--held-times", type=int, default=16)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.report.read_text())
    run = args.report.parent.parent
    config = json.loads((run / "starspin_config.json").read_text())
    catalogue = json.loads((run / "point_sources.json").read_text())
    with np.load(args.report.parent / "candidates.npz") as z:
        centre = {k: z[k] for k in z.files if k.startswith("gains_A_multisource_amplitude_YY_")}
        ids = z["antenna_ids"].astype(int)
        used_mjd = z["times_mjd"][:]
        train_w = z["train_weights"][:]
        held_w = z["held_weights"][:]
        base_w = z["baseline_weights"][:]
    with np.load(run / "flags_corrected_v2/transient_mask.npz") as z:
        mask = z["good"][:].astype(bool)
        frequency = z["frequency_hz"][:].astype(float)
    split = len(report["train_integrations"])
    choose_train = np.linspace(0, split - 1, args.train_times, dtype=int)
    choose_held = np.linspace(split, len(used_mjd) - 1, args.held_times, dtype=int)
    selected = np.r_[choose_train, choose_held]
    integration = np.array(report["used_integrations"])[selected]
    times_mjd = used_mjd[selected]
    ntrain = len(choose_train)
    pos, components, aa, l, m, geo = make_geometry(
        config, catalogue, ids, times_mjd)
    with h5py.File(report["capture"]) as f:
        pairs_all = f["baseline_antenna_ids"][:].astype(int)
        keep = np.all(np.isin(pairs_all, ids), axis=1)
        pairs = pairs_all[keep]
        index = {int(ant): i for i, ant in enumerate(ids)}
        ai = np.array([index[a] for a in pairs[:, 0]])
        bi = np.array([index[b] for b in pairs[:, 1]])
        observed = np.empty((len(integration), len(frequency), len(pairs)), np.complex64)
        dataset = f["visibilities"]
        for i, t in enumerate(integration):
            v = dataset[int(t), :, :, 0, 0, :][:, keep]
            observed[i] = v[..., 0] + 1j * v[..., 1]
            if i % 8 == 0:
                print("read", i + 1, "/", len(integration), flush=True)
    nant = len(ids)
    gains = np.full((len(frequency), nant), np.nan + 0j, np.complex64)
    fine_scores = np.full((len(frequency), 2), np.nan)
    retained = np.zeros(len(frequency), bool)
    coarse_delay = np.full((40, nant), np.nan)
    # Cross-baseline matrices, no autos. The same 20% held-baseline split as
    # the centre-bin run is preserved for every fine channel.
    for coarse in range(40):
        sl = slice(coarse * 27, (coarse + 1) * 27)
        freq = frequency[sl]
        model = point_model(freq, 400 + coarse, components, aa, l, m, geo)
        data = np.zeros_like(model)
        data[:, :, ai, bi] = observed[:, sl]
        data[:, :, bi, ai] = observed[:, sl].conj()
        time_good = mask[integration, sl]
        ckey = f"gains_A_multisource_amplitude_YY_{coarse}"
        initial = centre.get(ckey)
        for local in range(27):
            fi = coarse * 27 + local
            if time_good[:ntrain, local].sum() < 20 or time_good[ntrain:, local].sum() < 10:
                continue
            w = time_good[:, local, None, None]
            try:
                g = solve(data[:ntrain, local], model[:ntrain, local],
                          w[:ntrain] * train_w, initial=initial,
                          iterations=150)
            except ValueError:
                continue
            gains[fi] = g
            prediction = model[:, local] * g[None, :, None] * g.conj()[None, None, :]
            fine_scores[fi, 0] = metrics(data[ntrain:, local], prediction[ntrain:],
                                         w[ntrain:] * base_w)["correlation"]
            fine_scores[fi, 1] = metrics(data[:ntrain, local], prediction[:ntrain],
                                         w[:ntrain] * held_w)["correlation"]
            retained[fi] = True
        # Select slope-fit bins using only the training-time mask. The held-out
        # scores above are diagnostics and never enter the delay estimate.
        good = retained[sl]
        coarse_delay[coarse] = phase_slope(freq, gains[sl], good)
        print("fit coarse", 400 + coarse, "usable bins", int(good.sum()), flush=True)
    count = np.isfinite(coarse_delay).sum(axis=0)
    delay = np.nanmedian(coarse_delay, axis=0)
    spread = 1.4826 * np.nanmedian(np.abs(coarse_delay - delay), axis=0)
    # A delay is only reported if a majority of independent coarse bands
    # support it; the dispersion is empirical rather than formal uncertainty.
    delay[count < 20] = np.nan
    delay_scores = np.full((len(frequency), 2, 2), np.nan)
    # Compare the centre-bin gains carried unchanged across each coarse band
    # with the SAME gains plus one global per-antenna phase slope. The last
    # axis of scores is [held time, held baselines]; the middle axis is
    # [unchanged, delay corrected]. Neither candidate is refit to held data.
    for coarse in range(40):
        ckey = f"gains_A_multisource_amplitude_YY_{coarse}"
        if ckey not in centre or not np.all(np.isfinite(delay)):
            continue
        sl = slice(coarse * 27, (coarse + 1) * 27)
        freq = frequency[sl]
        model = point_model(freq, 400 + coarse, components, aa, l, m, geo)
        data = np.zeros_like(model)
        data[:, :, ai, bi] = observed[:, sl]
        data[:, :, bi, ai] = observed[:, sl].conj()
        centre_freq = report["frequencies_hz"][coarse]
        for local in range(27):
            fi = coarse * 27 + local
            if not retained[fi]:
                continue
            w = mask[integration, fi, None, None]
            for trial, g in enumerate((centre[ckey],
                                       centre[ckey] * np.exp(
                                           2j * np.pi * (freq[local] - centre_freq) * delay))):
                prediction = model[:, local] * g[None, :, None] * g.conj()[None, None, :]
                delay_scores[fi, trial, 0] = metrics(
                    data[ntrain:, local], prediction[ntrain:],
                    w[ntrain:] * base_w)["correlation"]
                delay_scores[fi, trial, 1] = metrics(
                    data[:ntrain, local], prediction[:ntrain],
                    w[:ntrain] * held_w)["correlation"]
    validation_medians = np.nanmedian(delay_scores, axis=0)
    result = dict(status="offline YY candidate; no hardware or science gains changed",
                  capture=report["capture"], input_report=str(args.report),
                  n_train_times=ntrain, n_held_times=len(integration)-ntrain,
                  antenna_ids=ids.tolist(), per_antenna_delay_ns=(delay*1e9).tolist(),
                  per_antenna_delay_mad_ns=(spread*1e9).tolist(),
                  supporting_coarse_bands=count.tolist(),
                  fitted_fine_bins=int(retained.sum()),
                  median_held_time_correlation=float(np.nanmedian(fine_scores[:, 0])),
                  median_held_baseline_correlation=float(np.nanmedian(fine_scores[:, 1])),
                  delay_validation={
                      "unchanged_centre_gain_held_time": float(validation_medians[0, 0]),
                      "with_delay_held_time": float(validation_medians[1, 0]),
                      "unchanged_centre_gain_held_baselines": float(validation_medians[0, 1]),
                      "with_delay_held_baselines": float(validation_medians[1, 1]),
                      "note": "One global per-antenna delay from training fine bins; centre gains fixed; held times and baselines were not used to fit the delay."})
    np.savez_compressed(args.output / "fine_yy_candidates.npz",
                        frequency_hz=frequency, antenna_ids=ids,
                        gains=gains, held_scores=fine_scores,
                        coarse_delay_s=coarse_delay, delay_s=delay,
                        delay_mad_s=spread, fitted=retained,
                        delay_validation_scores=delay_scores)
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].plot(frequency / 1e6, fine_scores[:, 0], ".", ms=2,
                 label="Held time")
    axes[0].plot(frequency / 1e6, fine_scores[:, 1], ".", ms=2,
                 label="Held baselines")
    axes[0].set(ylabel="Correlation with full point-source sky", ylim=(-.3, 1.05))
    axes[0].legend()
    axes[1].plot(np.arange(400, 440), coarse_delay * 1e9, ".", ms=3)
    outliers = int(np.sum(np.abs(coarse_delay * 1e9) > 1000))
    axes[1].set(xlabel="Coarse channel", ylabel="Per-antenna phase slope (ns)",
                ylim=(-250, 1000), title=f"Central range; {outliers} outliers beyond ±1000 ns clipped")
    for ax in axes:
        ax.grid(alpha=.25)
    fig.savefig(args.output / "fine_yy_delays.png", dpi=150)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

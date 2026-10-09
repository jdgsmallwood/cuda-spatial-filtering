#!/usr/bin/env python3
"""Fit stable antenna gains to moving sky plus an additive stationary term.

Training means are removed from both data and sky model. This eliminates a
constant per-baseline nuisance without attributing it to a clock drift. Fit
gains and choose a solar flux grid point on training data only; evaluate the
frozen gains and frozen additive term on the chronological second half.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from astropy.coordinates import AltAz, SkyCoord, EarthLocation, get_sun
from astropy.time import Time
from astropy import units as u
from astropy.utils import iers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import j1
from lambda_commissioning.modelling import _aep_interp
from lambda_commissioning.point_calibration import solve


def fit_with_stationary(data, model, training, weight):
    data_mean = data[training].mean(axis=0)
    model_mean = model[training].mean(axis=0)
    gain = solve(data[training] - data_mean, model[training] - model_mean,
                 weight[None], iterations=500)
    product = gain[:, None] * gain[None].conj()
    stationary = data_mean - product * model_mean
    prediction = stationary[None] + model * product[None]
    return gain, stationary, prediction


def directions(coordinate, frame):
    sky = coordinate.transform_to(frame)
    direction = np.stack((np.cos(sky.alt.rad) * np.sin(sky.az.rad),
        np.cos(sky.alt.rad) * np.cos(sky.az.rad), np.sin(sky.alt.rad)), axis=-1)
    return direction, sky.alt.deg > 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--config", type=Path, default=Path("config.json"))
    p.add_argument("--catalogue", type=Path, default=Path(
        "/data/BLACKMESA_1/sma075/lambda-commissioning/runs/rfi_cal_400_439_20261003/point_sources.json"))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    with np.load(a.cache) as z:
        # Centre bin only in this first model comparison; final channel masks
        # and any exported gains must be evaluated on every retained fine bin.
        fidx = np.arange(1, len(z["frequency_hz"]), 3)
        raw = z["raw"][:, fidx].astype(complex)
        freq, ids, pos = z["frequency_hz"][fidx], z["antenna_ids"], z["positions_m"]
        utc, training = z["utc"], z["training"].astype(bool)
    config = json.loads(a.config.read_text())
    catalogue = json.loads(a.catalogue.read_text())
    components = [dict(c, source=s["name"]) for s in catalogue["sources"] for c in s["components"]]
    site = config["array_location"]
    location = EarthLocation.from_geodetic(site["longitude_deg"] * u.deg,
        site["latitude_deg"] * u.deg, site["height_m"] * u.m)
    times = Time(utc, format="unix")
    sun_dir, sun_up = directions(get_sun(times), AltAz(obstime=times, location=location))
    cat_dir, cat_up = directions(SkyCoord([c["ra_deg"] for c in components],
        [c["dec_deg"] for c in components], unit="deg")[None],
        AltAz(obstime=times[:, None], location=location))
    geo_sun = sun_dir @ pos.T
    geo_cat = cat_dir @ pos.T
    baseline = pos[:, None] - pos[None]
    length = np.linalg.norm(baseline, axis=-1)
    weights = (length >= 5).astype(float)
    upper = np.triu(weights > 0, 1)
    # Reserve 20% of baselines for selecting among solar-flux candidates.
    pairs = np.argwhere(upper)
    np.random.default_rng(20261004).shuffle(pairs)
    validation = np.zeros_like(upper)
    ii, jj = pairs[:len(pairs) // 5].T
    validation[ii, jj] = True
    fit_weights = weights * ~(validation | validation.T)
    solar_fluxes = [1e4, 3e4, 1e5, 3e5, 1e6]
    names = ("Sun_only", "Sun_plus_catalogue")
    gains = np.zeros((len(freq), 2, 2, len(ids)), complex)
    saved_stationary = np.zeros((len(freq), 2, 2, len(ids), len(ids)), complex)
    fitted_flux = np.zeros((len(freq), 2, 2))
    results = []
    for fi, frequency in enumerate(freq):
        k = 2 * np.pi * frequency / 299792458.
        sun_a = np.exp(1j * k * geo_sun)
        sun = sun_a[:, :, None] * sun_a[:, None, :].conj()
        radial = np.einsum("td,ijd->tij", sun_dir, baseline)
        projected = np.sqrt(np.maximum(0, length[None] ** 2 - radial ** 2))
        x = k * np.deg2rad(.27) * projected
        disk = np.divide(2 * j1(x), x, out=np.ones_like(x), where=x != 0)
        sun *= disk
        cat_a = np.exp(1j * k * geo_cat)
        fluxes = np.array([c["s200_jy"] * np.exp(c["alpha"] * np.log(frequency / 200e6) +
            c.get("beta", 0) * np.log(frequency / 200e6) ** 2) for c in components])
        for pi, pol in enumerate(("XX", "YY")):
            data = raw[:, fi, pi]
            norm = np.sqrt(np.median(data[training].diagonal(axis1=-2, axis2=-1).real, axis=0))
            data = data / (norm[:, None] * norm[None])
            beam = _aep_interp(int(round(frequency / 781250.)), pol)
            sun_beam = beam(sun_dir[:, [1, 0]]) * sun_up
            cat_beam = beam(cat_dir[..., [1, 0]]) * cat_up
            sun_model = sun * sun_beam[:, None, None]
            cat_model = np.einsum("ts,tsi,tsj->tij", cat_beam * fluxes[None], cat_a,
                                   cat_a.conj(), optimize=True)
            static = data[training].mean(axis=0)
            static_error = np.mean(abs(data[~training][:, upper] - static[upper]) ** 2)
            for ni, name in enumerate(names):
                candidates = []
                flux_grid = [1e5] if name == "Sun_only" else solar_fluxes
                for sf in flux_grid:
                    model = sf * (frequency / 330e6) ** 2 * sun_model
                    if name == "Sun_plus_catalogue":
                        model = model + cat_model
                    gain, stationary, prediction = fit_with_stationary(data, model, training, fit_weights)
                    # Nuisance means use early data on every baseline; only
                    # the variable sky component is withheld on these pairs.
                    select_error = np.mean(abs((data - prediction)[training][:, validation]) ** 2)
                    candidates.append((select_error, sf, gain, stationary, prediction))
                select_error, sf, gain, stationary, prediction = min(candidates, key=lambda item: item[0])
                error = np.mean(abs((data - prediction)[~training][:, upper]) ** 2)
                gains[fi, pi, ni] = gain
                saved_stationary[fi, pi, ni] = stationary
                fitted_flux[fi, pi, ni] = sf
                row = dict(frequency_hz=float(frequency), polarization=pol, model=name,
                    solar_flux_330_jy=sf, train_withheld_variable_component_rms=float(np.sqrt(select_error)),
                    held_error_improvement_over_stationary=float(1 - error / static_error),
                    held_cross_residual_rms=float(np.sqrt(error)),
                    region="solar_control" if frequency < 324e6 else "stationary_region")
                results.append(row)
        if fi % 5 == 0:
            print("joint fit", fi + 1, "/", len(freq), flush=True)
    summary = {}
    for pol in ("XX", "YY"):
        summary[pol] = {}
        for region in ("solar_control", "stationary_region"):
            summary[pol][region] = {}
            for name in names:
                rows = [r for r in results if r["polarization"] == pol and r["region"] == region and r["model"] == name]
                summary[pol][region][name] = dict(channels=len(rows),
                    median_held_error_improvement=float(np.median([r["held_error_improvement_over_stationary"] for r in rows])),
                    median_held_cross_rms=float(np.median([r["held_cross_residual_rms"] for r in rows])))
    report = dict(input_cache=str(a.cache), summary=summary, rows=results,
        model="D(t)=diag(g) [Sun(t)+catalogue(t)] diag(g)^H + constant covariance",
        training_samples=int(training.sum()), held_samples=int((~training).sum()),
        min_baseline_m=5, solar_radius_deg=.27, catalogue=str(a.catalogue),
        method="Stable gains from mean-subtracted early data/model; choose solar flux on early withheld variable baseline components; freeze gains and additive covariance on later data.",
        limitations=["Antenna input mapping is exactly the captured map, including its 8/12 swaps.",
            "Only one central fine bin per coarse channel is tested here; these are diagnostic gains, not a full-band calibration export.",
            "Additive nuisance means on withheld baseline pairs still use their early measurements.",
            "No time-varying gain or clock drift was fitted; variation of stationary emission remains in the residual.",
            "Absolute solar flux grid depends on an unverified direction-dependent beam; fitted values do not establish absolute flux calibration.",
            "Catalogue uses compact approximations to extended sources and omits diffuse Galactic emission."])
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(a.output / "diagnostic_gains.npz", frequency_hz=freq,
        antenna_ids=ids, gains=gains, stationary=saved_stationary,
        models=np.array(names), solar_flux_330_jy=fitted_flux)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for pi, pol in enumerate(("XX", "YY")):
        for name in names:
            rows = [r for r in results if r["polarization"] == pol and r["model"] == name]
            axs[pi].plot([r["frequency_hz"] / 1e6 for r in rows],
                         [r["held_error_improvement_over_stationary"] for r in rows], ".-", label=name)
        axs[pi].set(title=pol + ": stable gains + stationary covariance",
            xlabel="MHz", ylabel="Held-time error improvement over stationary", ylim=(-1, 1))
        axs[pi].axhline(0, color="0.5", lw=.8)
        axs[pi].axvline(324, color="0.5", ls=":")
        axs[pi].legend()
        axs[pi].grid(alpha=.25)
    fig.savefig(a.output / "joint_sky_stationary.png", dpi=160)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

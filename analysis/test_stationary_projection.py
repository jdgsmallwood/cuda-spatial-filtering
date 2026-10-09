#!/usr/bin/env python3
"""Offline fixed spatial-mode experiment; input visibilities are never modified.

Estimate one mode from the first chronological half, with a fixed per-input
normalization. Compare held data before/after projection, including its change
to a fitted moving-Sun model. The lower solar band is a deliberate failure
control, not a recommendation to filter it. Per-baseline temporal fits are
fringe diagnostics, not an antenna gain solution or a source detection.
"""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy import units as u
from astropy.time import Time
from astropy.utils import iers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate_native_sky import packet_clock_times


def project(matrix, vector):
    """P R P^H, where P=I-uu^H and u is a unit column vector."""
    p = np.eye(len(vector)) - np.outer(vector, vector.conj())
    return p @ matrix @ p


def fixed_mode(training):
    mean = np.mean(training, axis=0)
    eig, vec = np.linalg.eigh((mean + mean.conj().T) / 2)
    return vec[:, -1], eig


def subtract_covariance_component(matrix, vector, mask):
    """Remove one real covariance coefficient, without removing white noise.

    The spatial vector and baseline mask are fixed from training/geometry.
    Each matrix supplies one real amplitude. Applying this same linear
    operator to a sky prediction measures the associated signal loss.
    Autocorrelations do not enter the amplitude fit.
    """
    mode = np.outer(vector, vector.conj())
    denominator = np.sum(abs(mode[mask]) ** 2)
    if denominator <= 0:
        raise ValueError("Mode has no support on selected cross-correlations")
    amplitude = np.sum((matrix[..., mask] * mode[mask].conj()).real, axis=-1) / denominator
    return matrix - amplitude[..., None, None] * mode, amplitude


def fit_motion(data, phasor, train):
    """Fit a complex constant plus a moving term independently per baseline.

    Centred least squares avoids dividing by a vanishing fringe change.
    Coefficients use training samples only. Low-fringe baselines get zero
    moving coefficient and are counted separately by the caller.
    """
    f = phasor[train]
    d = data[train]
    fc, dc = f.mean(axis=0), d.mean(axis=0)
    variance = np.mean(abs(f - fc) ** 2, axis=0)
    coeff = np.divide(np.mean((f - fc).conj() * (d - dc), axis=0),
                      variance, out=np.zeros_like(dc), where=variance > .01)
    stationary = dc - coeff * fc
    return stationary[None] + coeff[None] * phasor, coeff, variance


def read_capture(path, config, timing, stride):
    site = config["array_location"]
    location = EarthLocation.from_geodetic(site["longitude_deg"] * u.deg,
        site["latitude_deg"] * u.deg, site["height_m"] * u.m)
    all_times = packet_clock_times(path, timing, location)
    with h5py.File(path) as h:
        n = len(h["visibilities"])
        missing = h["vis_missing_nums"][:]
        # The stored third column is a percentage; select on missing count.
        ids = h["antenna_ids"][:]
        ids = ids[(ids > 0) & (ids != 14)]
        pairs = h["baseline_antenna_ids"][:]
        keep = np.isin(pairs, ids).all(axis=1)
        ai = {int(a): i for i, a in enumerate(ids)}
        ia, ib = np.array([[ai[int(a)], ai[int(b)]] for a, b in pairs[keep]]).T
        per = int(h.attrs["fine_channels_per_coarse"])
        first = int(h.attrs["first_raw_fine_bin"])
        if per != 27 or first != 3:
            raise ValueError("This experiment requires the audited unique 27-bin axis")
        nc = int(h.attrs["max_channel"]) - int(h.attrs["min_channel"]) + 1
        channels = (np.arange(nc)[:, None] * per + np.array([6, 13, 20])).ravel()
        freq = (int(h.attrs["min_channel"]) + channels // per) * 781250. + \
            (channels % per + first - 16) * 781250. / 27
        indices = np.arange(3, n - 3, stride)
        indices = indices[missing[indices, 0] == 0]
        if len(indices) < 50:
            raise ValueError("Too few zero-loss integrations")
        value = np.zeros((len(indices), len(channels), 2, len(ids), len(ids)),
                         np.complex64)
        where = {int(t): i for i, t in enumerate(indices)}
        chunk = h["visibilities"].chunks[0]
        for start in range(0, n, chunk):
            wanted = [t for t in range(start, min(n, start + chunk)) if t in where]
            if not wanted:
                continue
            block = h["visibilities"][start:min(n, start + chunk)]
            for ti in wanted:
                v = block[ti - start][channels][:, keep]
                for pi, hand in enumerate((1, 0)):
                    z = v[:, :, hand, hand, 0] + 1j * v[:, :, hand, hand, 1]
                    value[where[ti], :, pi, ia, ib] = z.T
                    value[where[ti], :, pi, ib, ia] = z.conj().T
            if start % (chunk * 10) == 0:
                print("read", start, "/", n, flush=True)
        manifest = json.loads(h["audit/run_manifest_json"][()])
    return value, freq, ids, indices, all_times[indices], location, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--config", type=Path, default=Path("config.json"))
    p.add_argument("--timing-pcap", type=Path, required=True)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--reuse-selected", type=Path,
                   help="Reuse the selected-covariance cache from this exact capture")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    config = json.loads(a.config.read_text())
    if a.reuse_selected:
        provenance = json.loads(a.reuse_selected.with_name("report.json").read_text())
        if Path(provenance["capture"]).resolve() != a.capture.resolve():
            p.error("Selected-data cache belongs to another capture")
        with np.load(a.reuse_selected) as z:
            raw, freq, ids, indices = z["raw"], z["frequency_hz"], z["antenna_ids"], z["integrations"]
            times = Time(z["utc"], format="unix")
        site = config["array_location"]
        location = EarthLocation.from_geodetic(site["longitude_deg"] * u.deg,
            site["latitude_deg"] * u.deg, site["height_m"] * u.m)
        with h5py.File(a.capture) as h:
            manifest = json.loads(h["audit/run_manifest_json"][()])
    else:
        raw, freq, ids, indices, times, location, manifest = read_capture(
            a.capture, config, a.timing_pcap, a.stride)
    positions = {int(i): [v["east"], v["north"], v["up"]]
        for board in config["antenna_positions"].values() for i, v in board.items()}
    pos = np.array([positions[int(i)] for i in ids])
    pos -= pos[0]
    baselines = pos[:, None] - pos[None]
    use = np.triu(np.linalg.norm(baselines, axis=-1) >= 5, 1)
    train = times.unix < (times.unix[0] + times.unix[-1]) / 2
    held = ~train
    frame = AltAz(obstime=times, location=location)
    directions = {}
    altitudes = {}
    sources = {"Sun": get_sun(times), "Fornax_A": SkyCoord(50.6738, -37.2083, unit="deg"),
               "Pictor_A": SkyCoord(79.9571, -45.7789, unit="deg")}
    for name, source in sources.items():
        sky = source.transform_to(frame)
        directions[name] = np.column_stack((np.cos(sky.alt.rad) * np.sin(sky.az.rad),
            np.cos(sky.alt.rad) * np.cos(sky.az.rad), np.sin(sky.alt.rad)))
        altitudes[name] = [float(sky.alt.deg[0]), float(sky.alt.deg[-1])]
    summary = []
    examples = {}
    for fi, frequency in enumerate(freq):
        # Phase sign matches the native E_i conj(E_j) solar predictor. Constant
        # per-baseline phases are absorbed by the training-only coefficients.
        phases = {}
        for name, direction in directions.items():
            geo = direction @ pos.T
            phase = np.exp(2j * np.pi * frequency / 299792458. * geo)
            phases[name] = phase[:, :, None] * phase[:, None, :].conj()
        for pi, pol in enumerate(("XX", "YY")):
            r = raw[:, fi, pi].astype(complex)
            auto = r.diagonal(axis1=-2, axis2=-1).real
            scale = np.sqrt(np.median(auto[train], axis=0))
            if np.any(~np.isfinite(scale) | (scale <= 0)):
                continue
            r /= scale[None, :, None] * scale[None, None, :]
            if not np.isfinite(r).all():
                continue
            uvec, eigenvalues = fixed_mode(r[train])
            rr = project(r, uvec)
            excess, amplitude = subtract_covariance_component(r, uvec, use)
            frozen = r - amplitude[train].mean() * np.outer(uvec, uvec.conj())
            # The actual spatial projection also projects thermal noise. Use
            # its identical action on every prediction, including the constant.
            static = np.broadcast_to(r[train].mean(axis=0), r.shape)
            row = dict(frequency_hz=float(frequency), polarization=pol,
                region="solar_control" if frequency < 324e6 else "stationary_region",
                leading_eigenvalue=float(eigenvalues[-1]),
                leading_trace_fraction=float(eigenvalues[-1] / eigenvalues.sum()),
                participation=float(1 / np.sum(abs(uvec) ** 4)),
                held_cross_rms_before=float(np.sqrt(np.mean(abs(r[held][:, use]) ** 2))),
                held_cross_rms_after=float(np.sqrt(np.mean(abs(rr[held][:, use]) ** 2))),
                held_cross_rms_frozen_subtraction=float(np.sqrt(np.mean(abs(frozen[held][:, use]) ** 2))),
                held_cross_rms_component_subtraction=float(np.sqrt(np.mean(abs(excess[held][:, use]) ** 2))),
                component_amplitude_train_median=float(np.median(amplitude[train])),
                component_amplitude_held_median=float(np.median(amplitude[held])),
                held_total_power_retained=float(np.trace(rr[held], axis1=-2, axis2=-1).real.sum() /
                                                np.trace(r[held], axis1=-2, axis2=-1).real.sum()),
                motion={})
            for name, phase in phases.items():
                pred, coeff, variance = fit_motion(r, phase, train)
                valid = use & (variance > .01)
                pclean = project(pred, uvec)
                staticclean = project(static, uvec)
                def error(data, prediction):
                    return float(np.mean(abs((data - prediction)[held][:, valid]) ** 2))
                e0, e1 = error(r, static), error(r, pred)
                ep0, ep1 = error(rr, staticclean), error(rr, pclean)
                moving = coeff[None] * phase
                movingclean = project(moving, uvec)
                moving_subtracted, _ = subtract_covariance_component(moving, uvec, use)
                row["motion"][name] = dict(baselines=int(valid.sum()),
                    raw_improvement_over_stationary=1 - e1 / e0,
                    projected_improvement_over_stationary=1 - ep1 / ep0,
                    held_fit_cross_rms=np.sqrt(e1),
                    held_projected_fit_cross_rms=np.sqrt(ep1),
                    fitted_moving_covariance_power_retained=float(
                        np.sum(abs(movingclean[held]) ** 2) / np.sum(abs(moving[held]) ** 2)),
                    component_subtraction_moving_cross_power_retained=float(
                        np.sum(abs(moving_subtracted[held][:, use]) ** 2) /
                        np.sum(abs(moving[held][:, use]) ** 2)))
            # Separate fit of reversed solar motion is a deliberately incorrect
            # motion control with the same number of free coefficients.
            reverse, _, variance = fit_motion(r, phases["Sun"].conj(), train)
            mask = use & (variance > .01)
            reverse_error = np.mean(abs((r - reverse)[held][:, mask]) ** 2)
            stationary_error = np.mean(abs((r - static)[held][:, mask]) ** 2)
            row["reverse_sun_improvement_over_stationary"] = float(1 - reverse_error / stationary_error)
            summary.append(row)
            if pi == 1 and fi in (13, 91):
                examples[str(fi)] = dict(raw=r, projected=rr, mode=uvec, frequency=frequency)
        if fi % 15 == 0:
            print("fit", fi + 1, "/", len(freq), flush=True)
    medians = {}
    for pol in ("XX", "YY"):
        medians[pol] = {}
        for region in ("solar_control", "stationary_region"):
            rows = [r for r in summary if r["polarization"] == pol and r["region"] == region]
            medians[pol][region] = dict(channels=len(rows),
                cross_rms_ratio=float(np.median([r["held_cross_rms_after"] / r["held_cross_rms_before"] for r in rows])),
                frozen_subtraction_cross_rms_ratio=float(np.median([r["held_cross_rms_frozen_subtraction"] / r["held_cross_rms_before"] for r in rows])),
                component_subtraction_cross_rms_ratio=float(np.median([r["held_cross_rms_component_subtraction"] / r["held_cross_rms_before"] for r in rows])),
                leading_trace_fraction=float(np.median([r["leading_trace_fraction"] for r in rows])),
                total_power_retained=float(np.median([r["held_total_power_retained"] for r in rows])),
                motion={name: {k: float(np.median([r["motion"][name][k] for r in rows]))
                    for k in ("raw_improvement_over_stationary", "projected_improvement_over_stationary",
                              "fitted_moving_covariance_power_retained",
                              "component_subtraction_moving_cross_power_retained")}
                    for name in phases})
    report = dict(capture=str(a.capture), mode_fit="first chronological half, fixed thereafter",
        training_integrations=int(train.sum()), held_integrations=int(held.sum()),
        missing_packet_gate="zero missing samples in every used integration",
        min_baseline_m=5, excluded_antennas=[14], altitudes_start_end_deg=altitudes,
        input_mapping_8_12=[r for r in manifest["forward_mapping"] if r["antenna_id"] in (8, 12)],
        offline_polarization_swaps=[], medians=medians, rows=summary,
        limitations=["Exploratory frequency regions were identified from previous inspection of this capture.",
            "Projector fits no held-time spatial direction, but removes a spatial degree of freedom and changes the sky and noise response.",
            "Covariance-component subtraction fits one real amplitude per held integration on a fixed spatial mode; its reported sky response uses the same operator. It does not guarantee a positive semidefinite residual covariance.",
            "Moving terms have one independent complex amplitude per baseline, not antenna gains; they test fringe evolution only.",
            "The fitted moving-term transfer is conditional on that temporal model, not a measured absolute flux-recovery calibration.",
            "A smaller residual is not sufficient evidence of better astronomical sensitivity or a point-source detection.",
            "The Sun point model neglects source structure and time-dependent beam amplitudes."])
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not a.reuse_selected or a.reuse_selected.resolve() != (a.output / "selected_covariances.npz").resolve():
        np.savez_compressed(a.output / "selected_covariances.npz", raw=raw,
            frequency_hz=freq, antenna_ids=ids, integrations=indices,
            utc=times.unix, training=train, positions_m=pos,
            **{name + "_direction": direction for name, direction in directions.items()})
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for pi, pol in enumerate(("XX", "YY")):
        rows = [r for r in summary if r["polarization"] == pol]
        f = np.array([r["frequency_hz"] for r in rows]) / 1e6
        for key, label in (("held_cross_rms_before", "Original"),
                ("held_cross_rms_after", "Spatial projection"),
                ("held_cross_rms_component_subtraction", "Covariance-mode subtraction")):
            axs[pi, 0].plot(f, [r[key] for r in rows], label=label)
        axs[pi, 0].set(yscale="log", title=pol + ": held-time cross-correlation amplitude",
                       ylabel="RMS / training auto power")
        for name in phases:
            axs[pi, 1].plot(f, [r["motion"][name]["raw_improvement_over_stationary"] for r in rows], label=name)
        axs[pi, 1].plot(f, [r["reverse_sun_improvement_over_stationary"] for r in rows], label="Reversed Sun", ls=":")
        axs[pi, 1].set(title=pol + ": held-time motion test (original data)",
            ylabel="Fractional error improvement over stationary", ylim=(-2, 1))
        for ax in axs[pi]:
            ax.axvline(324, color="0.5", ls=":")
            ax.set_xlabel("MHz")
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.savefig(a.output / "stationary_projection.png", dpi=150)
    print(json.dumps(medians, indent=2), flush=True)


if __name__ == "__main__":
    main()

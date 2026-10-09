#!/usr/bin/env python3
"""All-sky dirty images from every validated fine channel in each hand.

The Sun-centered CLEAN product is made separately by image_solar_native_clean.py.
This widefield diagnostic uses w-stacked bilinear UV gridding; it is not a
primary-beam-corrected flux map. Gain phases are fitted on the first time
window and the image uses the independent late window.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time
from astropy.utils import iers

from lambda_commissioning.utils import make_telescope_model
from lambda_commissioning.vistools import BRIGHT_SOURCE_CATALOG

C = 299792458.0


def grid_image(uu, vv, ww, vis, n=256, uv_cell=.5, w_cell=.25):
    """Hermitian w-stacked image with bilinear UV gridding."""
    u_all = np.r_[uu, -uu]
    v_all = np.r_[vv, -vv]
    w_all = np.r_[ww, -ww]
    value = np.r_[vis, vis.conjugate()]
    u_pixel = u_all / uv_cell + n // 2
    v_pixel = v_all / uv_cell + n // 2
    u_floor = np.floor(u_pixel).astype(int)
    v_floor = np.floor(v_pixel).astype(int)
    u_fraction = u_pixel - u_floor
    v_fraction = v_pixel - v_floor
    wi = np.rint(w_all / w_cell).astype(int)
    w_min, w_max = int(wi.min()), int(wi.max())
    stack = w_max - w_min + 1
    grid = np.zeros(stack * n * n, np.complex128)
    psf_grid = np.zeros_like(grid)
    for du in (0, 1):
        ui = u_floor + du
        uw = u_fraction if du else 1. - u_fraction
        for dv in (0, 1):
            vi = v_floor + dv
            weight = uw * (v_fraction if dv else 1. - v_fraction)
            inside = (ui >= 0) & (ui < n) & (vi >= 0) & (vi < n)
            flat = ((wi[inside] - w_min) * n + vi[inside]) * n + ui[inside]
            np.add.at(grid, flat, value[inside] * weight[inside])
            np.add.at(psf_grid, flat, weight[inside])
    grid = grid.reshape(stack, n, n)
    psf_grid = psf_grid.reshape(stack, n, n)
    axis = np.fft.fftshift(np.fft.fftfreq(n, d=uv_cell))
    ll, mm = np.meshgrid(axis, axis)
    visible = ll * ll + mm * mm <= 1.
    nn = np.sqrt(np.maximum(0., 1. - ll * ll - mm * mm))
    image = np.zeros((n, n), np.complex128)
    psf = np.zeros_like(image)
    for k in range(stack):
        phase = np.exp(2j * np.pi * ((k + w_min) * w_cell) * (nn - 1.))
        image += np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid[k]))) * n * n * phase
        psf += np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(psf_grid[k]))) * n * n * phase
    image = image.real / max(2 * len(vis), 1)
    psf = psf.real / max(2 * len(vis), 1)
    image[~visible] = np.nan
    psf[~visible] = np.nan
    return axis, image, psf, int(len(vis)), stack


def direction_cosines(altaz):
    alt, az = np.asarray(altaz.alt.rad), np.asarray(altaz.az.rad)
    return np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az)


def clean_sun(dirty, psf, axis, sun_x, sun_y, radius_deg=4., gain=.1,
              max_iterations=400, stop_fraction=.04):
    """Högbom CLEAN within a Sun-centered mask, using the measured widefield PSF."""
    n = len(axis)
    center = n // 2
    sx = int(np.argmin(abs(axis - sun_x)))
    sy = int(np.argmin(abs(axis - sun_y)))
    xx, yy = np.meshgrid(axis, axis)
    mask = ((xx - sun_x)**2 + (yy - sun_y)**2 <
            np.sin(np.deg2rad(radius_deg))**2)
    residual = np.nan_to_num(dirty, nan=0.).copy()
    kernel = np.nan_to_num(psf, nan=0.)
    components = np.zeros_like(residual)
    initial = float(np.max(abs(residual[mask])))
    count = 0
    for _ in range(max_iterations):
        iy, ix = np.unravel_index(np.argmax(np.where(mask, abs(residual), -1)),
                                  residual.shape)
        peak = residual[iy, ix]
        if abs(peak) < stop_fraction * initial:
            break
        component = gain * peak
        components[iy, ix] += component
        residual -= component * np.roll(kernel, (iy - center, ix - center), axis=(0, 1))
        count += 1
    inner = (xx * xx + yy * yy < np.sin(np.deg2rad(2.))**2) & (kernel > 0)
    sigma = np.sqrt(np.sum(kernel[inner] * (xx[inner]**2 + yy[inner]**2)) /
                    max(2 * np.sum(kernel[inner]), 1e-30))
    sigma_pixels = max(1., sigma / (axis[1] - axis[0]))
    restored = (gaussian_filter(components, sigma_pixels) *
                (2 * np.pi * sigma_pixels**2) + residual)
    visible = xx * xx + yy * yy <= 1.
    for image in (components, residual, restored):
        image[~visible] = np.nan
    return components, residual, restored, count, float(sigma_pixels)


def overlay(ax, frame, sun, plane, sources):
    theta = np.linspace(0, 2 * np.pi, 720)
    for elevation in (0, 30, 60):
        radius = np.cos(np.deg2rad(elevation))
        ax.plot(radius * np.sin(theta), radius * np.cos(theta),
                color="white", lw=.7, alpha=.55, zorder=3)
    px, py = direction_cosines(plane)
    px = np.where(plane.alt.deg > 0, px, np.nan)
    py = np.where(plane.alt.deg > 0, py, np.nan)
    line, = ax.plot(px, py, color="#f5c542", lw=1.8, alpha=.9,
                    zorder=5, label="Galactic plane")
    line.set_path_effects([pe.Stroke(linewidth=3, foreground="#242032"), pe.Normal()])
    sx, sy = direction_cosines(sun)
    ax.plot(sx, sy, marker="x", color="#6dffde", ms=11, mew=2.3, zorder=8)
    ax.annotate("Sun", (sx, sy), xytext=(6, 6), textcoords="offset points",
                color="#6dffde", weight="bold", fontsize=9, zorder=9,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
    for name, source in sources:
        pos = source.transform_to(frame)
        if pos.alt.deg <= 0:
            continue
        x, y = direction_cosines(pos)
        ax.plot(x, y, marker="+", color="white", ms=8, mew=1.5, zorder=7)
        ax.annotate(name, (x, y), xytext=(5, -10), textcoords="offset points",
                    color="white", weight="bold", fontsize=7.5, zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
    ax.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), xlabel="East direction cosine",
           ylabel="North direction cosine")
    ax.set_aspect("equal")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("validation", type=Path)
    ap.add_argument("calibration_report", type=Path)
    ap.add_argument("config", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.validation) as z:
        freq = z["frequencies_hz"]
        good = z["good"]
        data = z["data"][2]
        gain = z["gains"]
        selected = z["selected"]
        order = z["antenna_order"]
        stamp = Time(z["times_jd"][2], format="jd")
    report = json.loads(args.calibration_report.read_text())
    site = json.loads(args.config.read_text())["array_location"]
    location = EarthLocation.from_geodetic(site["longitude_deg"] * u.deg,
                                           site["latitude_deg"] * u.deg,
                                           site["height_m"] * u.m)
    frame = AltAz(obstime=stamp, location=location)
    sun = get_sun(stamp).transform_to(frame)
    galactic = SkyCoord(l=np.arange(0, 360.5, .5) * u.deg,
                        b=np.zeros(721) * u.deg, frame="galactic").transform_to(frame)
    names = ("For A", "Pic A", "Cyg A", "Cas A", "Tau A", "Cen A", "Vir A")
    sources = [(name, BRIGHT_SOURCE_CATALOG[name]) for name in names
               if name in BRIGHT_SOURCE_CATALOG]
    array = make_telescope_model(np.arange(1, 37))
    order_idx = order - 1
    xyz = np.stack((array.uu_m[order_idx][:, order_idx],
                    array.vv_m[order_idx][:, order_idx],
                    array.ww_m[order_idx][:, order_idx]), axis=-1)
    images = {}
    psfs = {}
    restored = {}
    residuals = {}
    components = {}
    details = {}
    for pi, pol in enumerate(("XX", "YY")):
        use = selected.copy()
        excluded = {entry["antenna_id"] for entry in report["excluded_receptors"]
                    if entry["polarization"] == pol}
        for ant in excluded:
            loc = np.flatnonzero(order == ant)
            if len(loc):
                use[loc[0], :] = False
                use[:, loc[0]] = False
        ii, jj = np.where(use)
        channels = np.flatnonzero(good[:, pi])
        if not len(channels) or not len(ii):
            raise ValueError(f"No accepted channels or baselines for {pol}")
        baselines = xyz[ii, jj]
        wave = freq[channels] / C
        uvw = (wave[:, None, None] * baselines[None]).reshape(-1, 3)
        unit = gain[channels, pi] / np.maximum(abs(gain[channels, pi]), 1e-30)
        vis = data[channels, pi][:, ii, jj] * unit[:, ii].conjugate() * unit[:, jj]
        axis, image, psf, nvis, nstack = grid_image(
            uvw[:, 0], uvw[:, 1], uvw[:, 2], vis.ravel())
        images[pol] = image
        psfs[pol] = psf
        sx, sy = direction_cosines(sun)
        component, residual, clean_image, count, sigma_pixels = clean_sun(
            image, psf, axis, float(sx), float(sy))
        components[pol] = component
        residuals[pol] = residual
        restored[pol] = clean_image
        near = (axis[None, :] - sx)**2 + (axis[:, None] - sy)**2 < .08**2
        peak_y, peak_x = np.unravel_index(np.nanargmax(np.where(near, image, np.nan)), image.shape)
        details[pol] = {"accepted_fine_channels": int(len(channels)),
                        "frequency_bounds_mhz": [float(freq[channels[0]] / 1e6),
                                                 float(freq[channels[-1]] / 1e6)],
                        "baselines": int(len(ii)), "visibility_samples": nvis,
                        "w_stacks": nstack,
                        "sun_clean_iterations": count,
                        "clean_beam_sigma_deg": float(np.rad2deg(sigma_pixels *
                                                               (axis[1] - axis[0]))),
                        "residual_rms_outside_sun": float(np.nanstd(
                            np.where(((axis[None, :] - sx)**2 +
                                      (axis[:, None] - sy)**2) > np.sin(np.deg2rad(4.))**2,
                                     residual, np.nan))),
                        "sun_peak_offset_direction_cosine": float(np.hypot(axis[peak_x] - sx,
                                                                             axis[peak_y] - sy))}
    combined = .5 * (images["XX"] / np.nanmax(images["XX"]) +
                     images["YY"] / np.nanmax(images["YY"]))
    combined_restored = .5 * (restored["XX"] / np.nanmax(images["XX"]) +
                              restored["YY"] / np.nanmax(images["YY"]))
    planes = (("XX", images["XX"], restored["XX"]),
              ("YY", images["YY"], restored["YY"]),
              ("XX + YY", combined, combined_restored))
    fig, axs = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
    for col, (name, dirty, clean_image) in enumerate(planes):
        _valid = dirty[np.isfinite(dirty)]
        _scale = max(float(np.percentile(abs(_valid), 80)), 1e-12)
        _reference = np.arcsinh(dirty / _scale)
        _limit = float(np.percentile(abs(_reference[np.isfinite(_reference)]), 99.9))
        for row, (kind, plane) in enumerate((("dirty", dirty),
                                            ("Sun CLEAN restored", clean_image))):
            ax = axs[row, col]
            display = np.arcsinh(plane / _scale)
            im = ax.imshow(display, origin="lower", extent=(-1, 1, -1, 1),
                           cmap="RdBu_r", vmin=-_limit, vmax=_limit)
            overlay(ax, frame, sun, galactic, sources)
            ax.set_title(f"{name} {kind} | " +
                         (f"{details[name]['accepted_fine_channels']} fine channels"
                          if name != "XX + YY" else "both validated hands"))
            fig.colorbar(im, ax=ax, fraction=.046, pad=.03,
                         label="asinh relative correlated brightness")
    fig.suptitle(f"Sun observation, {stamp.isot} UTC | all accepted channels per hand\n"
                 "Galactic plane and bright-source positions overlaid; color is not flux calibrated",
                 fontsize=13)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    np.savez_compressed(args.output.with_suffix(".npz"), axis=axis,
                        dirty_xx=images["XX"], dirty_yy=images["YY"],
                        psf_xx=psfs["XX"], psf_yy=psfs["YY"],
                        restored_xx=restored["XX"], restored_yy=restored["YY"],
                        residual_xx=residuals["XX"], residual_yy=residuals["YY"],
                        components_xx=components["XX"], components_yy=components["YY"],
                        utc=stamp.isot)
    details["method"] = "All validated fine channels in each polarization; first-window phase-only Sun-disk gains applied to the independent late window; w-stacked bilinear uv grid and matched dirty/restored color scale; no primary-beam correction or absolute flux scale"
    details["excluded_receptors"] = report["excluded_receptors"]
    details["utc"] = stamp.isot
    args.output.with_suffix(".json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()

"""Small, side-effect-free helpers for the Starspin calibration viewer."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

C = 299792458.0


def load_report(path):
    return json.loads(Path(path).read_text())


def score_rows(report, pol="YY", stage="amplitude"):
    return [r for r in report["results"] if r["pol"] == pol and r["stage"] == stage]


def selected_frequency(report, index):
    return float(report["frequencies_hz"][index])


def gain_key(method, stage, pol, index):
    return f"gains_{method}_{stage}_{pol}_{index}"


def choose_capture(path, report_path, report):
    supplied = Path(path).expanduser() if path else Path(report["capture"])
    if supplied.is_file():
        return supplied
    sibling = Path(report_path).parent.parent / Path(report["capture"]).name
    return sibling if sibling.is_file() else supplied


def read_visibility_slice(capture_path, integration, channel, pol):
    """Read one integration and one fine channel, never the full capture."""
    with h5py.File(capture_path, "r") as f:
        if not (0 <= integration < len(f["vis_seq_nums"])):
            raise IndexError("integration outside capture")
        if not (0 <= channel < f["visibilities"].shape[1]):
            raise IndexError("fine channel outside capture")
        hand = {"XX": 1, "YY": 0}[pol]
        vis = f["visibilities"][integration, channel, :, hand, hand, :]
        z = vis[:, 0].astype(np.float64) + 1j * vis[:, 1].astype(np.float64)
        pairs = f["baseline_antenna_ids"][:].astype(int)
        missing = int(f["vis_missing_nums"][integration, 0])
    return pairs, z, missing


def read_mean_visibility(capture_path, integrations, channel, pol):
    """Average selected complete integrations at one fine channel."""
    indices = np.asarray(integrations, dtype=int)
    if len(indices) == 0:
        raise ValueError("No integrations selected for image")
    with h5py.File(capture_path, "r") as f:
        hand = {"XX": 1, "YY": 0}[pol]
        pairs = f["baseline_antenna_ids"][:].astype(int)
        missing = f["vis_missing_nums"][indices, 0]
        if np.any(missing != 0):
            raise ValueError("Selected integrations have missing packets")
        total = np.zeros(len(pairs), dtype=np.complex128)
        dataset = f["visibilities"]
        for integration in indices:
            values = dataset[int(integration), channel, :, hand, hand, :]
            total += values[:, 0] + 1j * values[:, 1]
    return pairs, total / len(indices)


def read_masked_spectrum(capture_path, integrations, mask_good, pol="YY"):
    """Average every unmasked time/fine-channel cell, reading HDF5 chunks once.

    Returns the packed baseline list, complex mean [frequency, baseline], and
    number of complete integrations contributing to each fine channel.
    """
    indices = np.unique(np.asarray(integrations, dtype=int))
    if not len(indices):
        raise ValueError("No integrations selected")
    with h5py.File(capture_path, "r") as f:
        dataset = f["visibilities"]
        ntime, nfreq, npair = dataset.shape[:3]
        if mask_good.shape != (ntime, nfreq):
            raise ValueError("RFI mask does not match capture dimensions")
        if indices[0] < 0 or indices[-1] >= ntime:
            raise IndexError("Integration outside capture")
        if np.any(f["vis_missing_nums"][indices, 0] != 0):
            raise ValueError("Selected integrations have missing packets")
        pairs = f["baseline_antenna_ids"][:].astype(int)
        total = np.zeros((nfreq, npair), np.complex128)
        counts = np.zeros(nfreq, np.int32)
        hand = {"XX": 1, "YY": 0}[pol]
        chunk_rows = dataset.chunks[0] if dataset.chunks else 1
        for chunk_start in np.unique(indices // chunk_rows) * chunk_rows:
            chunk_stop = min(chunk_start + chunk_rows, ntime)
            block = dataset[chunk_start:chunk_stop, :, :, hand, hand, :]
            for integration in indices[(indices >= chunk_start) & (indices < chunk_stop)]:
                keep = mask_good[integration]
                values = block[integration - chunk_start]
                total[keep] += values[keep, :, 0] + 1j * values[keep, :, 1]
                counts[keep] += 1
    mean = np.divide(total, counts[:, None],
                     out=np.full_like(total, np.nan + 0j),
                     where=counts[:, None] > 0)
    return pairs, mean, counts


def wideband_dirty_image(pairs, spectrum, counts, frequency_hz, positions, *,
                         gains=None, antenna_ids=None, allowed_pairs=None,
                         baseline_min_m=10.0, pixels=512):
    """Bilinear UV-gridded dirty image using all valid fine channels.

    Each unmasked integration/channel contributes through ``counts``. For a
    readable structural comparison, a robust per-channel visibility scale is
    removed before gridding; this image has no flux-density calibration.
    The array is essentially coplanar; we retain the direct imager for exact
    w-term inspection of a selected channel.
    """
    pairs = np.asarray(pairs, int)
    spectrum = np.asarray(spectrum)
    counts = np.asarray(counts)
    frequency_hz = np.asarray(frequency_hz)
    if spectrum.shape != (len(frequency_hz), len(pairs)) or counts.shape != frequency_hz.shape:
        raise ValueError("Spectrum, frequency and count shapes differ")
    baseline = np.array([np.subtract(positions[int(a)], positions[int(b)])
                         for a, b in pairs], dtype=float)
    length = np.linalg.norm(baseline, axis=1)
    pair_good = (length >= baseline_min_m) & (pairs[:, 0] != pairs[:, 1])
    if antenna_ids is not None:
        pair_good &= np.all(np.isin(pairs, antenna_ids), axis=1)
    if allowed_pairs is not None:
        pair_good &= np.asarray(allowed_pairs, bool)
    if not pair_good.any():
        raise ValueError("No usable cross baselines")
    values = spectrum[:, pair_good].astype(np.complex128, copy=True)
    selected_pairs = pairs[pair_good]
    if gains is not None:
        if antenna_ids is None:
            raise ValueError("antenna_ids required with gains")
        gain_index = {int(a): i for i, a in enumerate(antenna_ids)}
        ia = np.array([gain_index[int(a)] for a in selected_pairs[:, 0]])
        ib = np.array([gain_index[int(b)] for b in selected_pairs[:, 1]])
        denominator = gains[:, ia] * np.conj(gains[:, ib])
        values = np.divide(values, denominator,
                           out=np.full_like(values, np.nan + 0j),
                           where=np.isfinite(denominator) & (np.abs(denominator) > 1e-12))
    scale = np.nanmedian(np.abs(values), axis=1)
    channel_good = (counts > 0) & np.isfinite(scale) & (scale > 0)
    if not channel_good.any():
        raise ValueError("No unmasked fine channels with finite visibilities")
    values = values[channel_good] / scale[channel_good, None]
    freq = frequency_hz[channel_good]
    weights = np.broadcast_to(counts[channel_good, None], values.shape)
    valid = np.isfinite(values)
    uv = baseline[pair_good, :2][None] * (freq[:, None, None] / C)
    uu = np.broadcast_to(uv[:, :, 0], values.shape)[valid]
    vv = np.broadcast_to(uv[:, :, 1], values.shape)[valid]
    z = values[valid]
    weight = weights[valid].astype(float)
    # Visible hemisphere spans l,m in [-1,1]. UV grid spacing is 0.5 lambda;
    # 512 pixels retain baselines to 128 lambda.
    x = 2 * uu + pixels / 2
    y = 2 * vv + pixels / 2
    within = (x >= 1) & (x < pixels - 2) & (y >= 1) & (y < pixels - 2)
    x, y, z, weight = x[within], y[within], z[within], weight[within]
    if not len(z):
        raise ValueError("UV grid is too small for the available baselines")
    uvgrid = np.zeros((pixels, pixels), np.complex128)
    for u, v, sample in ((x, y, z), (pixels - x, pixels - y, np.conj(z))):
        ix = np.floor(u).astype(int)
        iy = np.floor(v).astype(int)
        fx = u - ix
        fy = v - iy
        for dx in (0, 1):
            for dy in (0, 1):
                interp = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
                np.add.at(uvgrid, (iy + dy, ix + dx), sample * weight * interp)
    axis = np.linspace(-1, 1, pixels, endpoint=False)
    image = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(uvgrid))).real
    image /= 2 * np.sum(weight)
    # Bilinear gridding convolves UV with a triangle; undo its broad image
    # attenuation without altering the zero-frequency normalization.
    east, north = np.meshgrid(axis, axis)
    image /= np.sinc(east / 2)**2 * np.sinc(north / 2)**2
    image[east * east + north * north > 1] = np.nan
    return axis, image, int(channel_good.sum()), int(len(z))


def dirty_image(pairs, vis, positions, frequency_hz, *, gains=None,
                baseline_min_m=10.0, allowed_pairs=None, extent_deg=90.0,
                pixels=181):
    """Zenith-centred direct Fourier dirty image (no CLEAN or primary-beam correction).

    Visibility convention matches compare_point_calibration.py: V_ij has
    positive geometric phase for position_i - position_j, so imaging uses
    the negative phase. Auto-correlations and short baselines are excluded.
    """
    baseline = np.array([np.subtract(positions[int(a)], positions[int(b)])
                         for a, b in pairs], dtype=float)
    length = np.linalg.norm(baseline, axis=1)
    valid = np.isfinite(vis) & (length >= baseline_min_m) & (pairs[:, 0] != pairs[:, 1])
    if allowed_pairs is not None:
        valid &= np.asarray(allowed_pairs, dtype=bool)
    if gains is not None:
        denominator = np.array([gains.get(int(a), np.nan) *
                                np.conj(gains.get(int(b), np.nan))
                                for a, b in pairs])
        valid &= np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
        vis = np.divide(vis, denominator, out=np.zeros_like(vis), where=valid)
    uvw = baseline[valid] * (frequency_hz / C)
    samples = vis[valid]
    if len(samples) == 0:
        raise ValueError("No valid cross baselines in this image")
    limit = np.sin(np.deg2rad(extent_deg))
    grid = np.linspace(-limit, limit, pixels)
    east, north = np.meshgrid(grid, grid)
    inside = east * east + north * north < 1
    up = np.sqrt(np.maximum(1 - east * east - north * north, 0)) - 1
    image = np.zeros((pixels, pixels), dtype=float)
    # Keep the temporary phase array small. Including the conjugate baselines
    # makes this exactly the real part of the complex sum.
    for start in range(0, len(samples), 32):
        uv = uvw[start:start + 32]
        phase = -2j * np.pi * (uv[:, 0, None, None] * east +
                              uv[:, 1, None, None] * north +
                              uv[:, 2, None, None] * up)
        image += np.sum((samples[start:start + 32, None, None] *
                         np.exp(phase)).real, axis=0)
    image /= len(samples)
    image[~inside] = np.nan
    return grid, image, int(len(samples))


def source_offsets(catalogue, config, mjd):
    """Expected apparent source component positions in local east/north."""
    from astropy import units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time
    from astropy.utils import iers

    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    loc = config["array_location"]
    site = EarthLocation.from_geodetic(loc["longitude_deg"] * u.deg,
                                       loc["latitude_deg"] * u.deg,
                                       loc["height_m"] * u.m)
    out = []
    for source in catalogue["sources"]:
        components = source["components"]
        if not components:
            continue
        component = components[0]
        sky = SkyCoord(component["ra_deg"] * u.deg,
                       component["dec_deg"] * u.deg)
        altaz = sky.transform_to(AltAz(obstime=Time(mjd, format="mjd"), location=site))
        alt, az = altaz.alt.rad, altaz.az.rad
        if alt > 0:
            out.append((source["name"], np.cos(alt) * np.sin(az),
                        np.cos(alt) * np.cos(az)))
    return out

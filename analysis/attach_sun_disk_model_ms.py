#!/usr/bin/env python3
"""Copy a raw LAMBDA36 MS and attach a moving Sun-disk MODEL_DATA column.

This is a phase-model diagnostic. The apparent Sun flux is fixed to one
arbitrary unit, so solved gain amplitudes have no absolute flux meaning.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord, get_sun
from astropy.time import Time
from astropy.utils import iers
from casacore.tables import makearrcoldesc, maketabdesc, table
from pyuvdata import UVData, utils as uvutils
from pyuvdata.telescopes import Telescope
from scipy.special import j1

from lambda_commissioning.constants import NARRIBRI, c
from lambda_commissioning.modelling import calc_lmn
from lambda_commissioning.utils import make_telescope_model

iers.conf.auto_download = False
iers.conf.auto_max_age = None


def native_sun_disk(freq, uu, vv, ww, stamp, radius_deg):
    """Return native-covariance Sun visibility [baseline, frequency]."""
    sun = get_sun(stamp).transform_to(AltAz(obstime=stamp, location=NARRIBRI))
    l, m, n = (float(v[0]) for v in calc_lmn(
        np.array([sun.alt.deg]), np.array([sun.az.deg]), degrees=True))
    radial = uu * l + vv * m + ww * (n - 1)
    projected = np.sqrt(np.maximum(
        0, uu * uu + vv * vv + ww * ww - (uu * l + vv * m + ww * n) ** 2))
    x = 2 * np.pi * np.deg2rad(radius_deg) * projected[:, None] * freq[None, :] / c
    disk = np.divide(2 * j1(x), x, out=np.ones_like(x), where=x > 1e-10)
    return disk * np.exp(-2j * np.pi * radial[:, None] * freq[None, :] / c)


def attach(source: Path, output: Path, radius_deg: float,
           flags_from: Path | None = None) -> None:
    if output.exists():
        raise FileExistsError(output)
    if not 0 <= radius_deg <= 1:
        raise ValueError("Sun radius must be 0..1 degree")
    with table(str(source), ack=False) as src:
        times = src.getcol("TIME")
        a = src.getcol("ANTENNA1")
        b = src.getcol("ANTENNA2")
        unique_times, starts, counts = np.unique(times, return_index=True,
                                                 return_counts=True)
        if len(set(counts)) != 1 or not np.array_equal(starts, np.arange(len(starts)) * counts[0]):
            raise ValueError("MS rows must be regular time-major baselines")
        pairs = np.column_stack([a[:counts[0]], b[:counts[0]]])
        if not np.array_equal(np.column_stack([a, b]), np.tile(pairs, (len(unique_times), 1))):
            raise ValueError("baseline row order changes across integrations")
        ra, dec = None, None
        copied = src.copy(str(output), deep=True, valuecopy=True)
        copied.close()
    with table(str(source / "FIELD"), ack=False) as field:
        ra, dec = field.getcell("PHASE_DIR", 0).ravel()
    with table(str(source / "SPECTRAL_WINDOW"), ack=False) as spw:
        freq = spw.getcell("CHAN_FREQ", 0)
    with table(str(output), readonly=False, ack=False) as dest:
        if "MODEL_DATA" not in dest.colnames():
            dest.addcols(maketabdesc(makearrcoldesc(
                "MODEL_DATA", 0j, shape=[len(freq), 2],
                datamanagertype="TiledShapeStMan", datamanagergroup="MODEL_DATA",
                valuetype="complex")))
        dest.putcolkeyword("MODEL_DATA", "QuantumUnits", "arbitrary Sun units")
    if flags_from is not None:
        with table(str(flags_from), ack=False) as reference, table(
                str(output), readonly=False, ack=False) as dest:
            if reference.nrows() != dest.nrows():
                raise ValueError("flag reference has a different number of rows")
            for name in ("TIME", "ANTENNA1", "ANTENNA2"):
                if not np.array_equal(reference.getcol(name), dest.getcol(name)):
                    raise ValueError(f"flag reference {name} differs")
            for start in range(0, dest.nrows(), counts[0]):
                mask = reference.getcol("FLAG", start, counts[0])
                dest.putcol("FLAG", mask, start, counts[0])
                dest.putcol("FLAG_ROW", mask.all(axis=(1, 2)), start, counts[0])

    array = make_telescope_model(np.arange(1, 37))
    uu = array.uu_m[pairs[:, 0], pairs[:, 1]]
    vv = array.vv_m[pairs[:, 0], pairs[:, 1]]
    ww = array.ww_m[pairs[:, 0], pairs[:, 1]]
    enu = np.column_stack([array.east, array.north, np.zeros(36)])
    ecef = uvutils.ECEF_from_ENU(
        enu, latitude=NARRIBRI.lat.rad, longitude=NARRIBRI.lon.rad,
        altitude=NARRIBRI.height.to_value(u.m))
    origin = np.array([value.to_value(u.m) for value in NARRIBRI.geocentric])
    telescope = Telescope.new(
        name="LAMBDA36", location=NARRIBRI, instrument="LAMBDA36",
        antenna_names=[f"A{i+1}" for i in range(36)],
        antenna_numbers=np.arange(36), antenna_positions=ecef-origin)
    with table(str(source), ack=False) as src, table(str(output), readonly=False, ack=False) as dest:
        for ti, seconds in enumerate(unique_times):
            first, count = int(starts[ti]), int(counts[ti])
            stamp = Time(seconds / 86400 + 2400000.5, format="jd", scale="utc")
            native = native_sun_disk(freq, uu, vv, ww, stamp, radius_deg)
            # Raw HDF5 stores covariance V_ij, while this MS uses conj(V_ij).
            unphased = np.repeat(native.conj()[:, :, None], 2, axis=2)
            flags = src.getcol("FLAG", first, count)
            uvd = UVData.new(
                freq_array=freq, polarization_array=np.array([-5, -6]),
                telescope=telescope, times=np.array([stamp.jd]), antpairs=pairs,
                data_array=unphased.astype(np.complex64), flag_array=flags,
                nsample_array=np.ones(unphased.shape, np.float32),
                integration_time=float(src.getcell("INTERVAL", first)),
                channel_width=float(np.median(np.diff(freq))), vis_units="Jy",
                do_blt_outer=True)
            uvd.phase(ra=ra, dec=dec, epoch="J2000", cat_name="zenith_midpoint",
                      cat_type="sidereal", use_ant_pos=True)
            uvw_error = np.max(np.abs(uvd.uvw_array - src.getcol("UVW", first, count)))
            if uvw_error > 1e-3:
                raise ValueError(f"UVW model mismatch at time {ti}: {uvw_error} m")
            dest.putcol("MODEL_DATA", uvd.data_array.astype(np.complex64), first, count)
            if ti % 10 == 0:
                print(f"modelled {ti + 1}/{len(unique_times)} times", flush=True)
        dest.putkeyword("LAMBDA_MODEL_KIND", "moving_uniform_sun_disk_phase_diagnostic")
        dest.putkeyword("LAMBDA_MODEL_SUN_RADIUS_DEG", radius_deg)
    report = {"source": str(source.resolve()), "output": str(output.resolve()),
              "flags_from": None if flags_from is None else str(flags_from.resolve()),
              "times": len(unique_times), "baselines": len(pairs),
              "frequencies": len(freq), "sun_radius_deg": radius_deg,
              "apparent_flux_unit": "arbitrary", "phase_reference": "zenith_midpoint"}
    Path(str(output) + ".model.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output}/MODEL_DATA", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--radius-deg", type=float, default=0.27)
    p.add_argument("--flags-from", type=Path)
    a = p.parse_args()
    attach(a.source, a.output, a.radius_deg, a.flags_from)

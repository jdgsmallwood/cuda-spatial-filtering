#!/usr/bin/env python3
"""Test fixed sky-source fringe phases in Sun-subtracted visibilities.

The test is exploratory: a positive real matched amplitude in independent
time/frequency/hand splits is evidence for emission at the trial direction,
but a point-source model is inadequate for extended objects such as Cen A.
"""

import argparse
import json
from pathlib import Path

from astropy import units as u
from astropy.coordinates import SkyCoord, get_sun
from astropy.time import Time
from casacore.tables import table
import numpy as np

from plot_wsclean_allsky import BRIGHT_SOURCE_CATALOG


C = 299792458.0
SOURCE_NAMES = ("Cen A", "Her A", "Hyd A", "Pic A", "Vir A", "Sgr A*")


def direction_cosines(ra, dec, ra0, dec0):
    difference = ra - ra0
    return np.array([np.cos(dec) * np.sin(difference),
                     np.sin(dec) * np.cos(dec0) -
                     np.cos(dec) * np.sin(dec0) * np.cos(difference),
                     np.sin(dec) * np.sin(dec0) +
                     np.cos(dec) * np.cos(dec0) * np.cos(difference) - 1])


def main(source: Path, model_path: Path, scale_path: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    with table(str(source / "FIELD"), ack=False) as field:
        ra0, dec0 = field.getcell("PHASE_DIR", 0).ravel()
    with table(str(source / "SPECTRAL_WINDOW"), ack=False) as spectral:
        frequencies = spectral.getcell("CHAN_FREQ", 0)[0:411:4]
    with np.load(scale_path) as fit:
        apparent_sun = fit["apparent_flux"][:, 0:411:4]
    candidates = {name: BRIGHT_SOURCE_CATALOG[name] for name in SOURCE_NAMES}
    cen = candidates["Cen A"]
    for offset_ra, offset_dec in ((-10, 0), (-5, 0), (5, 0), (10, 0),
                                  (0, -10), (0, -5), (0, 5), (0, 10)):
        candidates[f"Cen control {offset_ra:+}RA {offset_dec:+}Dec"] = SkyCoord(
            ra=(cen.ra.deg + offset_ra) * u.deg,
            dec=(cen.dec.deg + offset_dec) * u.deg)
    direction = {name: direction_cosines(sky.ra.rad, sky.dec.rad, ra0, dec0)
                 for name, sky in candidates.items()}
    sum_ = {name: np.zeros((2, 2, 2, 3), complex) for name in candidates}
    count = {name: np.zeros((2, 2, 2, 3), int) for name in candidates}
    with table(str(source), ack=False) as data, table(str(model_path), ack=False) as model:
        if data.nrows() != model.nrows() or data.nrows() % 630:
            raise ValueError("source and Sun model rows differ")
        for column in ("TIME", "ANTENNA1", "ANTENNA2"):
            if not np.array_equal(data.getcol(column), model.getcol(column)):
                raise ValueError(f"{column} differs")
        first_stamp = Time(data.getcell("TIME", 0) / 86400 + 2400000.5,
                           format="jd", scale="utc")
        sun = get_sun(first_stamp)
        sun_direction = direction_cosines(sun.ra.rad, sun.dec.rad, ra0, dec0)
        uvw_first = data.getcol("UVW", 0, 630)
        actual_sun = model.getcol("MODEL_DATA", 0, 630)[:, 0, 0]
        geometric_sun = np.exp(2j * np.pi * frequencies[0] / C *
                               (uvw_first @ sun_direction))
        sun_phase_coherence = abs(np.mean(
            actual_sun * geometric_sun.conj() / abs(actual_sun)))
        if sun_phase_coherence < .99:
            raise ValueError("point-source fringe sign fails the Sun-model check")
        baseline_length = np.linalg.norm(uvw_first, axis=1)
        selected = np.flatnonzero(baseline_length > 7)
        if data.nrows() // 630 != apparent_sun.shape[0]:
            raise ValueError("apparent Sun scale has a different number of times")
        for ti in range(data.nrows() // 630):
            uvw = data.getcol("UVW", ti * 630, 630)[selected]
            vis = data.getcol("DATA", ti * 630, 630)[selected, 0:411:4]
            flag = data.getcol("FLAG", ti * 630, 630)[selected, 0:411:4]
            group = np.where(baseline_length[selected] < 15, 1, 2)
            for pi in range(2):
                flux = apparent_sun[ti, :, pi]
                okay = (~flag[:, :, pi]) & np.isfinite(flux[None, :]) & (flux[None, :] > 0)
                scaled = np.divide(vis[:, :, pi], flux[None, :],
                                   out=np.zeros_like(vis[:, :, pi]),
                                   where=np.isfinite(flux[None, :]) & (flux[None, :] > 0))
                for name, cosines in direction.items():
                    phase = 2 * np.pi / C * (uvw @ cosines)[:, None] * frequencies[None, :]
                    rotated = scaled * np.exp(-1j * phase)
                    for ci, freq_slice in enumerate((slice(0, 52), slice(52, None))):
                        for gi in range(3):
                            use = okay[:, freq_slice] & ((group == gi)[..., None]
                                                           if gi else True)
                            sum_[name][pi, ti // 61, ci, gi] += rotated[:, freq_slice][use].sum()
                            count[name][pi, ti // 61, ci, gi] += np.count_nonzero(use)
    rows = []
    for name in candidates:
        mean = sum_[name] / np.maximum(count[name], 1)
        split = mean[:, :, :, 0]
        rows.append({"Source": name,
                     "Matched real / Sun": round(float(split.real.mean()), 6),
                     "Matched imaginary / Sun": round(float(split.imag.mean()), 6),
                     "Positive splits": f"{int(np.count_nonzero(split.real > 0))}/8",
                     "Minimum split real / Sun": round(float(split.real.min()), 6),
                     "XX split real": np.round(split[0].real.ravel(), 6).tolist(),
                     "YY split real": np.round(split[1].real.ravel(), 6).tolist(),
                     "Short-baseline real / Sun": round(float(mean[:, :, :, 1].real.mean()), 6),
                     "Long-baseline real / Sun": round(float(mean[:, :, :, 2].real.mean()), 6),
                     "control": name.startswith("Cen control")})
    output.write_text(json.dumps({"source_ms": str(source),
                                  "sun_model": str(model_path),
                                  "phase_sign_sun_model_coherence": float(sun_phase_coherence),
                                  "frequency_mhz": [float(frequencies[0] / 1e6),
                                                    float(frequencies[-1] / 1e6)],
                                  "method": "real part of geometric fringe-matched "
                                  "Sun-subtracted visibility, normalized to fitted "
                                  "apparent Sun; 7 m baseline cut; 2 time x 2 "
                                  "frequency x 2 hand splits; selected every 4th "
                                  "fine channel for computational efficiency",
                                  "rows": rows}, indent=2) + "\n")
    print(json.dumps({"sun_phase_coherence": float(sun_phase_coherence),
                      "rows": rows[:6]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("sun_model", type=Path)
    parser.add_argument("apparent_sun_fit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.source, args.sun_model, args.apparent_sun_fit, args.output)

#!/usr/bin/env python3
"""WCS-aware all-sky WSClean display with bright sources and Galactic plane."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib import patheffects
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord, get_sun
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.utils import iers
from astropy.wcs import WCS

from lambda_commissioning.constants import NARRIBRI

# Fixed positions from lambda_commissioning.vistools.BRIGHT_SOURCE_CATALOG.
# Keep the plotting tool independent of that module's optional analysis imports.
BRIGHT_SOURCE_CATALOG = {
    "Cen A": SkyCoord("13h25m28s", "-43d01m09s"),
    "Tau A": SkyCoord("05h34m32s", "+22d00m52s"),
    "Her A": SkyCoord("16h51m08s", "+04d59m33s"),
    "Hyd A": SkyCoord("09h18m06s", "-12d05m44s"),
    "Pic A": SkyCoord("05h19m50s", "-45d46m44s"),
    "Vir A": SkyCoord("12h30m49s", "+12d23m28s"),
    "Orn A": SkyCoord("05h35m17s", "-05d23m23s"),
    "Cas A": SkyCoord("23h23m24s", "+58d48m54s"),
    "Cyg A": SkyCoord("19h59m28s", "+40d44m02s"),
    "For A": SkyCoord("03h22m42s", "-37d12m30s"),
    "Sgr A*": SkyCoord("17h45m40s", "-29d00m28s"),
}

iers.conf.auto_download = False
iers.conf.auto_max_age = None
iers.conf.iers_degraded_accuracy = "warn"


def visible_pixel(wcs, sky, image, stamp):
    altitude = sky.transform_to(AltAz(obstime=stamp, location=NARRIBRI)).alt.deg
    x, y = wcs.world_to_pixel(sky.icrs)
    if not (np.isfinite(x) and np.isfinite(y) and altitude > 5
            and 0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
        return None
    if not np.isfinite(image[int(y), int(x)]):
        return None
    returned = wcs.pixel_to_world(x, y)
    if returned.separation(sky.icrs).deg > 1e-4:
        return None
    return float(x), float(y), float(altitude)


def main(xx_path: Path, yy_path: Path, report: Path, output: Path,
         imaging_label: str = "joint multiscale CLEAN",
         masked: bool = False,
         xx_band_label: str | None = None,
         yy_band_label: str | None = None) -> None:
    if output.exists():
        raise FileExistsError(output)
    xx, xx_header = fits.getdata(xx_path, header=True)
    yy, yy_header = fits.getdata(yy_path, header=True)
    images = [np.squeeze(xx), np.squeeze(yy)]
    wcs = [WCS(xx_header).celestial, WCS(yy_header).celestial]
    if images[0].shape != images[1].shape:
        raise ValueError("XX and YY images have different shapes")
    if not np.allclose(wcs[0].wcs.crval, wcs[1].wcs.crval, atol=1e-8):
        raise ValueError("XX and YY images have different phase centres")
    details = json.loads(report.read_text())
    stamp = Time(xx_header["DATE-OBS"], scale="utc") + TimeDelta(
        float(details["duration_s"]) / 2, format="sec")
    # get_sun() returns an apparent GCRS direction. Using its numeric RA/Dec
    # on this ICRS map is adequate at this array's degree-scale resolution;
    # transforming the finite-distance Sun object to ICRS would instead
    # describe its barycentric position and put the marker in the wrong sky.
    apparent_sun = get_sun(stamp)
    sun = SkyCoord(ra=apparent_sun.ra, dec=apparent_sun.dec, frame="icrs")
    plane = SkyCoord(l=np.arange(0, 360.25, .25) * u.deg,
                     b=np.zeros(1441) * u.deg, frame="galactic").icrs
    plane_alt = plane.transform_to(AltAz(obstime=stamp, location=NARRIBRI)).alt.deg
    grid_y, grid_x = np.indices(images[0].shape)
    sky_grid = wcs[0].pixel_to_world(grid_x, grid_y)
    latitude = NARRIBRI.lat.rad
    hour_angle = (stamp.sidereal_time("apparent", longitude=NARRIBRI.lon).rad
                  - sky_grid.ra.rad)
    sin_alt = (np.sin(sky_grid.dec.rad) * np.sin(latitude)
               + np.cos(sky_grid.dec.rad) * np.cos(latitude) * np.cos(hour_angle))
    above_horizon = sin_alt > np.sin(np.deg2rad(5))

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    norm = SymLogNorm(linthresh=2.5e5, vmin=-5e6, vmax=5e6)
    colors = plt.get_cmap("RdBu_r").copy()
    colors.set_bad("#d8d8d8")
    outlined = [patheffects.withStroke(linewidth=3, foreground="black")]
    source_rows = []
    for index, (pol, image, coord) in enumerate(zip(("XX", "YY"), images, wcs)):
        ax = fig.add_subplot(1, 2, index + 1, projection=coord)
        shown = ax.imshow(np.ma.masked_where(~above_horizon, image),
                          origin="lower", cmap=colors, norm=norm)
        band_label = ((xx_band_label or "1,045 initially accepted bins · 312–343 MHz")
                      if pol == "XX" else
                      (yy_band_label or "391 initially accepted bins · 312–325 MHz"))
        ax.set_title(f"{pol} | {band_label}", fontsize=13)
        ax.set_xlabel("Right ascension")
        ax.set_ylabel("Declination")
        ax.coords.grid(color="#333333", alpha=.18)

        gx, gy = coord.world_to_pixel(plane)
        valid = (np.isfinite(gx) & np.isfinite(gy) & (plane_alt > 5)
                 & (gx >= 0) & (gx < image.shape[1])
                 & (gy >= 0) & (gy < image.shape[0]))
        valid[1:] &= np.hypot(np.diff(gx), np.diff(gy)) < 25
        ax.plot(np.where(valid, gx, np.nan), np.where(valid, gy, np.nan),
                color="#f72585", lw=2.6, label="Galactic plane",
                path_effects=[patheffects.withStroke(linewidth=4, foreground="black")])

        for name, source in BRIGHT_SOURCE_CATALOG.items():
            position = visible_pixel(coord, source, image, stamp)
            if position is None:
                continue
            x, y, altitude = position
            ax.plot(x, y, "o", ms=8, mfc="none", mec="#ccff33", mew=2.2)
            ax.annotate(name, (x, y), xytext=(7, 6), textcoords="offset points",
                        color="#ccff33", weight="bold", fontsize=9,
                        path_effects=outlined)
            if index == 0:
                source_rows.append({"name": name, "pixel": [x, y],
                                    "elevation_deg": altitude})

        sun_position = visible_pixel(coord, sun, image, stamp)
        if sun_position is not None:
            x, y, altitude = sun_position
            ax.plot(x, y, marker="*", ms=15, mfc="#fff066", mec="black", mew=.7)
            ax.annotate("Sun", (x, y), xytext=(10, -14), textcoords="offset points",
                        color="#fff066", weight="bold", fontsize=10,
                        path_effects=outlined)
            if index == 0:
                source_rows.append({"name": "Sun", "pixel": [x, y],
                                    "elevation_deg": altitude})
        if index == 0:
            ax.legend(loc="lower right", facecolor="black", labelcolor="white")

    fig.colorbar(shown, ax=fig.axes, orientation="horizontal", shrink=.65,
                 label="relative correlator units per beam (Sun saturated; no flux scale)")
    fig.suptitle("2026-10-03 Sun capture | sky above 5° elevation | "
                 f"{imaging_label} | expected sources",
                 fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    manifest = {"xx_fits": str(xx_path.resolve()), "yy_fits": str(yy_path.resolve()),
                "midpoint_utc": stamp.isot, "sources": source_rows,
                "galactic_plane": "IAU Galactic latitude b=0 deg",
                "image_display_elevation_min_deg": 5,
                "imaging_label": imaging_label,
                "display": "shared signed logarithmic scale, clipped at +/-5e6 correlator units",
                "caveats": ["source markers are positions, not detections",
                            "no primary beam or absolute flux correction",
                            ("exploratory Sun-model residual time-frequency mask"
                             if masked else "no independent transient-RFI time mask")]}
    Path(str(output) + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {output}; {len(source_rows) - 1} visible catalogue sources")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("xx_fits", type=Path)
    p.add_argument("yy_fits", type=Path)
    p.add_argument("report", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--imaging-label", default="joint multiscale CLEAN")
    p.add_argument("--masked", action="store_true")
    p.add_argument("--xx-band-label")
    p.add_argument("--yy-band-label")
    a = p.parse_args()
    main(a.xx_fits, a.yy_fits, a.report, a.output, a.imaging_label, a.masked,
         a.xx_band_label, a.yy_band_label)

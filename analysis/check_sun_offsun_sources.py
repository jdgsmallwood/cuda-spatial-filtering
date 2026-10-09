#!/usr/bin/env python3
"""Check catalogue positions in independent Sun-subtracted image splits."""

import argparse
import json
from pathlib import Path
import warnings

from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np

from plot_wsclean_allsky import BRIGHT_SOURCE_CATALOG


def main(directory: Path, prefix: str, output: Path):
    figure_path = output.with_suffix(".png")
    if output.exists() or figure_path.exists():
        raise FileExistsError(output)
    images = []
    names = []
    wcs_reference = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        for pol in ("XX", "YY"):
            for time in range(2):
                for channel in range(2):
                    name = f"{prefix}-t{time:04d}-{channel:04d}-{pol}-dirty.fits"
                    data, header = fits.getdata(directory / name, header=True)
                    data = np.squeeze(data)
                    wcs = WCS(header).celestial
                    if wcs_reference is None:
                        wcs_reference = wcs
                    elif (not np.allclose(wcs.wcs.crval, wcs_reference.wcs.crval)
                          or not np.allclose(wcs.wcs.cdelt, wcs_reference.wcs.cdelt)):
                        raise ValueError(f"sky grid differs for {name}")
                    yy, xx = np.indices(data.shape)
                    off_sun = np.isfinite(data) & (np.hypot(xx - 200, yy - 367) > 25)
                    median = np.median(data[off_sun])
                    scale = 1.4826 * np.median(abs(data[off_sun] - median))
                    if not scale > 0:
                        raise ValueError(f"zero robust scale for {name}")
                    images.append((data - median) / scale)
                    names.append(name)
    cube = np.stack(images)
    stack = np.nanmedian(cube, axis=0)
    yy, xx = np.indices(stack.shape)
    off_sun = np.isfinite(stack) & (np.hypot(xx - 200, yy - 367) > 25)
    visibility_report = directory / "wsclean_joint_allsky_amp_phase_sunpeeled_dirty_sources.png.json"
    if visibility_report.is_file():
        visible_names = {row["name"] for row in
                         json.loads(visibility_report.read_text())["sources"]}
    else:
        visible_names = set(BRIGHT_SOURCE_CATALOG)
    rows = []
    for source, coordinate in BRIGHT_SOURCE_CATALOG.items():
        if source not in visible_names:
            continue
        x, y = wcs_reference.world_to_pixel(coordinate)
        x, y = float(np.asarray(x)), float(np.asarray(y))
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < stack.shape[1] and 0 <= yi < stack.shape[0]):
            continue
        values = cube[:, yi, xi]
        rows.append({"Source": source, "Pixel": [round(x, 2), round(y, 2)],
                     "Median split contrast": round(float(stack[yi, xi]), 3),
                     "Positive splits": f"{int(np.count_nonzero(values > 0))}/8",
                     "Sky-pixel percentile": round(float(np.mean(
                         stack[off_sun] <= stack[yi, xi]) * 100), 2),
                     "Split contrasts": [round(float(v), 3) for v in values]})
    rows.sort(key=lambda row: row["Median split contrast"], reverse=True)
    result = {"input_prefix": prefix, "splits": names, "rows": rows,
              "method": "per-image robust off-Sun normalization; median of two "
                        "time bins x two frequency bins x XX/YY at a fixed "
                        "catalogue position; exploratory image contrast, not "
                        "formal detection significance"}
    output.write_text(json.dumps(result, indent=2) + "\n")

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    ax = fig.add_subplot(111, projection=wcs_reference)
    norm = SymLogNorm(linthresh=.5, vmin=-4, vmax=4)
    shown = ax.imshow(stack, origin="lower", cmap="RdBu_r", norm=norm)
    for row in rows:
        x, y = row["Pixel"]
        ax.plot(x, y, "o", ms=9, mec="#dfff00", mfc="none", mew=2)
        ax.annotate(row["Source"], (x, y), xytext=(6, 7),
                    textcoords="offset points", color="#dfff00",
                    weight="bold", fontsize=9)
    ax.plot(200, 367, "*", ms=14, color="#fff066", mec="black")
    ax.set_title("Sun-subtracted residual: median across 2 time × 2 frequency × 2 hand splits")
    ax.set_xlabel("Right ascension")
    ax.set_ylabel("Declination")
    ax.coords.grid(color="black", alpha=.15)
    fig.colorbar(shown, ax=ax, label="median contrast in per-image robust-noise units")
    fig.savefig(figure_path, dpi=160)
    print(json.dumps({"report": str(output), "figure": str(figure_path),
                      "top_sources": rows[:3]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("prefix")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.directory, args.prefix, args.output)

#!/usr/bin/env python3
"""Compare matched natural and Briggs+5 m WSClean all-sky imaging."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def metrics(image: np.ndarray, off: np.ndarray) -> dict[str, float]:
    return {
        "off_sun_rms": float(np.sqrt(np.mean(np.square(image[off])))),
        "off_sun_p99_abs": float(np.percentile(np.abs(image[off]), 99)),
        "sun_pixel": float(image[367, 200]),
        "marked_sidelobe_pixel": float(image[273, 432]),
    }


def main(directory: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    fig = plt.figure(figsize=(22, 11), constrained_layout=True)
    norm = SymLogNorm(linthresh=2.5e5, vmin=-5e6, vmax=5e6)
    report = {}
    for row, pol in enumerate(("xx", "yy")):
        yy, xx = np.indices((512, 512))
        off = ((xx - 200) ** 2 + (yy - 367) ** 2 > 25 ** 2)
        report[pol] = {}
        for col, (weight, kind) in enumerate((
                ("natural", "image"), ("briggs0_min5m", "image"),
                ("natural", "residual"), ("briggs0_min5m", "residual"))):
            prefix = (f"wsclean_{pol}_joint4_clean" if weight == "natural"
                      else f"wsclean_{pol}_joint4_briggs0_min5m_clean")
            data, header = fits.getdata(directory / f"{prefix}-MFS-{kind}.fits",
                                        header=True)
            data = np.squeeze(data)
            wcs = WCS(header).celestial
            valid = np.isfinite(data) & off
            ax = fig.add_subplot(2, 4, row * 4 + col + 1, projection=wcs)
            shown = ax.imshow(data, origin="lower", cmap="RdBu_r", norm=norm)
            ax.plot(200, 367, marker="*", ms=10, color="#fff066",
                    markeredgecolor="black", markeredgewidth=.5)
            ax.plot(432, 273, marker="o", ms=9, fillstyle="none",
                    color="#00ff88", mew=1.8)
            ax.set_title(f"{pol.upper()} | {weight.replace('_', ' + ')} | {kind}")
            ax.set_xlabel("RA")
            if col == 0:
                ax.set_ylabel("Dec")
            ax.coords.grid(color="black", alpha=.15)
            report[pol][f"{weight}_{kind}"] = metrics(data, valid)
            if kind == "image":
                report[pol][f"{weight}_beam_deg"] = {
                    "major": float(header["BMAJ"]),
                    "minor": float(header["BMIN"]),
                }
        for weight in ("natural", "briggs0_min5m"):
            prefix = (f"wsclean_{pol}_joint4_clean" if weight == "natural"
                      else f"wsclean_{pol}_joint4_briggs0_min5m_clean")
            psf = np.squeeze(fits.getdata(directory / f"{prefix}-MFS-psf.fits"))
            py, px = np.indices(psf.shape)
            off_psf = ((px - 256) ** 2 + (py - 256) ** 2 > 25 ** 2)
            report[pol][f"{weight}_psf_max_abs_outside_25px"] = float(
                np.nanmax(np.abs(psf[off_psf])))
        natural = report[pol]["natural_residual"]["off_sun_rms"]
        briggs = report[pol]["briggs0_min5m_residual"]["off_sun_rms"]
        report[pol]["residual_rms_briggs_over_natural"] = briggs / natural
    fig.colorbar(shown, ax=fig.axes, orientation="horizontal", shrink=.6,
                 label="relative correlator units per beam (shared signed log scale)")
    fig.suptitle("20-minute Sun capture | natural vs Briggs 0 with 5 m minimum "
                 "baseline | yellow: Sun; green: Sun grating response")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    Path(str(output) + ".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.directory, args.output)

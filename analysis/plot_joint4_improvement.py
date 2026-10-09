#!/usr/bin/env python3
"""Compare single-band and joint-band WSClean Sun images on one scale."""

import argparse
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np


def main(single: Path, joint: Path, output: Path) -> None:
    paths = (Path(f"{single}-XX-image.fits"), Path(f"{joint}-MFS-image.fits"),
             Path(f"{joint}-MFS-residual.fits"))
    fig = plt.figure(figsize=(15, 5.4), constrained_layout=True)
    norm = SymLogNorm(linthresh=3e5, vmin=-5e6, vmax=5e6)
    for index, (title, path) in enumerate(zip(
            ("Single-band restored", "Joint four-band restored", "Joint four-band residual"),
            paths)):
        data, header = fits.getdata(path, header=True)
        ax = fig.add_subplot(1, 3, index + 1, projection=WCS(header).celestial)
        shown = ax.imshow(np.squeeze(data), origin="lower", cmap="RdBu_r", norm=norm)
        ax.plot(200, 367, "+", color="cyan", ms=12, mew=2)
        ax.plot(432, 273, "o", color="lime", fillstyle="none", ms=10, mew=2)
        ax.set_title(title)
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
    fig.colorbar(shown, ax=fig.axes, orientation="horizontal", shrink=.65,
                 label="correlator units per beam (no absolute flux calibration)")
    fig.suptitle("XX Sun image: frequency-aware CLEAN suppresses the moving sidelobe")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"wrote {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("single", type=Path)
    p.add_argument("joint", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    main(a.single, a.joint, a.output)

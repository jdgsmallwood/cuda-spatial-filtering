#!/usr/bin/env python3
"""Make a FITS CLEAN mask around the Sun for this WSClean image geometry."""

import argparse
from pathlib import Path

from astropy.io import fits
import numpy as np


def main(reference: Path, output: Path, center_x: float, center_y: float,
         radius_deg: float) -> None:
    if output.exists():
        raise FileExistsError(output)
    data, header = fits.getdata(reference, header=True)
    pixel_deg = abs(float(header["CDELT1"]))
    radius_px = radius_deg / pixel_deg
    y, x = np.indices(data.shape[-2:])
    mask = ((x - center_x) ** 2 + (y - center_y) ** 2 <= radius_px ** 2)
    fits.PrimaryHDU(mask.astype(np.float32).reshape(data.shape),
                    header=header).writeto(output)
    print(f"wrote {output}: {mask.sum()} pixels, radius {radius_px:.1f} px")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--x", type=float, default=200.)
    parser.add_argument("--y", type=float, default=367.)
    parser.add_argument("--radius-deg", type=float, default=4.)
    args = parser.parse_args()
    main(args.reference, args.output, args.x, args.y, args.radius_deg)

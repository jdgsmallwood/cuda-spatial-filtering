#!/usr/bin/env python3
"""Render dirty/restored/residual WSClean images with a matched display scale."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def main(prefix: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    images = {}
    header = None
    for pol in ("XX", "YY"):
        for kind in ("dirty", "image", "residual"):
            data, current = fits.getdata(f"{prefix}-{pol}-{kind}.fits", header=True)
            images[pol, kind] = np.squeeze(data)
            header = current
    wcs = WCS(header).celestial
    fig = plt.figure(figsize=(17, 11), constrained_layout=True)
    norm = SymLogNorm(linthresh=3e5, vmin=-1.5e7, vmax=1.5e7)
    metrics = {}
    for row, pol in enumerate(("XX", "YY")):
        yy, xx = np.indices(images[pol, "dirty"].shape)
        valid = np.isfinite(images[pol, "dirty"])
        # The moving Sun is near (200, 367) for this capture and WCS.
        off = valid & ((xx - 200) ** 2 + (yy - 367) ** 2 > 25 ** 2)
        metrics[pol] = {}
        for col, kind in enumerate(("dirty", "image", "residual")):
            data = images[pol, kind]
            ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection=wcs)
            shown = ax.imshow(data, origin="lower", cmap="RdBu_r", norm=norm)
            ax.plot(200, 367, marker="+", color="cyan", ms=12, mew=2)
            ax.plot(432, 273, marker="o", color="lime", fillstyle="none", ms=10, mew=1.5)
            ax.set_title(f"{pol}: {'restored' if kind == 'image' else kind}")
            ax.set_xlabel("RA")
            ax.set_ylabel("Dec")
            ax.coords.grid(color="black", alpha=.15)
            metrics[pol][kind] = {
                "off_sun_rms": float(np.sqrt(np.mean(np.square(data[off])))),
                "off_sun_p99_abs": float(np.percentile(np.abs(data[off]), 99)),
                "sun_pixel": float(data[367, 200]),
                "marked_patch_pixel": float(data[273, 432]),
            }
        metrics[pol]["residual_over_dirty_rms"] = (
            metrics[pol]["residual"]["off_sun_rms"] /
            metrics[pol]["dirty"]["off_sun_rms"])
    fig.colorbar(shown, ax=fig.axes, orientation="horizontal", shrink=.55,
                 label="correlator units per beam (no absolute flux calibration)")
    fig.suptitle("Full 20-minute Sun capture; fixed color scale across all panels. "
                 "Cyan: Sun. Green: prominent off-Sun patch.")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    Path(str(output) + ".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prefix", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    main(a.prefix, a.output)

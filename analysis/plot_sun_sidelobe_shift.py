#!/usr/bin/env python3
"""Show the frequency motion of a candidate Sun sidelobe in WSClean images."""

import argparse
import json
from pathlib import Path

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


def main(prefix: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), constrained_layout=True)
    records = []
    for index, ax in enumerate(axes):
        image, header = fits.getdata(f"{prefix}-{index:04d}-dirty.fits", header=True)
        image = np.squeeze(image)
        freq = float(header["CRVAL3"])
        sub = image[243:304, 402:463]
        iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
        px, py = int(ix + 402), int(iy + 243)
        sun = float(image[367, 200])
        ax.imshow(image / sun, origin="lower", cmap="inferno", vmin=-.03, vmax=.42)
        ax.plot(200, 367, "+", color="cyan", ms=13, mew=2)
        ax.plot(px, py, "o", color="lime", fillstyle="none", ms=12, mew=2)
        ax.set_xlim(150, 470)
        ax.set_ylim(200, 440)
        ax.set_title(f"{freq/1e6:.2f} MHz")
        ax.set_xlabel("image x pixel")
        if index == 0:
            ax.set_ylabel("image y pixel")
        records.append({"frequency_mhz": freq / 1e6,
                        "peak_pixel": [px, py],
                        "offset_from_sun_pixel": [px - 200, py - 367],
                        "peak_over_sun": float(image[py, px] / sun)})
    ratio_x = records[-1]["offset_from_sun_pixel"][0] / records[0]["offset_from_sun_pixel"][0]
    ratio_y = records[-1]["offset_from_sun_pixel"][1] / records[0]["offset_from_sun_pixel"][1]
    expected = records[0]["frequency_mhz"] / records[-1]["frequency_mhz"]
    fig.suptitle("Sun at cyan cross; moving bright patch at green circle. "
                 f"Offset ratios x={ratio_x:.3f}, y={ratio_y:.3f}; frequency inverse={expected:.3f}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    result = {"bands": records, "offset_ratio_x": ratio_x,
              "offset_ratio_y": ratio_y, "inverse_frequency_ratio": expected}
    Path(str(output) + ".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prefix", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    main(a.prefix, a.output)

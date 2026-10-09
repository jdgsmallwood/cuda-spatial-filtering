#!/usr/bin/env python3
"""Summarize matched-scale solar WSClean frequency-mask trials."""

import argparse
import json
from pathlib import Path

from astropy.io import fits
import numpy as np


TRIALS = {
    "XX": [
        ("Full band", "wsclean_xx_amp_prev_phase_prev_r0_min3m_clean"),
        ("Full band, 318.4–319.7 MHz notch", "wsclean_xx_amp_phase_notch_full"),
        ("312–334 MHz, notch", "wsclean_xx_amp_phase_notch_312_334"),
        ("312–324 MHz, notch", "wsclean_xx_amp_phase_notch_312_324"),
        ("312–317.5 MHz", "wsclean_xx_amp_phase_cleanlow_312_317p5_v2"),
    ],
    "YY": [
        ("Full band", "wsclean_yy_amp_prev_phase_prev_r0_min7m_clean"),
        ("Full band, 318.4–319.7 MHz notch", "wsclean_yy_amp_phase_notch_full"),
        ("312–324 MHz, notch", "wsclean_yy_amp_phase_notch_312_324"),
        ("312–317.5 MHz", "wsclean_yy_amp_phase_cleanlow_312_317p5"),
    ],
}


def main(directory: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for pol, settings in TRIALS.items():
        for label, prefix in settings:
            residual_path = directory / f"{prefix}-MFS-residual.fits"
            image_path = directory / f"{prefix}-MFS-image.fits"
            if not (residual_path.is_file() and image_path.is_file()):
                raise FileNotFoundError(residual_path)
            data, header = fits.getdata(residual_path, header=True)
            data = np.squeeze(data)
            image = np.squeeze(fits.getdata(image_path))
            yy, xx = np.indices(data.shape)
            off = np.isfinite(data) & (np.hypot(xx - 200, yy - 367) > 25)
            rms = float(np.sqrt(np.mean(data[off] ** 2)))
            sun = float(image[367, 200])
            rows.append({"Hand": pol, "Frequency selection": label,
                         "Off-Sun RMS (M)": round(rms / 1e6, 3),
                         "RMS / Sun peak (%)": round(100 * rms / sun, 3),
                         "Beam major (deg)": round(float(header["BMAJ"]), 3),
                         "Beam minor (deg)": round(float(header["BMIN"]), 3),
                         "Prefix": prefix})
    output.write_text(json.dumps({"rows": rows,
                                  "statistic": "off-Sun radius >25 pixels; "
                                  "same Briggs 0 and multiscale CLEAN settings; "
                                  "XX minimum baseline 3 m, YY 7 m"},
                                 indent=2) + "\n")
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.directory, args.output)

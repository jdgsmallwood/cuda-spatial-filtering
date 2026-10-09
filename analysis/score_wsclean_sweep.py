#!/usr/bin/env python3
"""Score matched WSClean imaging settings against the off-Sun residual mask."""

import argparse
import csv
import json
from pathlib import Path

from astropy.io import fits
import numpy as np


SETTINGS = (
    ("natural_all", "joint4_clean", 630),
    ("briggs0_5m", "joint4_briggs0_min5m_clean", 447),
    ("briggs-0.5_5m", "joint4_rm0p5_min5m_clean", 447),
    ("briggs+0.5_5m", "joint4_rp0p5_min5m_clean", 447),
    ("briggs0_3m", "joint4_r0_min3m_clean", 559),
    ("briggs0_7m", "joint4_r0_min7m_clean", 324),
)


def score(directory: Path) -> list[dict]:
    rows = []
    for pol in ("xx", "yy"):
        for label, suffix, baselines in SETTINGS:
            prefix = directory / f"wsclean_{pol}_{suffix}-MFS"
            residual_file = Path(f"{prefix}-residual.fits")
            if not residual_file.exists():
                continue
            residual = np.squeeze(fits.getdata(residual_file))
            image, header = fits.getdata(f"{prefix}-image.fits", header=True)
            image = np.squeeze(image)
            psf = np.squeeze(fits.getdata(f"{prefix}-psf.fits"))
            yy, xx = np.indices(residual.shape)
            off = (np.isfinite(residual)
                   & ((xx - 200)**2 + (yy - 367)**2 > 25**2))
            off_psf = ((xx - 256)**2 + (yy - 256)**2 > 25**2)
            rms = float(np.sqrt(np.mean(residual[off]**2)))
            sun_peak = float(image[367, 200])
            rows.append({
                "polarization": pol.upper(),
                "setting": label,
                "baselines_ge_cut": baselines,
                "off_sun_residual_rms": rms,
                "residual_rms_over_sun_peak": rms / sun_peak,
                "off_sun_residual_p99_abs": float(np.percentile(abs(residual[off]), 99)),
                "sun_peak": sun_peak,
                "marked_sidelobe_residual": float(residual[273, 432]),
                "beam_major_deg": float(header["BMAJ"]),
                "beam_minor_deg": float(header["BMIN"]),
                "psf_max_abs_outside_25px": float(np.max(abs(psf[off_psf]))),
            })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = score(args.directory)
    stem = args.directory / "wsclean_settings_sweep"
    Path(f"{stem}.json").write_text(json.dumps(rows, indent=2) + "\n")
    with Path(f"{stem}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for pol in ("XX", "YY"):
        print(pol)
        for row in sorted((r for r in rows if r["polarization"] == pol),
                          key=lambda r: r["off_sun_residual_rms"]):
            print(f"  {row['setting']:20} RMS={row['off_sun_residual_rms']/1e6:.3f}M "
                  f"beam={row['beam_major_deg']:.2f}x{row['beam_minor_deg']:.2f}deg "
                  f"baselines={row['baselines_ge_cut']}")

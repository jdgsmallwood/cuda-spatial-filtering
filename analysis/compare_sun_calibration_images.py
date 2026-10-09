#!/usr/bin/env python3
"""Matched-scale before/after Sun calibration and masking comparison."""

import argparse
import json
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np


PREFIXES = {
    "XX": ("wsclean_xx_joint4_r0_min3m_clean",
           "wsclean_xx_swapped_8_12_drift_modelmask_r0_min3m_clean"),
    "YY": ("wsclean_yy_joint4_r0_min7m_clean",
           "wsclean_yy_swapped_8_12_drift_modelmask_r0_min7m_clean"),
}


def main(directory: Path, output: Path, causal_amplitude: bool = False,
         prior_phase: bool = False):
    if output.exists():
        raise FileExistsError(output)
    fig = plt.figure(figsize=(22, 11), constrained_layout=True)
    norm = SymLogNorm(linthresh=2.5e5, vmin=-5e6, vmax=5e6)
    report = {}
    for row, (pol, (old, default_new)) in enumerate(PREFIXES.items()):
        if prior_phase:
            old = default_new.replace("_modelmask_", "_modelmask_amp_prev_")
            new = (f"wsclean_{pol.lower()}_amp_prev_phase_prev_r0_"
                   f"min{'3' if pol == 'XX' else '7'}m_clean")
            old_label = "Prior-bin amplitude"
            new_label = "Prior-bin amplitude + phase"
        else:
            new = (default_new.replace("_modelmask_", "_modelmask_amp_prev_")
                   if causal_amplitude else default_new)
            old_label = "Original calibration"
            new_label = ("Hand + drift + mask + prior amp" if causal_amplitude
                         else "Hand + drift + mask")
        report[pol] = {}
        for col, (prefix, state, kind) in enumerate((
                (old, old_label, "image"),
                (new, new_label, "image"),
                (old, old_label, "residual"),
                (new, new_label, "residual"))):
            data, header = fits.getdata(directory / f"{prefix}-MFS-{kind}.fits",
                                        header=True)
            data = np.squeeze(data)
            yy, xx = np.indices(data.shape)
            off = (xx - 200) ** 2 + (yy - 367) ** 2 > 25 ** 2
            ax = fig.add_subplot(2, 4, row * 4 + col + 1,
                                 projection=WCS(header).celestial)
            shown = ax.imshow(data, origin="lower", cmap="RdBu_r", norm=norm)
            ax.plot(200, 367, "*", color="#fff066", ms=10, mec="black", mew=.5)
            ax.set_title(f"{pol}: {state}\n{kind}", fontsize=12)
            ax.set_xlabel("RA")
            if col == 0:
                ax.set_ylabel("Dec")
            ax.coords.grid(color="black", alpha=.15)
            report[pol][f"{'old' if prefix == old else 'new'}_{kind}"] = {
                "off_sun_rms": float(np.sqrt(np.mean(data[off] ** 2))),
                "off_sun_p99_abs": float(np.percentile(np.abs(data[off]), 99)),
                "sun_pixel": float(data[367, 200]),
                "beam_major_deg": float(header["BMAJ"]),
                "beam_minor_deg": float(header["BMIN"]),
                "fits": str((directory / f"{prefix}-MFS-{kind}.fits").resolve()),
            }
        old_rms = report[pol]["old_residual"]["off_sun_rms"]
        new_rms = report[pol]["new_residual"]["off_sun_rms"]
        report[pol]["residual_rms_reduction_fraction"] = 1 - new_rms / old_rms
    fig.colorbar(shown, ax=fig.axes, orientation="horizontal", shrink=.65,
                 label="relative correlator units per beam; shared signed log scale")
    title = ("Sun image: previous-bin phase correction on amplitude-calibrated data"
             if prior_phase else
             "Sun image: original fit vs corrected hands, phase drift, "
             "time-frequency mask" + (" and prior-bin amplitude gains"
                                      if causal_amplitude else ""))
    fig.suptitle(title, fontsize=15)
    fig.savefig(output, dpi=150)
    Path(str(output) + ".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--causal-amplitude", action="store_true")
    parser.add_argument("--prior-phase", action="store_true")
    args = parser.parse_args()
    main(args.directory, args.output, args.causal_amplitude, args.prior_phase)

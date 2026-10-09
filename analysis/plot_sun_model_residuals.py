#!/usr/bin/env python3
"""Show held-out Sun-model residuals across time, frequency and baselines."""

import argparse
import json
from pathlib import Path

from casacore.tables import table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(diagnostic: Path, ms_path: Path, output: Path):
    with np.load(diagnostic) as saved:
        residual = saved["residual_fraction"]
        freq = saved["frequency_hz"] / 1e6
        by_pair = saved["per_baseline_residual"]
        flux = saved["apparent_flux"]
        phase = saved["phase_residual_rad"]
        held = saved["held_indices"]
    with table(str(ms_path), ack=False) as ms:
        a = ms.getcol("ANTENNA1", 0, 630)
        b = ms.getcol("ANTENNA2", 0, 630)
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    mins = np.arange(residual.shape[0]) * 10 / 60
    summary = {}
    for pi, label in enumerate(("XX", "YY")):
        im = axes[0, pi].imshow(residual[:, :, pi].T, origin="lower", aspect="auto",
                 extent=[mins[0], mins[-1], freq[0], freq[-1]], vmin=0, vmax=0.5,
                 cmap="magma")
        axes[0, pi].set_title(f"{label}: held-out |data − moving Sun| / Sun")
        axes[0, pi].set_xlabel("Minutes from start")
        axes[0, pi].set_ylabel("MHz")
        fig.colorbar(im, ax=axes[0, pi])
        med_time = np.nanmedian(residual[:, :, pi], axis=1)
        axes[1, pi].plot(mins, med_time, lw=1.5, label="median held-out residual")
        axes[1, pi].plot(mins, np.nanpercentile(residual[:, :, pi], 90, axis=1),
                         lw=0.8, alpha=0.6, label="90th percentile")
        axes[1, pi].set_title(f"{label}: residual by integration")
        axes[1, pi].set_xlabel("Minutes from start")
        axes[1, pi].set_ylabel("Fraction of apparent Sun")
        axes[1, pi].set_ylim(0, min(2, np.nanpercentile(residual[:, :, pi], 99.5)))
        axes[1, pi].legend(fontsize=8)
        med_freq = np.nanmedian(residual[:-1, :, pi], axis=0)
        axes[2, pi].plot(freq, med_freq, lw=0.8, label="median fractional residual")
        axes[2, pi].plot(freq, np.nanmedian(phase[:-1, :, pi], axis=0),
                         lw=0.8, alpha=0.7, label="median phase error (rad)")
        axes[2, pi].set_title(f"{label}: frequency structure")
        axes[2, pi].set_xlabel("MHz")
        axes[2, pi].set_ylim(0, min(1, np.nanpercentile(med_freq, 99.5) * 1.2))
        axes[2, pi].legend(fontsize=8)
        accepted = np.isfinite(med_freq)
        worst_freq = np.argsort(np.nan_to_num(med_freq, nan=-1))[-12:][::-1]
        ant_score = np.full(36, np.nan)
        for antenna in range(36):
            baselines = np.flatnonzero(((a == antenna) | (b == antenna)))
            baselines = np.intersect1d(baselines, held)
            ant_score[antenna] = np.nanmedian(by_pair[:-1, baselines, pi])
        ranked = np.argsort(np.nan_to_num(ant_score, nan=-1))[-10:][::-1]
        ranked = ranked[np.isfinite(ant_score[ranked])]
        summary[label] = {
            "median_held_fraction": float(np.nanmedian(residual[:-1, :, pi])),
            "median_time_fraction_min_max": [float(np.nanmin(med_time[:-1])),
                                             float(np.nanmax(med_time[:-1]))],
            "worst_integrations_excluding_last": np.argsort(med_time[:-1])[-10:][::-1].tolist(),
            "worst_frequencies_mhz": freq[worst_freq].tolist(),
            "highest_residual_antenna_ids": (ranked+1).tolist(),
            "highest_residual_antenna_scores": ant_score[ranked].tolist(),
            "apparent_flux_first_last_complete_ratio": float(
                np.nanmedian(flux[-2, accepted, pi]) /
                np.nanmedian(flux[0, accepted, pi])),
            "apparent_flux_final_partial_ratio": float(
                np.nanmedian(flux[-1, accepted, pi]) /
                np.nanmedian(flux[0, accepted, pi])),
        }
    fig.savefig(output, dpi=150)
    Path(str(output) + ".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("diagnostic", type=Path)
    p.add_argument("ms", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    main(a.diagnostic, a.ms, a.output)

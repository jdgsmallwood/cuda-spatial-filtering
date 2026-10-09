#!/usr/bin/env python3
"""Plot held-out Sun-model frequency errors and short-time FPGA phase terms."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def circular_mean(phase, valid, axis):
    z = np.where(valid, np.exp(1j * phase), np.nan + 0j)
    with np.errstate(invalid="ignore"):
        return np.angle(np.nanmean(z, axis=axis))


def main(residual_path: Path, phase_path: Path, config_path: Path,
         output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    with np.load(residual_path) as data:
        frequency = data["frequency_hz"] / 1e6
        residual = data["residual_fraction"]
        phase_error = np.rad2deg(data["phase_residual_rad"])
    with np.load(phase_path) as data:
        antenna_phase = data["phase"]
        supported = data["supported"]
    config = json.loads(config_path.read_text())
    board_of = {int(antenna): int(board)
                for board, antennas in config["antenna_positions"].items()
                for antenna in antennas}
    figure, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    colors = ("#0969da", "#d1242f")
    metrics = {"source": str(residual_path), "phase_fit": str(phase_path),
               "excluded_frequency_mhz": {"both": [318.4, 319.7],
                                           "XX_above": 334, "YY_above": 324},
               "hands": {}}
    for pi, pol in enumerate(("XX", "YY")):
        med = np.nanmedian(residual[:, :, pi], axis=0)
        med_phase = np.nanmedian(phase_error[:, :, pi], axis=0)
        axes[0, 0].plot(frequency, med, color=colors[pi], lw=1.3,
                        label=pol)
        axes[0, 1].plot(frequency, med_phase, color=colors[pi], lw=1.3,
                        label=pol)
        board_phase = []
        for board in range(4):
            ids = np.array([ant - 1 for ant in range(1, 37)
                            if board_of[ant] == board])
            board_phase.append(circular_mean(antenna_phase[:, pi, ids],
                                             supported[:, pi, ids], axis=1))
        times = np.arange(len(board_phase[0])) / 6.0
        board_stats = {}
        for board in (1, 2, 3):
            difference = np.rad2deg(np.unwrap(np.angle(np.exp(
                1j * (board_phase[board] - board_phase[0])))))
            axes[1, pi].plot(times, difference, lw=1.4,
                             label=f"FPGA {board} − 0")
            finite = np.isfinite(difference)
            slope = np.polyfit(times[finite], difference[finite], 1)[0]
            board_stats[str(board)] = {
                "relative_phase_range_deg": float(np.ptp(difference[finite])),
                "linear_slope_deg_per_min": float(slope),
            }
        axes[1, pi].set_title(f"{pol}: broadband FPGA phase relative to FPGA 0")
        axes[1, pi].set_xlabel("Time since capture start (min)")
        axes[1, pi].set_ylabel("Relative phase (degrees)")
        axes[1, pi].grid(alpha=.2)
        axes[1, pi].legend(loc="best", ncol=3, fontsize=9)
        metrics["hands"][pol] = {"board_phase": board_stats}
    for ax in axes[0]:
        ax.axvspan(318.4, 319.7, color="#ffc107", alpha=.28,
                   label="tested notch")
        ax.axvline(324, color="#d1242f", ls="--", alpha=.6)
        ax.axvline(334, color="#0969da", ls="--", alpha=.6)
        ax.set_xlim(312, 343.5)
        ax.set_xlabel("Frequency (MHz)")
        ax.grid(alpha=.2)
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].set_ylabel("Median held-baseline |data − Sun| / |Sun|")
    axes[0, 1].set_ylim(0, 35)
    axes[0, 1].set_ylabel("Median held-baseline phase error (degrees)")
    axes[0, 0].legend(loc="upper left")
    axes[0, 1].legend(loc="upper left")
    figure.suptitle("Latest calibrated Sun: held-out frequency errors and residual FPGA phase",
                    fontsize=16)
    figure.savefig(output, dpi=160)
    Path(str(output) + ".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("residual", type=Path)
    parser.add_argument("phase", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.residual, args.phase, args.config, args.output)

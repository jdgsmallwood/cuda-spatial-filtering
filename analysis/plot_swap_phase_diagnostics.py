#!/usr/bin/env python3
"""Show the phase-residual evidence for capture-local X/Y swaps."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(directory: Path):
    with np.load(directory / "validation.npz") as saved:
        arrays = {key: saved[key] for key in saved.files}
    return arrays, json.loads((directory / "report.json").read_text())


def phases(arrays, report, pol_index, window, common_good, common_excluded):
    upper = np.triu(np.ones((36, 36), bool), 1)
    allowed = ~np.isin(arrays["antenna_order"], list(common_excluded))
    ii, jj = np.where(arrays["selected"] & upper
                      & allowed[:, None] & allowed[None, :])
    f = np.flatnonzero(common_good)
    data = arrays["data"][window, f, pol_index][:, ii, jj]
    model = arrays["model"][window, f, pol_index][:, ii, jj]
    gain = arrays["gains"][f, pol_index]
    predicted = model * gain[:, ii] * gain[:, jj].conj()
    residual = np.angle(data * predicted.conj())
    return residual, ii, jj, f


def main(nominal: Path, corrected: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    old, old_report = load(nominal)
    new, new_report = load(corrected)
    if not np.array_equal(old["frequencies_hz"], new["frequencies_hz"]):
        raise ValueError("frequency axes disagree")
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    results = {}
    for pi, pol in enumerate(("XX", "YY")):
        common_good = old["good"][:, pi] & new["good"][:, pi]
        def excluded(report):
            return (set(report["excluded_antennas"])
                    | {entry["antenna_id"] for entry in report["excluded_receptors"]
                       if entry["polarization"] == pol})
        common_excluded = excluded(old_report) | excluded(new_report)
        old_phase, ii, jj, freq = phases(old, old_report, pi, 2,
                                         common_good, common_excluded)
        new_phase, _, _, _ = phases(new, new_report, pi, 2,
                                    common_good, common_excluded)
        pair_rows = {}
        for label, phase in (("nominal", old_phase), ("swap_8_12", new_phase)):
            unit = np.exp(1j * phase)
            mean = np.angle(unit.mean())
            centred = np.angle(unit * np.exp(-1j*mean))
            pair_rows[label] = {
                "late_circular_mean_deg": float(np.rad2deg(mean)),
                "late_circular_std_deg": float(np.rad2deg(
                    np.sqrt(-2*np.log(abs(unit.mean()))))),
                "late_fraction_abs_phase_over_20deg": float(np.mean(
                    abs(centred) > np.deg2rad(20))),
                "late_abs_phase_q95_deg": float(np.rad2deg(
                    np.percentile(abs(centred), 95))),
                "baselines": int(len(ii)),
                "frequencies": int(len(freq)),
            }
            axes[0, pi].hist(np.rad2deg(centred).ravel(), bins=np.arange(-90,91,3),
                             density=True, histtype="step", linewidth=2,
                             label=("nominal" if label == "nominal"
                                    else "swap 8/12"))
        axes[0, pi].set_xlim(-65, 65)
        axes[0, pi].set_yscale("log")
        axes[0, pi].set_title(f"{pol}: late-window residual phase")
        axes[0, pi].set_xlabel("Residual phase (degrees)")
        axes[0, pi].set_ylabel("Density")
        axes[0, pi].legend()
        results[pol] = pair_rows
        if pol == "YY":
            old_tail = np.mean(abs(old_phase) > np.deg2rad(20), axis=0)
            new_tail = np.mean(abs(new_phase) > np.deg2rad(20), axis=0)
            ant_ids = np.arange(1, 37)
            old_ant, new_ant = [], []
            for ant in range(36):
                use = (ii == ant) | (jj == ant)
                old_ant.append(float(np.mean(old_tail[use])) if np.any(use) else np.nan)
                new_ant.append(float(np.mean(new_tail[use])) if np.any(use) else np.nan)
            x = np.arange(36)
            axes[1, 0].bar(x-.2, old_ant, width=.4, label="nominal", alpha=.85)
            axes[1, 0].bar(x+.2, new_ant, width=.4, label="swap 8/12", alpha=.85)
            axes[1, 0].set_xticks(x, ant_ids, fontsize=7)
            axes[1, 0].set_xlabel("Antenna ID")
            axes[1, 0].set_ylabel("Fraction of phase errors >20°")
            axes[1, 0].set_title("YY late-window baseline tails by antenna")
            axes[1, 0].legend()
            for ant in (8, 12):
                axes[1, 0].axvspan(ant-1-.5, ant-1+.5, color="gold", alpha=.18)
            old_f = np.mean(abs(old_phase) > np.deg2rad(20), axis=1)
            new_f = np.mean(abs(new_phase) > np.deg2rad(20), axis=1)
            hz = old["frequencies_hz"][freq] / 1e6
            axes[1, 1].plot(hz, old_f, lw=1.3, label="nominal")
            axes[1, 1].plot(hz, new_f, lw=1.3, label="swap 8/12")
            axes[1, 1].set_xlabel("Fine frequency (MHz)")
            axes[1, 1].set_ylabel("Fraction of phase errors >20°")
            axes[1, 1].set_title("YY late-window frequency structure")
            axes[1, 1].legend()
            results["YY"]["antenna_tail_fraction"] = {
                str(ant): {"nominal": old_ant[ant-1],
                           "swap_8_12": new_ant[ant-1]}
                for ant in (8, 12)}
    fig.suptitle("Sun phase residuals on common baselines and accepted bins; "
                 "late window is independent of the first-window gain fit")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    Path(str(output) + ".json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nominal", type=Path)
    parser.add_argument("corrected", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.nominal, args.corrected, args.output)

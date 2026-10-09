#!/usr/bin/env python3
"""Compare solar motion with frozen and reversed controls using a saved fit.

All predictions share exactly the same early-window visibility prediction.
No late-window gain adjustment is made. Channel-selection effects and
same-board versus cross-board baselines are reported explicitly.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def scores(data, prediction, mask):
    d, p = data[:, mask], prediction[:, mask]
    numerator = np.sum((p.conj() * d).real, axis=1)
    denominator = np.sqrt(np.sum(abs(p) ** 2, axis=1) *
                          np.sum(abs(d) ** 2, axis=1))
    return np.divide(numerator, denominator,
                     out=np.full(len(d), np.nan), where=denominator > 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("validation", type=Path)
    ap.add_argument("config", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.validation) as z:
        data, model = z["data"], z["model"]
        gain, metrics = z["gains"], z["metrics"]
        good, selected = z["good"], z["selected"]
        held = z["withheld"]
        freq, antenna, times = z["frequencies_hz"], z["antenna_order"], z["times_jd"]
    conf = json.loads(args.config.read_text())
    board_of = {int(a): int(b) for b, ants in conf["antenna_positions"].items()
                for a in ants}
    board = np.array([board_of[int(a)] for a in antenna])
    same = board[:, None] == board[None, :]
    masks = {"all": selected, "same_fpga": selected & same,
             "cross_fpga": selected & ~same, "withheld_baselines": held}
    names = ("Sun", "stationary", "reverse_solar_motion")
    result = {name: np.empty((2, 2, len(freq), len(masks))) for name in names}
    reproduction_error = 0.
    for pi in range(2):
        product = gain[:, pi, :, None] * gain[:, pi, None, :].conj()
        early = model[0, :, pi] * product
        for ti, wi in enumerate((1, 2)):
            # Reverse the solar phase evolution while preserving the early
            # prediction and the same projected disk-amplitude evolution.
            ratio = np.divide(model[wi, :, pi], model[0, :, pi],
                              out=np.ones_like(early),
                              where=abs(model[0, :, pi]) > 1e-12)
            predictions = {"Sun": early * ratio, "stationary": early,
                           "reverse_solar_motion": early * ratio.conj()}
            for name, prediction in predictions.items():
                for mi, mask in enumerate(masks.values()):
                    result[name][pi, ti, :, mi] = scores(data[wi, :, pi], prediction, mask)
            reproduction_error = max(reproduction_error, float(np.nanmax(abs(
                result["Sun"][pi, ti, :, 0] - metrics[:, pi, wi + 1]))))
    assert reproduction_error < 1e-10, reproduction_error
    bands = {"full_band": np.ones(len(freq), bool),
             "below_324_MHz": freq < 324e6,
             "at_or_above_324_MHz": freq >= 324e6}
    summary = {}
    for pi, pol in enumerate(("XX", "YY")):
        summary[pol] = {}
        selections = {"all_channels": np.ones(len(freq), bool),
                      "early_withheld_score_above_0.55": metrics[:, pi, 1] > .55,
                      "previously_accepted_using_late_data": good[:, pi]}
        for selection, keep in selections.items():
            summary[pol][selection] = {}
            for band, band_keep in bands.items():
                use = keep & band_keep
                row = {"channels": int(use.sum())}
                for ti, window in enumerate(("middle", "late")):
                    row[window] = {group: {name: float(np.median(
                        result[name][pi, ti, use, mi])) if use.any() else None
                        for name in names} for mi, group in enumerate(masks)}
                summary[pol][selection][band] = row
    report = {"validation": str(args.validation),
              "method": "Same early gain prediction; compare solar evolution, frozen early geometry, and reversed solar phase evolution; no new gains fitted.",
              "windows_jd": times.tolist(),
              "baseline_counts": {name: int(mask.sum()) for name, mask in masks.items()},
              "max_saved_metric_reproduction_error": reproduction_error,
              "summary": summary,
              "limitations": [
                  "Only the three saved short windows are tested.",
                  "The previously accepted mask used late data for selection; its reported scores are conditional, not a new blind test.",
                  "Stationary coherence does not distinguish terrestrial emission from coherent instrumental pickup.",
                  "An inadequate sky model can imitate gain drift."]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "scores.npz", frequency_hz=freq,
                        **{name: values for name, values in result.items()})
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True,
                             layout="constrained")
    for pi, pol in enumerate(("XX", "YY")):
        for ti, window in enumerate(("middle", "late")):
            ax = axes[pi, ti]
            for name in names:
                ax.plot(freq / 1e6, result[name][pi, ti, :, 0], lw=.7, label=name)
            ax.axvline(324, ls=":", color="0.5")
            ax.set(title=f"{pol}: {window} window, fixed early gains", ylim=(-1, 1.02),
                   xlabel="Frequency (MHz)", ylabel="Visibility correlation")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.savefig(args.output / "model_controls.png", dpi=150)
    print(json.dumps({pol: summary[pol]["all_channels"] for pol in summary}, indent=2))


if __name__ == "__main__":
    main()

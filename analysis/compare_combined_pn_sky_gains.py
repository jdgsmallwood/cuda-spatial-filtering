#!/usr/bin/env python3
"""Cross-check combined PN correction against independent sky gain phases."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def phase_steps(sky, pn, good, sign, kind):
    corrected = np.angle(np.exp(1j * (sky + sign * pn)))
    if kind == "within_coarse":
        difference = np.angle(np.exp(1j * (corrected[:, 1:] - corrected[:, :-1])))
        valid = good[:, 1:] & good[:, :-1]
    else:
        difference = np.angle(np.exp(1j * (corrected[1:, 0] - corrected[:-1, -1])))
        valid = good[1:, 0] & good[:-1, -1]
    return abs(difference[valid])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sky_candidates", type=Path)
    ap.add_argument("pn_candidate", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--mapping", type=Path, default=Path("stream_antenna_map.json"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.sky_candidates) as z:
        ids = z["antenna_ids"].astype(int)
        gains = z["gains"].reshape(40, 27, -1)
    with np.load(args.pn_candidate) as z:
        phase = z["phase_rad"]
        stable = z["stable"]
    mapping = json.loads(args.mapping.read_text())["fpgas"]
    raw = {int(stream["antenna_id"]): int(board) * 20 + i
           for board, fpga in mapping.items()
           for i, stream in enumerate(fpga["streams"])
           if stream["antenna_id"] > 0 and stream["polarization"] == 0}
    ref_sky = int(np.flatnonzero(ids == 1)[0])
    ref_raw = raw[1]
    arrays = {kind: {name: [] for name in ("sky_only", "pn_then_sky", "opposite_pn")}
              for kind in ("within_coarse", "between_coarse")}
    ant_count = 0
    for si, ant in enumerate(ids):
        ant = int(ant)
        if ant in (1, 8, 12, 14) or ant not in raw:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            sky = np.angle(gains[:, :, si] / gains[:, :, ref_sky])
        pn = phase[:, :, raw[ant]] - phase[:, :, ref_raw]
        quality = np.isfinite(sky) & stable[:, raw[ant], None] & stable[:, ref_raw, None]
        ant_count += 1
        for kind in arrays:
            for name, sign in (("sky_only", 0), ("pn_then_sky", -1),
                               ("opposite_pn", +1)):
                arrays[kind][name].append(phase_steps(sky, pn, quality, sign, kind))
    merged = {kind: {name: np.concatenate(chunks) for name, chunks in candidates.items()}
              for kind, candidates in arrays.items()}
    metrics = {kind: {name: {"n_steps": len(value),
                             "median_abs_step_rad": float(np.median(value)),
                             "p90_abs_step_rad": float(np.quantile(value, .9))}
                      for name, value in candidates.items()}
               for kind, candidates in merged.items()}
    report = {
        "sky_candidates": str(args.sky_candidates), "pn_candidate": str(args.pn_candidate),
        "n_antenna_hands": ant_count, "excluded_antenna_ids": [1, 8, 12, 14],
        "same_quality_cells_for_all_candidates": True,
        "metrics": metrics,
        "limitation": "Sky gains were fitted independently per fine bin from a nighttime model with RFI and modest visibility correlation. Phase smoothness is a cross-check, not a held-out visibility or image-quality improvement."
    }
    (args.output / "pn_combined_sky_gain_check.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, kind in zip(axes, merged):
        for name, value in merged[kind].items():
            ordered = np.sort(value)
            ax.plot(ordered, np.arange(1, len(value) + 1) / len(value), label=name)
        ax.set(title=kind.replace("_", " "), xlabel="Absolute adjacent phase step (rad)",
               ylabel="Fraction of steps", xlim=(0, np.pi), ylim=(0, 1))
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(args.output / "pn_combined_sky_gain_check.png", dpi=170)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

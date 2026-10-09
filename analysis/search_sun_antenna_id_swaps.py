#!/usr/bin/env python3
"""Search whole-antenna ID swaps using held-time solar phase evolution."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def unit(values):
    return values / np.maximum(np.abs(values), 1e-30)


def main(validation: Path, report_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    report = json.loads(report_path.read_text())
    with np.load(validation) as saved:
        data = saved["data"]
        model = saved["model"]
        gains = saved["gains"]
        accepted = saved["good"]
        selected = saved["selected"]
        withheld = saved["withheld"]
        order = saved["antenna_order"]
    if not np.array_equal(order, np.arange(1, 37)):
        raise ValueError("candidate indexing assumes unswapped capture IDs")
    upper = np.triu(np.ones((36, 36), bool), 1)
    samples = {}
    for pi, pol in enumerate(("XX", "YY")):
        excluded = set(report["excluded_antennas"])
        excluded |= {entry["antenna_id"] for entry in report["excluded_receptors"]
                     if entry["polarization"] == pol}
        allowed = ~np.isin(order, list(excluded))
        ii, jj = np.where(selected & upper & allowed[:, None] & allowed[None, :])
        freq = np.flatnonzero(accepted[:, pi])
        g = gains[freq, pi]
        # A point-source phase factor per antenna. The disk visibility is
        # positive on this short array, so its phase has the same factorization.
        direction = unit(model[:, :, pi, :, 0][:, freq, :])
        evolution = direction / direction[0]
        error = []
        for wi in (1, 2):
            prediction = (model[wi, freq, pi][:, ii, jj]
                          * g[:, ii] * g[:, jj].conj())
            error.append(unit(data[wi, freq, pi][:, ii, jj]
                              * prediction.conj()))
        samples[pol] = {"ii": ii, "jj": jj, "freq": freq,
                        "evolution": evolution, "error": np.stack(error),
                        "withheld": withheld[ii, jj], "allowed": allowed}
    candidates = []
    for a, b in itertools.combinations(range(36), 2):
        row = {"pair": [a+1, b+1]}
        for pol, sample in samples.items():
            if not (sample["allowed"][a] and sample["allowed"][b]):
                continue
            ii, jj = sample["ii"], sample["jj"]
            touched = (ii == a) | (jj == a) | (ii == b) | (jj == b)
            if touched.sum() < 8:
                continue
            original = sample["error"][:, :, touched]
            ii, jj = ii[touched], jj[touched]
            evolution = sample["evolution"]
            change_a = evolution[1:, :, b] / evolution[1:, :, a]
            change_b = evolution[1:, :, a] / evolution[1:, :, b]
            left = np.where(ii[None, None, :] == a, change_a[:, :, None],
                            np.where(ii[None, None, :] == b,
                                     change_b[:, :, None], 1))
            right = np.where(jj[None, None, :] == a, change_a[:, :, None],
                             np.where(jj[None, None, :] == b,
                                      change_b[:, :, None], 1))
            alternative = original * left.conj() * right
            held = sample["withheld"][touched]
            for split, mask in (("train", ~held), ("withheld", held)):
                if mask.sum() < 3:
                    continue
                current = float(np.mean(original[:, :, mask].real))
                swapped = float(np.mean(alternative[:, :, mask].real))
                row[f"{pol}_{split}"] = {"baseline_count": int(mask.sum()),
                                        "current_cosine": current,
                                        "swap_cosine": swapped,
                                        "improvement": swapped-current}
        if row.keys() != {"pair"}:
            scores = [v["improvement"] for k,v in row.items()
                      if k.endswith("_train")]
            held_scores = [v["improvement"] for k,v in row.items()
                           if k.endswith("_withheld")]
            row["mean_train_improvement"] = float(np.mean(scores)) if scores else None
            row["mean_withheld_improvement"] = (float(np.mean(held_scores))
                                                if held_scores else None)
            candidates.append(row)
    candidates.sort(key=lambda row: row["mean_train_improvement"], reverse=True)
    output.write_text(json.dumps({"method": "Re-map geometric solar phase "
                         "evolution after keeping the first-window gain fixed; "
                         "score middle and late windows on training and held "
                         "baselines", "candidates": candidates}, indent=2) + "\n")
    for row in candidates[:12]:
        print(row["pair"], "train", round(row["mean_train_improvement"], 4),
              "withheld", round(row["mean_withheld_improvement"], 4)
              if row["mean_withheld_improvement"] is not None else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.validation, args.report, args.output)

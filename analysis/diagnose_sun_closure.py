#!/usr/bin/env python3
"""Measure antenna-gain-invariant closure error against a moving-Sun model.

The phase of (Vab/Mab)(Vbc/Mbc)conj(Vac/Mac) cancels *every* antenna phase,
including FPGA offsets, cable delays and fine-bin slopes. Nonzero closure
therefore localizes trouble to additive signals, incomplete sky/beam models,
nonclosing baseline effects or low SNR before another gain solve is attempted.
"""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

from casacore.tables import table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def select_triangles(antenna_a, antenna_b, baseline_m, board_for_antenna,
                     min_baseline_m, triangles_per_group, seed):
    lookup = {(int(a), int(b)): row for row, (a, b) in
              enumerate(zip(antenna_a, antenna_b))}
    groups = defaultdict(list)
    for a, b, c in combinations(sorted(board_for_antenna), 3):
        rows = (lookup.get((a, b)), lookup.get((b, c)), lookup.get((a, c)))
        if any(row is None or baseline_m[row] < min_baseline_m for row in rows):
            continue
        boards = len({board_for_antenna[a], board_for_antenna[b],
                      board_for_antenna[c]})
        group = ("one_board" if boards == 1 else
                 "two_boards" if boards == 2 else "three_boards")
        groups[group].append((a, b, c, *rows))
    rng = np.random.default_rng(seed)
    chosen = {}
    for name in ("one_board", "two_boards", "three_boards"):
        population = groups[name]
        count = min(len(population), triangles_per_group)
        indices = rng.choice(len(population), count, replace=False)
        chosen[name] = np.asarray([population[i] for i in indices], int)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_ms", type=Path)
    parser.add_argument("model_ms", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-baseline-m", type=float, default=7.0)
    parser.add_argument("--triangles-per-group", type=int, default=200)
    parser.add_argument("--max-time-bins", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".png").exists():
        parser.error("refusing to overwrite output")
    if args.triangles_per_group < 1 or args.min_baseline_m < 0:
        parser.error("invalid triangle count or baseline cut")
    config = json.loads(args.config.read_text())
    board = {int(ant)-1: int(fpga)
             for fpga, antennas in config["antenna_positions"].items()
             for ant in antennas}
    with table(str(args.data_ms), ack=False) as data, \
         table(str(args.model_ms), ack=False) as model, \
         table(str(args.model_ms / "SPECTRAL_WINDOW"), ack=False) as spectral:
        if data.nrows() != model.nrows() or data.nrows() % 630:
            raise ValueError("Measurement Set rows differ or are not 630-baseline bins")
        for column in ("TIME", "ANTENNA1", "ANTENNA2"):
            if not np.array_equal(data.getcol(column), model.getcol(column)):
                raise ValueError(f"Measurement Set {column} differs")
        ntime = data.nrows() // 630
        if args.max_time_bins:
            ntime = min(ntime, args.max_time_bins)
        freq = spectral.getcell("CHAN_FREQ", 0)
        a = data.getcol("ANTENNA1", 0, 630)
        b = data.getcol("ANTENNA2", 0, 630)
        length = np.linalg.norm(data.getcol("UVW", 0, 630), axis=1)
        chosen = select_triangles(a, b, length, board, args.min_baseline_m,
                                  args.triangles_per_group, seed=7)
        groups = list(chosen)
        counts = np.zeros((len(groups), len(freq), 2), np.int64)
        cosine_sum = np.zeros_like(counts, float)
        high_error_count = np.zeros_like(counts)
        histogram = np.zeros((len(groups), 2, 36), np.int64)
        time_cosine = np.full((ntime, len(groups), 2), np.nan)
        for ti in range(ntime):
            start = ti * 630
            vis = data.getcol("DATA", start, 630)
            pred = model.getcol("MODEL_DATA", start, 630)
            flag = data.getcol("FLAG", start, 630) | \
                   model.getcol("FLAG", start, 630)
            product = vis * pred.conj()
            amplitude = abs(product)
            valid = (~flag) & np.isfinite(product) & (amplitude > 0) & \
                    (abs(pred) > 0.5)
            unit = np.divide(product, amplitude, out=np.zeros_like(product),
                             where=valid)
            for gi, name in enumerate(groups):
                tri = chosen[name]
                ab, bc, ac = tri[:, 3], tri[:, 4], tri[:, 5]
                good = valid[ab] & valid[bc] & valid[ac]
                closure = unit[ab] * unit[bc] * unit[ac].conj()
                cosine = closure.real
                counts[gi] += good.sum(axis=0)
                cosine_sum[gi] += np.where(good, cosine, 0).sum(axis=0)
                angle = abs(np.angle(closure))
                high_error_count[gi] += (good & (angle > np.deg2rad(20))).sum(axis=0)
                for pi in range(2):
                    histogram[gi, pi] += np.histogram(
                        np.rad2deg(angle[:, :, pi][good[:, :, pi]]),
                        bins=np.linspace(0, 180, 37))[0]
                    count = good[:, :, pi].sum()
                    if count:
                        time_cosine[ti, gi, pi] = cosine[:, :, pi][good[:, :, pi]].mean()
            if ti % 20 == 0:
                print(f"closure bin {ti+1}/{ntime}", flush=True)
    mean_cos = np.divide(cosine_sum, counts,
                         out=np.full_like(cosine_sum, np.nan), where=counts > 0)
    frac_high = np.divide(high_error_count, counts,
                          out=np.full_like(cosine_sum, np.nan), where=counts > 0)
    bands = {"lower_312_317p5": (312e6, 317.5e6),
             "notch_318p4_319p7": (318.4e6, 319.7e6),
             "middle_320_324": (320e6, 324e6),
             "upper_334_343p5": (334e6, 343.5e6)}
    def band_score(group_index, mask, pol_index):
        n = int(counts[group_index, mask, pol_index].sum())
        return {
            "mean_closure_cosine":
                float(cosine_sum[group_index, mask, pol_index].sum() / n)
                if n else None,
            "fraction_above_20_deg":
                float(high_error_count[group_index, mask, pol_index].sum() / n)
                if n else None,
            "valid_cells": n,
        }

    summary = {}
    for gi, name in enumerate(groups):
        summary[name] = {
            "triangles": len(chosen[name]),
            "bands": {band: {pol: band_score(gi, mask, pi)
                            for pi, pol in enumerate(("XX", "YY"))}
            for band, (lo, hi) in bands.items()
            for mask in [((freq >= lo) & (freq < hi))]},
        }
    report = {
        "data_ms": str(args.data_ms), "model_ms": str(args.model_ms),
        "integration_bins": ntime,
        "min_baseline_m": args.min_baseline_m,
        "model_amplitude_gate": 0.5,
        "triangle_groups": summary,
        "interpretation": "Closure error is invariant to antenna-based phase, delay and gain corrections. An error here cannot be fixed by another antenna-delay fit alone.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output.with_suffix(".npz"), frequency_hz=freq,
                        groups=np.asarray(groups), counts=counts,
                        mean_cosine=mean_cos, fraction_above_20_deg=frac_high,
                        histogram=histogram, time_cosine=time_cosine)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), layout="constrained")
    for pi, ax in enumerate(axes):
        for gi, name in enumerate(groups):
            ax.plot(freq / 1e6, mean_cos[gi, :, pi], lw=1, label=name)
        ax.set(title=("XX", "YY")[pi] + " Sun-model closure",
               xlabel="Frequency (MHz)", ylabel="Mean cosine of closure error",
               ylim=(-1, 1))
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(args.output.with_suffix(".png"), dpi=160)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

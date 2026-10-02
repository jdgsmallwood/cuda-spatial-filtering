#!/usr/bin/env python3
"""Fit FPGA sample offsets from all pairwise packet-correlation delays.

Input JSON is a list of records with ``first``, ``second``, and ``delay``.
Each delay is the second FPGA's offset minus the first FPGA's offset, in
samples. All six pairs are required for the four-FPGA array. A failed closure
check prevents a misleading delay file from being written.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def fit_delays(pairs, reference=1, max_residual=2.0):
    nodes = set(range(4))
    expected = {(i, j) for i in nodes for j in nodes if i < j}
    found = {(int(p["first"]), int(p["second"])) for p in pairs}
    if len(pairs) != 6 or found != expected:
        raise ValueError("supply exactly one measurement for each of the six FPGA pairs")
    if reference not in nodes:
        raise ValueError("reference FPGA must be 0, 1, 2, or 3")

    free = [node for node in sorted(nodes) if node != reference]
    matrix = np.zeros((6, 3), dtype=float)
    measured = np.empty(6, dtype=float)
    for row, pair in enumerate(pairs):
        first, second = int(pair["first"]), int(pair["second"])
        if first != reference:
            matrix[row, free.index(first)] = -1
        if second != reference:
            matrix[row, free.index(second)] = 1
        measured[row] = float(pair["delay"])
    if not np.isfinite(measured).all() or np.any(measured != np.rint(measured)):
        raise ValueError("packet-correlation delays must be finite integer samples")
    if not np.isfinite(max_residual) or max_residual < 0:
        raise ValueError("max_residual must be finite and nonnegative")
    # Integer correlation peaks are exact sample hypotheses. Least-squares
    # averaging followed by rounding can move TWO FPGA offsets by one sample
    # when just ONE of the six pair peaks is wrong by two samples. Require a
    # unique integer solution supported by at least five edges instead.
    candidates = {}
    for omitted in range(6):
        keep = np.arange(6) != omitted
        fitted, *_ = np.linalg.lstsq(matrix[keep], measured[keep], rcond=None)
        integer = np.rint(fitted).astype(np.int64)
        residual = matrix @ integer - measured
        if np.count_nonzero(abs(residual) < 1e-6) >= 5:
            candidates[tuple(integer)] = residual
    if len(candidates) != 1:
        raise ValueError("pairwise delays fail closure: no unique solution supported by five edges")
    offsets, residuals = next(iter(candidates.items()))
    if np.max(np.abs(residuals)) > max_residual:
        detail = ", ".join(f"{p['first']}-{p['second']}:{r:+.1f}"
                           for p,r in zip(pairs,residuals))
        raise ValueError(f"pairwise delays fail closure ({detail})")
    result = {str(node): 0 if node == reference else int(offsets[free.index(node)])
              for node in sorted(nodes)}
    return result, residuals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path, help="JSON list of all six pairwise delays")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=int, default=1)
    parser.add_argument("--max-residual", type=float, default=2.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    pairs = json.loads(args.pairs.read_text())
    delays, residuals = fit_delays(pairs, args.reference, args.max_residual)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(delays, indent=2) + "\n")
    for pair, residual in zip(pairs, residuals):
        print(f"{pair['first']}-{pair['second']}: residual {residual:+.3f} samples")
    print(f"wrote {args.output}: {delays}")


if __name__ == "__main__":
    main()

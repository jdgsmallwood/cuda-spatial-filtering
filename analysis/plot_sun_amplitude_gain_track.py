#!/usr/bin/env python3
"""Show per-time Sun amplitude gains grouped by FPGA and receptor hand."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(fit_path: Path, stream_map: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    with np.load(fit_path) as fit:
        gains = fit["gain"][:-1]
        supported = fit["supported"][:-1]
    mapping = json.loads(stream_map.read_text())["fpgas"]
    order = []
    separators = []
    labels = []
    for board, record in sorted(mapping.items(), key=lambda item: int(item[0])):
        ants = sorted({entry["antenna_id"] for entry in record["streams"]
                       if entry["antenna_id"] > 0})
        order.extend(ant - 1 for ant in ants)
        labels.extend(f"{ant:02d}" for ant in ants)
        separators.append(len(order) - .5)
    if sorted(order) != list(range(36)):
        raise ValueError("stream map does not contain one of each antenna")
    fig, axes = plt.subplots(2, 1, figsize=(17, 10), sharex=True,
                             constrained_layout=True)
    for pi, (ax, pol) in enumerate(zip(axes, ("XX", "YY"))):
        values = gains[:, pi, order].T
        mask = supported[:, pi, order].T
        values = np.ma.masked_where(~mask, values)
        cm = plt.get_cmap("coolwarm").copy()
        cm.set_bad("#dddddd")
        im = ax.imshow(values, aspect="auto", origin="lower", cmap=cm,
                       vmin=.75, vmax=1.25,
                       extent=[0, (len(gains)-1)*10/60, -.5, 35.5])
        for sep in separators[:-1]:
            ax.axhline(sep, color="black", lw=1)
        ax.set_yticks(range(36), labels)
        ax.set_ylabel(f"{pol} antenna ID, grouped by FPGA")
        ax.set_title(f"{pol}: real Sun amplitude gain relative to median live antenna")
        fig.colorbar(im, ax=ax, label="voltage amplitude gain")
    axes[-1].set_xlabel("Minutes from start (10 s bin)")
    fig.savefig(output, dpi=150)
    print(f"wrote {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fit", type=Path)
    p.add_argument("stream_map", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    main(a.fit, a.stream_map, a.output)

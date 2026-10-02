#!/usr/bin/env python3
"""Summarize CASA K/B/G solution coverage by antenna and receptor."""

import argparse
from pathlib import Path

import numpy as np
from casacore.tables import table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_directory", type=Path)
    args = parser.parse_args()

    for name in ("delay.K", "bandpass.B", "phase.G", "gain.G"):
        path = args.calibration_directory / name
        if not path.is_dir():
            print(f"{name}: MISSING")
            continue
        with table(str(path), ack=False) as tab:
            flags = np.asarray(tab.getcol("FLAG"), dtype=bool)
            antennas = np.asarray(tab.getcol("ANTENNA1"), dtype=int)
        print(
            f"{name}: {np.count_nonzero(~flags)}/{flags.size} unflagged "
            f"({100.0 * np.mean(~flags):.2f}%)"
        )
        weak = []
        for antenna in sorted(set(antennas)):
            rows = antennas == antenna
            for pol, label in enumerate(("XX", "YY")):
                count = int(np.count_nonzero(~flags[rows, :, pol]))
                total = int(flags[rows, :, pol].size)
                if count == 0 or count < total / 2:
                    weak.append(f"A{antenna + 1}:{label}={count}/{total}")
        if weak:
            print("  zero/<50% receptors: " + ", ".join(weak))


if __name__ == "__main__":
    main()

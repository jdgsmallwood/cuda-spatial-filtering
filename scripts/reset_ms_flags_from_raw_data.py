#!/usr/bin/env python3
"""Restore an MS validity mask from raw DATA after CASA ``calflag`` trials.

The commissioning exporter represents unsupported raw samples as non-finite
DATA, so this recreates its original raw mask without relying on a CASA flag
backup. Run this only on a copied MS intended for a fresh calibration trial.
"""

import argparse
from pathlib import Path

import numpy as np
from casacore.tables import table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement_set", type=Path)
    args = parser.parse_args()
    with table(str(args.measurement_set), readonly=False, ack=False) as tab:
        data = tab.getcol("DATA")
        flags = ~np.isfinite(data)
        tab.putcol("FLAG", flags)
        tab.putcol("FLAG_ROW", np.all(flags, axis=(1, 2)))
        tab.flush()
    print(
        f"restored {args.measurement_set}: "
        f"{np.count_nonzero(flags)}/{flags.size} raw invalid samples flagged"
    )


if __name__ == "__main__":
    main()

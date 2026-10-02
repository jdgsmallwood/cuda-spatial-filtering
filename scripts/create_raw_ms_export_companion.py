#!/usr/bin/env python3
"""Create the minimal legacy companion required by the raw-MS exporter.

This is deliberately *not* a calibration product.  The commissioning MS
exporter currently requires a ``*_calibrated.hdf5`` argument even when its raw
``DATA``/``MODEL_DATA`` branch is the only output used for a fresh CASA solve.
This helper supplies unity gains and empty legacy masks for that compatibility
branch.  CASA must solve the exported raw DATA; this file must never be
exported as beamforming weights.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--antennas", type=int, default=36)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    with h5py.File(args.capture, "r") as source:
        fine_channels = int(source["visibilities"].shape[1])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as output:
        calibration = output.create_group("calibration")
        calibration.attrs["method"] = "UNITY COMPATIBILITY ONLY; raw MS CASA solve input"
        calibration.attrs["swap_pol_antennas"] = np.empty(0, dtype=np.int32)
        for polarization in ("XX", "YY"):
            group = calibration.create_group(polarization)
            group.create_dataset(
                "gains_real", data=np.ones((fine_channels, args.antennas), np.float32)
            )
            group.create_dataset(
                "gains_imag", data=np.zeros((fine_channels, args.antennas), np.float32)
            )
            group.create_dataset("flagged_fine_channels", data=np.empty(0, np.int64))
            group.create_dataset("flagged_antennas", data=np.empty(0, np.int64))
            group.create_dataset("flagged_time_indices", data=np.empty(0, np.int64))
    print(f"wrote unity raw-MS compatibility companion: {args.output}")


if __name__ == "__main__":
    main()

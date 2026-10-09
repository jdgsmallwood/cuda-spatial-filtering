#!/usr/bin/env python3
"""Adapt a validated Sun phase fit to the commissioning native-MS exporter.

The output is an interchange file, not a new fit. Rejected frequencies and
receptors stay flagged; their placeholder gains must never be used for imaging.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from lambda_commissioning.utils import read_hdf5_time_freq


def export(capture: Path, validation: Path, report_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    report = json.loads(report_path.read_text())
    with np.load(validation) as fit:
        frequencies = fit["frequencies_hz"]
        good = fit["good"]
        order = fit["antenna_order"].astype(int)
        gains = fit["gains"]
    _, capture_frequencies = read_hdf5_time_freq(str(capture))
    if (frequencies.shape != capture_frequencies.shape
            or not np.allclose(frequencies, capture_frequencies, atol=1e-6, rtol=0)):
        raise ValueError("validated fine frequencies disagree with capture")
    # The native exporter requires bitwise equality. The calibration fit used
    # the exact runtime axis, whose floating-point evaluation differs by <1e-7 Hz.
    frequencies = capture_frequencies
    if (gains.shape != (len(frequencies), 2, 36)
            or good.shape != (len(frequencies), 2)
            or sorted(order.tolist()) != list(range(1, 37))):
        raise ValueError("unexpected Sun fit shape or antenna order")
    swaps = sorted({int(v) for v in report.get("swap_pol_antennas", [])})
    if any(v < 1 or v > 36 for v in swaps):
        raise ValueError("invalid X/Y swap antenna ID")
    if report.get("antenna_id_swaps"):
        raise ValueError("this adapter does not support antenna ID swaps")
    excluded_all = set(report.get("excluded_antennas", []))
    excluded = {pol: set(excluded_all) for pol in ("XX", "YY")}
    for receptor in report.get("excluded_receptors", []):
        excluded[receptor["polarization"]].add(int(receptor["antenna_id"]))
    with h5py.File(output, "x") as dest:
        calibration = dest.create_group("calibration")
        calibration.attrs["swap_pol_antennas"] = np.array(swaps, dtype=np.int32)
        calibration.attrs["source_validation"] = str(validation.resolve())
        calibration.attrs["kind"] = "phase_only_validated_sun"
        for pi, pol in enumerate(("XX", "YY")):
            group = calibration.create_group(pol)
            phase = np.ones((len(frequencies), 36), np.complex64)
            measured = gains[:, pi]
            unit = np.divide(measured, np.abs(measured),
                             out=np.ones_like(measured),
                             where=np.isfinite(measured) & (np.abs(measured) > 0))
            phase[:, order - 1] = unit.astype(np.complex64)
            valid = good[:, pi] & np.isfinite(measured).all(axis=1)
            group.create_dataset("fine_frequencies_hz", data=frequencies)
            group.create_dataset("gains_real", data=phase.real)
            group.create_dataset("gains_imag", data=phase.imag)
            group.create_dataset("flagged_fine_channels", data=np.flatnonzero(~valid))
            group.create_dataset("flagged_antennas", data=np.array(sorted(excluded[pol]), dtype=np.int32))
            group.create_dataset("flagged_time_indices", data=np.empty(0, dtype=np.int32))
    print(f"wrote {output}: XX {good[:, 0].sum()}, YY {good[:, 1].sum()} accepted processing bins")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.capture, args.validation, args.report, args.output)

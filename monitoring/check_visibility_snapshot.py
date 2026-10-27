#!/usr/bin/env python3
"""Check a retained Redis snapshot against its matching HDF5 integration."""
import argparse
import json
from pathlib import Path
import re

import h5py
import numpy as np

from check_stack import redis_command


def check(path, port=6380, run=None):
    run = run or redis_command(port, "GET", "starspin:latest")
    prefix = "starspin:" + run
    rows = json.loads(redis_command(port, "JSON.GET", prefix + ":status", "."))
    status = {row["field"]: row["value"] for row in rows}
    integration = json.loads(status["last_integration"])
    count = 0
    with h5py.File(path) as file:
        ranges = file["vis_seq_nums"][:]
        matches = np.flatnonzero((ranges[:, 0] == integration["first_sample"]) &
                                 (ranges[:, 1] == integration["last_packet_sample"]))
        assert len(matches) == 1, "Snapshot integration not present in HDF5"
        data = file["visibilities"][matches[0]].astype(np.float64)
        mapping = {row["receiver"]: row["valid"] for row in json.loads(status["mapping"])}
        for kind in ("inputs", "baselines"):
            for name in redis_command(port, "SMEMBERS", prefix + ":" + kind) or []:
                match = re.fullmatch(r"rx(\d+)(?:_rx(\d+))?_p(\d+)", name)
                assert match, name
                first, second, pol = match.groups()
                first, second, pol = int(first), int(second or first), int(pol)
                baseline = second * (second + 1) // 2 + first
                values = data[:, baseline, pol, pol, 0] + 1j * data[:, baseline, pol, pol, 1]
                stride = (len(values) + 255) // 256
                averaged = np.array([np.mean(values[i:i + stride]) for i in range(0, len(values), stride)])
                spectrum = json.loads(redis_command(port, "JSON.GET", prefix + ":spectrum:" + name, "."))
                valid = mapping[first] and mapping[second] and integration["total_packets"] > integration["missing_packets"]
                amplitude = np.abs(averaged) if valid else np.full(len(averaged), np.nan)
                phase = np.where(amplitude > 0, np.angle(averaged, deg=True), np.nan)
                np.testing.assert_allclose([float(row["amplitude"]) for row in spectrum], amplitude, rtol=1e-12, atol=1e-8)
                np.testing.assert_allclose([float(row["phase_deg"]) for row in spectrum], phase, rtol=1e-12, atol=1e-8)
                if kind == "inputs":
                    power = averaged.real if valid else np.full(len(averaged), np.nan)
                    np.testing.assert_allclose([float(row["power"]) for row in spectrum], power, rtol=1e-12, atol=1e-8)
                count += 1
    result = {"run": run, "first_sample": integration["first_sample"], "spectra_matched_to_hdf5": count,
              "redis_failures": redis_command(port, "TS.GET", prefix + ":redis_failures")}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("--redis-port", type=int, default=6380)
    parser.add_argument("--run")
    args = parser.parse_args()
    check(args.hdf5, args.redis_port, args.run)

#!/usr/bin/env python3
"""Local synthetic replay benchmark, including paired complex-voltage writing.

Dimensions must match the selected build (10 receivers per FPGA, 32 fine bins).
Results measure this replay path, not NIC capacity or stable-storage fsync speed.
"""
import argparse
import csv
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fpgas", type=int, default=1, choices=(1, 2, 3, 4))
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--weights-only", action="store_true", help="Run only the weights-only baseline")
    parser.add_argument("--gpu-buffers", type=int, help="Override the app's GPU buffer count")
    args = parser.parse_args()
    if args.repeats < 1 or args.seconds <= 0 or args.channels < 1:
        parser.error("repeats and seconds must be positive")
    binary = args.binary.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from generate_codif_fixture import fixture_bytes
    cases = [("weights-only", 0, False, False)]
    cases += [(mode, rank, protected, False) for mode in ("coarse", "fine")
              for rank in (0, 1, 2, 4) for protected in (False, True)]
    cases += [(mode, 2, True, True) for mode in ("coarse", "fine")]
    if args.weights_only:
        cases = cases[:1]
    with tempfile.TemporaryDirectory(prefix="adaptive-benchmark-", dir=args.output) as directory:
        root = Path(directory).resolve()
        (root / "input.pcap").write_bytes(fixture_bytes(frames=1024, channels=args.channels, fpgas=args.fpgas))
        (root / "config.json").write_text(json.dumps({
            "frequency_plan": {"base_frequency_hz": 0, "channel_bandwidth_hz": 781250},
            "array_location": {"latitude_deg": -30.31, "longitude_deg": 149.55, "height_m": 240},
            "network_interfaces": {str(f): "lo" for f in range(args.fpgas)},
            "antenna_positions": {str(f): {str(f * 10 + i): {"east": float(f * 10 + i), "north": 0, "up": 0} for i in range(10)} for f in range(args.fpgas)}}))
        (root / "delays.json").write_text(json.dumps({str(f): 0 for f in range(args.fpgas)}))
        for repeat in range(args.repeats):
            order = list(cases)
            random.Random(41 + repeat).shuffle(order)
            for mode, rank, protected, diagnostic in order:
                beam, vis = root / "beam.h5", root / "vis.h5"
                command = [str(binary), "--decomposition", mode, "--null-rank", str(rank),
                           "--calibration", "unity", "--sun-protection", "rank-one" if protected else "off",
                           "--steering-utc", "2026-10-29T02:00:00Z", "--packet-format", "codif",
                           "-p", str(root / "input.pcap"), "--loop", "-i", ",".join(str(f) for f in range(args.fpgas)), "-f", "110",
                           "-c", str(root / "config.json"), "--delay-file", str(root / "delays.json"),
                           "--beam-output-file", str(beam), "--obs-length", str(args.seconds)]
                if diagnostic:
                    command += ["--record-visibilities", "--vis_output_file", str(vis)]
                if args.gpu_buffers is not None:
                    command += ["--gpu-buffers", str(args.gpu_buffers)]
                name = f"{repeat}-{mode}-r{rank}-p{int(protected)}-vis{int(diagnostic)}"
                with (args.output / (name + ".log")).open("w+") as log:
                    process = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=180)
                    log.seek(0)
                    text = log.read()
                if process.returncode:
                    raise RuntimeError(f"{name} failed; see its log")
                line = next(line for line in text.splitlines() if line.startswith("ADAPTIVE_BENCHMARK "))
                row = json.loads(line.split(" ", 1)[1])
                with h5py.File(beam) as file:
                    arrivals = file["arrivals"][:]
                    if arrivals.shape[1] != args.channels or arrivals.shape[-1] != args.fpgas:
                        raise RuntimeError("Benchmark dimensions do not match the executable")
                    if not np.all(np.count_nonzero(arrivals[:, :, 1:-1, :], axis=(0, 1, 2)) > 0):
                        raise RuntimeError("No successful input from one or more FPGA streams")
                    successful = int(np.count_nonzero(arrivals[:, :, 1:-1, :]))
                    if successful * 64 * 10 * 2 * 2 != row["successful_input_payload_bytes"]:
                        raise RuntimeError("Payload accounting disagrees with written arrival flags")
                    if file["beam_data"].shape[0] != row["pipeline_blocks"]:
                        raise RuntimeError("Not every processed block was written")
                    if not np.isfinite(file["beam_data"][:2].view(np.float16)).all():
                        raise RuntimeError("Non-finite voltages in output")
                row.update(repeat=repeat, fpgas=args.fpgas, coarse_channels=args.channels,
                           beam_file_bytes=beam.stat().st_size,
                           vis_file_bytes=vis.stat().st_size if diagnostic else 0)
                results.append(row)
                (args.output / "runs.json").write_text(json.dumps(results, indent=2) + "\n")
                print(name, f'{row["input_GB_per_s"]:.4f} GB/s', flush=True)
                beam.unlink()
                if diagnostic:
                    vis.unlink()
    with (args.output / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()

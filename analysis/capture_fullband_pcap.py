#!/usr/bin/env python3
"""Capture simultaneous unfiltered filterbank packets from all four FPGAs.

Use after the final FPGA tuning. This tool does not change any FPGA setting.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--packets-per-board", type=int, default=400_000)
    args = parser.parse_args()
    if args.packets_per_board < 12_000:
        parser.error("capture at least 12,000 packets per board")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    interfaces = json.loads(args.config.read_text())["network_interfaces"]
    args.output.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    processes = []
    failures = []
    try:
        for board in range(4):
            log = (args.output / f"dumpcap{board}.log").open("w")
            command = ["dumpcap", "-q", "-i", interfaces[str(board)],
                       "-f", "udp port 36001", "-c",
                       str(args.packets_per_board), "-w",
                       str(args.output / f"fpga{board}.pcapng")]
            process = subprocess.Popen(command, stdout=log, stderr=log)
            processes.append((board, process, log))
        for board, process, _ in processes:
            try:
                code = process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                process.kill()
                code = -1
            if code != 0:
                failures.append({"fpga": board, "exit_code": code})
    finally:
        for _, process, log in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
            log.close()
    report = {
        "start_utc": started,
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "interfaces": {str(i): interfaces[str(i)] for i in range(4)},
        "packets_per_board": args.packets_per_board,
        "filter": "udp port 36001",
        "tuning_changed": False,
        "failures": failures,
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if failures:
        raise RuntimeError(f"dumpcap failed on {failures}")


if __name__ == "__main__":
    main()

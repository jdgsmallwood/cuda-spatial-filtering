#!/usr/bin/env python3
"""Compare packet-counter epochs across FPGA PCAPs when sky lag votes are weak.

This uses host packet-arrival timestamps. It can detect a large counter jump or
drift, but NIC delivery latency prevents it from replacing a broadband
cross-correlation delay solve at one-sample precision.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from lambda_commissioning.utils import _LAMBDA_PACKET_DTYPE, get_udp_payload_bytes

SAMPLE_SECONDS = 1.08e-6


def read_board(path, packets):
    stamps, payloads = get_udp_payload_bytes(str(path), packets=packets)
    records = np.frombuffer(b"".join(payloads), dtype=_LAMBDA_PACKET_DTYPE)
    epochs = np.asarray(stamps) - records["seq_no"].astype(float) * SAMPLE_SECONDS
    return dict(median_epoch_unix=float(np.median(epochs)),
                p05_epoch_unix=float(np.percentile(epochs, 5)),
                p95_epoch_unix=float(np.percentile(epochs, 95)),
                arrival_epoch_p05_p95_spread_us=float(
                    (np.percentile(epochs, 95) - np.percentile(epochs, 5)) * 1e6),
                first_packet_utc_unix=float(stamps[0]), packet_count=len(records))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("delay_file", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("pcap_dirs", type=Path, nargs="+")
    p.add_argument("--packets", type=int, default=12000)
    p.add_argument("--reference", type=int, default=1)
    a = p.parse_args()
    delays = {int(k): int(v) for k, v in json.loads(a.delay_file.read_text()).items()}
    if set(delays) != set(range(4)):
        raise ValueError("delay file must contain all four FPGA IDs")
    captures = []
    for directory in a.pcap_dirs:
        boards = {str(b): read_board(directory / f"fpga{b}.pcapng", a.packets)
                  for b in range(4)}
        reference_epoch = boards[str(a.reference)]["median_epoch_unix"]
        reference_p05 = boards[str(a.reference)]["p05_epoch_unix"]
        for board in range(4):
            row = boards[str(board)]
            relative = (row["median_epoch_unix"] - reference_epoch) / SAMPLE_SECONDS
            relative_p05 = (row["p05_epoch_unix"] - reference_p05) / SAMPLE_SECONDS
            row["relative_counter_epoch_samples"] = float(relative)
            row["expected_relative_epoch_samples"] = delays[a.reference] - delays[board]
            row["residual_from_delay_file_samples"] = float(
                relative - (delays[a.reference] - delays[board]))
            row["p05_residual_from_delay_file_samples"] = float(
                relative_p05 - (delays[a.reference] - delays[board]))
        captures.append(dict(directory=str(directory), boards=boards))
    first = captures[0]["boards"]
    for capture in captures:
        for board in range(4):
            row = capture["boards"][str(board)]
            row["change_from_first_capture_samples"] = float(
                row["relative_counter_epoch_samples"] -
                first[str(board)]["relative_counter_epoch_samples"])
    report = dict(delay_file=str(a.delay_file), reference_fpga=a.reference,
                  sample_period_s=SAMPLE_SECONDS, captures=captures,
                  limitation="Arrival timestamp bias can differ across NICs or burst."
                             " This is a counter-continuity test, not proof of a"
                             " sub-sample residual. Only raw broadband correlation"
                             " plus closure can make the one-sample claim.")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    for capture in captures:
        print(capture["directory"], "median", [round(capture["boards"][str(b)]
              ["residual_from_delay_file_samples"], 2) for b in range(4)],
              "low-arrival-tail", [round(capture["boards"][str(b)]
              ["p05_residual_from_delay_file_samples"], 2) for b in range(4)])


if __name__ == "__main__":
    main()

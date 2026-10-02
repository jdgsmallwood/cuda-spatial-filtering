#!/usr/bin/env python3
"""Estimate coarse FPGA counter offsets from short, simultaneous PCAPs.

This aligns packet sample counters for a machinery test when ADC samples are
zero. It is not a geometric, analogue, or sky-derived delay calibration.
"""
import argparse
import json
from pathlib import Path
import statistics
import struct

SAMPLE_SECONDS = 1.08e-6
SAMPLES_PER_PACKET = 64


def read_pcap(path: Path):
    rows = []
    with path.open("rb") as stream:
        magic = stream.read(4)
        if magic == b"\xd4\xc3\xb2\xa1":
            order = "<"
        elif magic == b"\xa1\xb2\xc3\xd4":
            order = ">"
        else:
            raise ValueError(f"{path}: expected microsecond classic PCAP")
        stream.read(20)
        while record := stream.read(16):
            seconds, micros, captured, _ = struct.unpack(order + "IIII", record)
            packet = stream.read(captured)
            if len(packet) < 56:
                continue
            sample = struct.unpack_from("<Q", packet, 42)[0]
            channel = struct.unpack_from("<H", packet, 54)[0]
            if 176 <= channel <= 215 and sample > 0:
                rows.append((seconds + micros * 1e-6, sample))
    if not rows:
        raise ValueError(f"{path}: no 176--215 sample-counter packets")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="contains fpga0.pcap ... fpga3.pcap")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packets = [read_pcap(args.directory / f"fpga{i}.pcap") for i in range(4)]
    epoch = min(group[0][0] for group in packets)
    offsets = []
    for index, group in enumerate(packets):
        reference_counts = [sample - (timestamp - epoch) / SAMPLE_SECONDS
                            for timestamp, sample in group]
        median = statistics.median(reference_counts)
        mad = statistics.median(abs(value - median) for value in reference_counts)
        offsets.append(median)
        print(f"FPGA {index}: {len(group)} packets; timestamp-corrected "
              f"counter={median:.1f}; MAD={mad:.1f} samples")
    delays = {str(index): int(round((value - offsets[1]) / SAMPLES_PER_PACKET)
                               * SAMPLES_PER_PACKET)
              for index, value in enumerate(offsets)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(delays, indent=2) + "\n")
    print(f"packet-only diagnostic delays: {delays}")
    print(f"wrote {args.output}; unsuitable for coherent beam calibration")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize PN-off int8 sample headroom across eight-channel PCAP groups."""

import argparse
import json
from pathlib import Path

import numpy as np
from lambda_commissioning.utils import _LAMBDA_PACKET_DTYPE, get_udp_payload_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcap_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--packets-per-file", type=int, default=3000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    groups = []
    for start in range(400, 440, 8):
        board_rows = []
        for board in range(4):
            path = args.pcap_root / f"pn_off_{start}" / f"fpga{board}.pcapng"
            _, payloads = get_udp_payload_bytes(str(path),
                                                 packets=args.packets_per_file)
            records = np.frombuffer(b"".join(payloads), dtype=_LAMBDA_PACKET_DTYPE)
            amplitude = np.abs(records["samples"].astype(np.int16))
            scales = records["scales"].astype(np.int32)
            board_rows.append({
                "fpga": board, "packets": int(len(records)),
                "absolute_sample_p99": float(np.quantile(amplitude, .99)),
                "absolute_sample_p9999": float(np.quantile(amplitude, .9999)),
                "absolute_sample_max": int(amplitude.max()),
                "fraction_abs_ge_120": float(np.mean(amplitude >= 120)),
                "fraction_zero": float(np.mean(amplitude == 0)),
                "scale_min": int(scales.min()), "scale_max": int(scales.max()),
            })
        groups.append({"channels": [start, start + 7], "boards": board_rows})
    rows = [r for group in groups for r in group["boards"]]
    report = {
        "source": str(args.pcap_root), "adc_gain_setting": 20,
        "packets_per_file": args.packets_per_file, "groups": groups,
        "maximum_absolute_sample": max(r["absolute_sample_max"] for r in rows),
        "maximum_absolute_sample_p9999": max(r["absolute_sample_p9999"] for r in rows),
        "any_abs_ge_120": any(r["fraction_abs_ge_120"] > 0 for r in rows),
        "interpretation": "Checks only int8 packet-sample clipping and records the accompanying packet scale factors. It does not measure raw ADC clipping or analog headroom. A gain change requires raw ADC statistics/data as well as packet scale and sky-coherence checks.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "maximum_absolute_sample", "maximum_absolute_sample_p9999",
        "any_abs_ge_120")}, indent=2))


if __name__ == "__main__":
    main()

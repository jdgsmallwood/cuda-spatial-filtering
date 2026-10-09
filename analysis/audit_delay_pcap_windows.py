#!/usr/bin/env python3
"""Check an existing six-pair FPGA delay solution on disjoint raw-PCAP windows.

The production solver selects the mode of antenna-pair cross-correlation lags.
Graph closure alone cannot exclude a common false correlation peak. This audit
repeats the *same* lag calculation on non-overlapping capture-time windows,
reports each coarse channel separately, and compares the six measured edges
with a saved delay JSON. It never changes the observing delay file.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from scipy.fft import fft, ifft, next_fast_len


PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]


def lag_votes(x, y):
    """Match caltools' argmax |IFFT(X conj(Y))|, with a linear FFT length."""
    n = x.shape[2]
    transform_length = next_fast_len(2 * n - 1)
    counts = Counter()
    by_channel = {}
    for channel in range(x.shape[1]):
        X = fft(x[:, channel], n=transform_length, axis=1, workers=1)
        Y = fft(y[:, channel], n=transform_length, axis=1, workers=1)
        corr = ifft(X[:, None, :, :] * Y[None, :, :, :].conj(),
                    axis=2, workers=1)
        index = np.argmax(abs(corr), axis=2).ravel()
        signed = np.where(index < transform_length // 2, index,
                          index - transform_length)
        votes = Counter(map(int, signed))
        counts.update(votes)
        by_channel[channel] = votes
    return counts, by_channel


def top_votes(votes, count=5):
    return [{"lag_samples": lag, "votes": n}
            for lag, n in votes.most_common(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcap_dir", type=Path)
    parser.add_argument("delay_file", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--commissioning", type=Path, required=True)
    parser.add_argument("--packets", type=int, default=3000)
    parser.add_argument("--offset-seconds", type=float, nargs="+",
                        default=[0.02, 0.10],
                        help="Offsets from the latest PCAP start timestamp")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if args.packets < 64 or args.offset_seconds[0] < 0 or any(
            a >= b for a, b in zip(args.offset_seconds, args.offset_seconds[1:])):
        parser.error("invalid packets or ordered nonnegative offsets")
    sys.path.insert(0, str(args.commissioning / "src"))
    from lambda_commissioning.utils import (get_udp_payload_bytes,
                                            read_raw_lambda_packets)

    paths = [args.pcap_dir / f"fpga{board}.pcapng" for board in range(4)]
    start_times = [get_udp_payload_bytes(str(path), packets=1)[0][0]
                   for path in paths]
    common_start = max(start_times)
    configured = {int(k): int(v) for k, v in
                  json.loads(args.delay_file.read_text()).items()}
    if set(configured) != set(range(4)):
        parser.error("delay file must give all four FPGA IDs")
    windows = []
    for offset in args.offset_seconds:
        timestamp = common_start + offset
        data, first_samples = read_raw_lambda_packets(
            [str(p) for p in paths], args.packets,
            start_timestamp=timestamp)
        # FPGA 0 has four disconnected antenna input slots. The production
        # six-pair solver excludes them from any pair containing that board.
        data[0] = np.delete(data[0], [0, 2, 4, 6], axis=0)
        edges = []
        for i, j in PAIRS:
            votes, per_channel = lag_votes(data[i], data[j])
            best, best_votes = votes.most_common(1)[0]
            measured = int(first_samples[j] - first_samples[i] - best)
            expected = configured[j] - configured[i]
            edge = {
                "pair": [i, j], "measured_samples": measured,
                "configured_samples": expected,
                "residual_samples": measured - expected,
                "mode_lag_samples": best,
                "mode_votes": best_votes,
                "total_votes": sum(votes.values()),
                "top_lags": top_votes(votes),
                "channels": [
                    {"coarse_index": channel, "mode": top_votes(counts, 1)[0],
                     "votes_for_global_mode": counts[best],
                     "total_votes": sum(counts.values())}
                    for channel, counts in per_channel.items()],
            }
            edges.append(edge)
            print(f"window {offset:.3f}s FPGA {i}-{j}: "
                  f"residual {edge['residual_samples']:+d}, "
                  f"votes {best_votes}/{edge['total_votes']}", flush=True)
        windows.append({"offset_seconds": offset,
                        "requested_timestamp": timestamp,
                        "first_sample_counts": first_samples,
                        "edges": edges})
    report = {
        "pcap_dir": str(args.pcap_dir),
        "delay_file": str(args.delay_file),
        "packets_per_board_per_window": args.packets,
        "common_start_timestamp": common_start,
        "method": "Disjoint PCAP windows, same raw-pair FFT lag argmax as the production solver; FPGA0 disconnected input slots excluded.",
        "windows": windows,
        "all_six_edges_match_all_windows": all(
            edge["residual_samples"] == 0 for window in windows
            for edge in window["edges"]),
        "max_abs_residual_samples": max(
            abs(edge["residual_samples"]) for window in windows
            for edge in window["edges"]),
        "all_six_edges_within_one_sample_all_windows": all(
            abs(edge["residual_samples"]) <= 1 for window in windows
            for edge in window["edges"]),
        "limitation": "Tests this capture at two short time spans; a clock-rate fault later in the science observation requires a separate time-dependent check.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("All edges match:", report["all_six_edges_match_all_windows"])


if __name__ == "__main__":
    main()

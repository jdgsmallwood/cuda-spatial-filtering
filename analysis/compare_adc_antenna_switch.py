#!/usr/bin/env python3
"""Locate coherent raw-sample pickup with antenna input on/off PCAP controls."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lambda_commissioning.utils import _LAMBDA_PACKET_DTYPE, get_udp_payload_bytes


def metrics(records, receivers, hand):
    raw = records["samples"].astype(np.float32)
    voltage = (raw[..., 0] + 1j * raw[..., 1]) * records["scales"][:, None]
    samples = voltage[:, :, receivers, hand].reshape(-1, len(receivers))
    mean = samples.mean(axis=0)
    power = np.mean(abs(samples) ** 2, axis=0)
    centred = samples - mean
    covariance = samples.conj().T @ samples / len(samples)
    centred_covariance = centred.conj().T @ centred / len(samples)
    centred_power = np.mean(abs(centred) ** 2, axis=0)
    pairs = np.triu_indices(len(receivers), 1)
    norm = np.sqrt(power[:, None] * power[None, :])
    centred_norm = np.sqrt(centred_power[:, None] * centred_power[None, :])
    return {
        "packet_count": int(len(records)),
        "sample_count": int(len(samples)),
        "connected_receivers": len(receivers),
        "median_auto_power": float(np.median(power)),
        "median_abs_cross_coherence": float(np.median(abs(covariance[pairs] /
                                                     np.maximum(norm[pairs], 1e-30)))),
        "median_abs_centered_cross_coherence": float(np.median(abs(centred_covariance[pairs] /
                                                              np.maximum(centred_norm[pairs], 1e-30)))),
        "median_dc_power_fraction": float(np.median(abs(mean) ** 2 /
                                                    np.maximum(power, 1e-30))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("antennas_on", type=Path)
    ap.add_argument("antennas_off", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--mapping", type=Path, default=Path("stream_antenna_map.json"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(args.mapping.read_text())["fpgas"]
    results = {}
    for state, directory in (("antennas_on", args.antennas_on),
                             ("antennas_off", args.antennas_off)):
        results[state] = {}
        for board in range(4):
            _, payloads = get_udp_payload_bytes(
                str(directory / f"fpga{board}.pcapng"), packets=20000)
            records = np.frombuffer(b"".join(payloads), dtype=_LAMBDA_PACKET_DTYPE)
            channels = np.unique(records["freq_chan"])
            by_hand = {}
            for hand, name in ((0, "YY"), (1, "XX")):
                streams = [stream["stream"] for stream in mapping[str(board)]["streams"]
                           if stream["antenna_id"] > 0 and stream["antenna_id"] != 14
                           and stream["polarization"] == hand]
                receivers = np.asarray(streams) // 2
                raw_hands = np.asarray(streams) % 2
                if len(receivers) < 2:
                    raise ValueError(f"too few connected receivers on FPGA {board}")
                by_hand[name] = {
                    str(int(channel)): metrics(records[records["freq_chan"] == channel],
                                               receivers, raw_hands)
                    for channel in channels
                }
            results[state][str(board)] = by_hand
    channel_set = [set(results[state]["0"]["YY"]) for state in results]
    if channel_set[0] != channel_set[1]:
        raise ValueError("antenna on/off captures have different coarse channels")
    channels = np.array(sorted(map(int, channel_set[0])))
    ratios = {}
    for hand in ("XX", "YY"):
        on = np.array([results["antennas_on"][str(board)][hand][str(int(ch))]
                       ["median_abs_cross_coherence"]
                       for board in range(4) for ch in channels])
        off = np.array([results["antennas_off"][str(board)][hand][str(int(ch))]
                        ["median_abs_cross_coherence"]
                        for board in range(4) for ch in channels])
        ratios[hand] = {
            "median_on": float(np.median(on)),
            "median_off": float(np.median(off)),
            "on_over_off": float(np.median(on) / np.median(off)),
        }
    report = {
        "on_capture": str(args.antennas_on), "off_capture": str(args.antennas_off),
        "coarse_channels": channels.tolist(), "by_board_hand_channel": results,
        "stream_map": str(args.mapping), "excluded_antennas": [14],
        "summary": ratios,
        "interpretation": "The antenna-selection bit removes coherent YY power if the pickup enters upstream of that selection. This test does not distinguish an external transmitter from pickup on the antenna/front-end path.",
        "method": "Same-FPGA raw complex samples at zero lag; connected receivers only; packet scales included; covariance both before and after complex-mean subtraction.",
    }
    (args.output / "antenna_switch_control.json").write_text(
        json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for axis, hand in zip(axes, ("XX", "YY")):
        for state, marker in (("antennas_on", "o"), ("antennas_off", "x")):
            for board in range(4):
                value = [results[state][str(board)][hand][str(int(ch))]
                         ["median_abs_cross_coherence"] for ch in channels]
                axis.plot(channels, value, marker + "-", label=f"FPGA {board} {state}")
        axis.set(title=hand, xlabel="Coarse channel",
                 ylabel="Median absolute within-FPGA coherence", yscale="log")
        axis.grid(alpha=.25)
    axes[1].legend(fontsize=7, ncol=2)
    fig.savefig(args.output / "antenna_switch_control.png", dpi=170)
    print(json.dumps(ratios, indent=2))


if __name__ == "__main__":
    main()

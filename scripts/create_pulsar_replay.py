#!/usr/bin/env python3
"""Make a deterministic four-FPGA pulsar PCAP and matching measured gains.

The source is shared complex Gaussian voltage with a periodic power envelope.
Every antenna receives a geometric carrier phase, a fractional baseband delay,
and its own complex gain and receiver noise. The pulsar pipeline is expected
to conjugate-invert the measured gains and steer to the chosen RA/Dec.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import struct
import subprocess

import numpy as np

from create_pcap import ethernet_header, ipv4_header, pcap_global_header, udp_header

SAMPLE_SECONDS = 1.08e-6
SAMPLES_PER_PACKET = 64
SAMPLE_COUNT_BASE = 1024
RECEIVERS_PER_FPGA = 10
POLARIZATIONS = 2
C = 299792458.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--stream-map", type=Path, default=Path("stream_antenna_map.json"))
    parser.add_argument("--targets", type=Path, default=Path("config/crab_targets.json"))
    parser.add_argument("--pointing-bin", type=Path, default=Path("build/apps/pointing_enu"))
    parser.add_argument("--steering-utc", default="2026-09-23T19:00:00Z")
    parser.add_argument("--min-channel", type=int, default=176)
    parser.add_argument("--coarse-channels", type=int, default=1)
    parser.add_argument("--duration", type=float, default=11.0)
    parser.add_argument("--period", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--chunk-packets", type=int, default=32)
    args = parser.parse_args()
    if args.coarse_channels < 1 or args.duration <= 0 or args.period <= 0:
        parser.error("coarse channels, duration, and period must be positive")

    config = json.loads(args.config.read_text())
    streams = json.loads(args.stream_map.read_text())["fpgas"]
    target = json.loads(args.targets.read_text())["targets"][0]
    site = config["array_location"]
    epoch = datetime.strptime(args.steering_utc, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    # Use the pipeline's own casacore conversion, so injected and corrected
    # geometric phases use precisely the same sky frame and UTC convention.
    pointing_output = subprocess.check_output(
        [str(args.pointing_bin.resolve()), args.steering_utc,
         str(target["ra_deg"]), str(target["dec_deg"]),
         str(site["latitude_deg"]), str(site["longitude_deg"]),
         str(site["height_m"])], text=True
    )
    direction = np.array([float(x) for x in pointing_output.splitlines()[-1].split()])
    alt_deg = math.degrees(math.asin(direction[2]))
    az_deg = math.degrees(math.atan2(direction[0], direction[1])) % 360
    if alt_deg < 15:
        parser.error(f"target altitude {alt_deg:.2f} deg is below 15 deg")

    antenna_ids = np.full((4, RECEIVERS_PER_FPGA, POLARIZATIONS), -1, dtype=int)
    canonical_pols = np.zeros_like(antenna_ids)
    positions = np.zeros((4, RECEIVERS_PER_FPGA, 3), dtype=float)
    for fpga in range(4):
        for item in streams[str(fpga)]["streams"]:
            receiver, raw_pol = divmod(int(item["stream"]), POLARIZATIONS)
            antenna = int(item["antenna_id"])
            antenna_ids[fpga, receiver, raw_pol] = antenna
            canonical_pols[fpga, receiver, raw_pol] = int(item["polarization"])
            if antenna >= 0:
                p = config["antenna_positions"][str(fpga)][str(antenna)]
                positions[fpga, receiver] = [p["east"], p["north"], p["up"]]
    if not np.all(antenna_ids[:, :, 0] == antenna_ids[:, :, 1]):
        raise ValueError("each receiver's two raw pol streams must be one antenna")
    delay_seconds = np.einsum("frd,d->fr", positions, direction) / C

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pcap_path = args.output_dir / "pulsar.pcap"
    gains_path = args.output_dir / "measured_fine_gains.json"
    truth_path = args.output_dir / "truth.json"
    ephemeris_path = args.output_dir / "synthetic.par"
    (args.output_dir / "zero_fpga_delays.json").write_text(
        json.dumps({str(fpga): 0 for fpga in range(4)}, indent=2) + "\n"
    )
    rng = np.random.default_rng(args.seed)
    measured = {}
    for antenna in np.unique(antenna_ids):
        if antenna < 0:
            continue
        for pol in ("XX", "YY"):
            amplitude = rng.uniform(0.75, 1.25)
            phase = rng.uniform(-np.pi, np.pi)
            measured[(int(antenna), pol)] = amplitude * np.exp(1j * phase)

    fine_step_hz = 781250.0 / 27.0
    first_hz = args.min_channel * 781250.0 - 13 * fine_step_hz
    last_hz = (args.min_channel + args.coarse_channels - 1) * 781250.0 + 13 * fine_step_hz
    gains = {
        "channel_axis": "fine_processing_index",
        "excluded_receptors": [],
        "provenance": {
            "generator": "create_pulsar_replay.py",
            "processing_channels_with_boundary_duplicates": args.coarse_channels * 28,
            "unique_frequency_channels": args.coarse_channels * 27,
            "frequency_first_hz": first_hz,
            "frequency_last_hz": last_hz,
            "excluded_antennas": [14, 36],
        },
        "weights": {},
    }
    for fine_index in range(args.coarse_channels * 28):
        channel = str(args.min_channel + fine_index)
        gains["weights"][channel] = {}
        for pol in ("XX", "YY"):
            gains["weights"][channel][pol] = {
                str(ant): {"real": float(g.real), "imag": float(g.imag)}
                for (ant, gain_pol), g in measured.items() if gain_pol == pol
            }
    gains_path.write_text(json.dumps(gains, indent=2) + "\n")

    # A low spin frequency keeps barycentric Doppler drift negligible over this
    # short topocentric synthetic observation; absolute pulse phase is arbitrary.
    mjd = 40587.0 + epoch.timestamp() / 86400.0
    ephemeris_path.write_text(
        "PSRJ J0534+2200\n"
        "RAJ 05:34:31.973\nDECJ +22:00:52.061\n"
        f"F0 {1 / args.period:.12f}\nF1 0\nPEPOCH {mjd:.9f}\n"
        "DM 0.001\nUNITS TDB\n"
    )

    total_packets_per_stream = int(np.ceil(args.duration / (SAMPLE_SECONDS * SAMPLES_PER_PACKET)))
    sample_count_total = total_packets_per_stream * SAMPLES_PER_PACKET
    frame_prefix = {}
    for fpga in range(4):
        source_ip = f"10.0.{fpga}.10"
        payload_bytes = 22 + 40 + 64 * 10 * 2 * 2
        frame_prefix[fpga] = (
            ethernet_header(b"\xff" * 6, b"\x00\x0a\x95\x9d\x68\x16", 0x0800)
            + ipv4_header(source_ip, "192.168.1.255", payload_bytes)
            + udp_header(36001, 36001, payload_bytes)
        )
    scales = struct.pack("<20h", *([1] * 20))
    expected_packets = total_packets_per_stream * 4 * args.coarse_channels
    print(f"Writing {expected_packets:,} packets ({expected_packets * 2680 / 1e9:.2f} GB) to {pcap_path}", flush=True)
    with pcap_path.open("wb", buffering=4 * 1024 * 1024) as output:
        output.write(pcap_global_header())
        for packet_start in range(0, total_packets_per_stream, args.chunk_packets):
            packet_count = min(args.chunk_packets, total_packets_per_stream - packet_start)
            n = packet_count * SAMPLES_PER_PACKET
            sample_start = packet_start * SAMPLES_PER_PACKET
            sample_indices = sample_start + np.arange(-1, n + 2)
            pulse_phase = np.remainder(sample_indices * SAMPLE_SECONDS, args.period) / args.period
            pulse_distance = np.minimum(pulse_phase, 1 - pulse_phase)
            envelope = np.sqrt(1.0 + 12.0 * np.exp(-0.5 * (pulse_distance / 0.035) ** 2))
            # Reuse one source waveform across antennas; each coarse channel
            # receives a separately seeded realization of the same pulsar.
            for coarse in range(args.coarse_channels):
                source = (
                    rng.standard_normal((2, n + 3))
                    + 1j * rng.standard_normal((2, n + 3))
                ) * envelope[None, :] * 7.0 / np.sqrt(2)
                freq_hz = (args.min_channel + coarse) * 781250.0
                raw = np.zeros((4, 10, 2, n), dtype=np.complex64)
                for fpga in range(4):
                    for receiver in range(10):
                        antenna = int(antenna_ids[fpga, receiver, 0])
                        if antenna < 0:
                            continue
                        shift = delay_seconds[fpga, receiver] / SAMPLE_SECONDS
                        left = source[:, 1:n + 1] if shift >= 0 else source[:, :n]
                        right = source[:, 2:n + 2] if shift >= 0 else source[:, 1:n + 1]
                        shifted = left + abs(shift) * (right - left)
                        carrier = np.exp(2j * np.pi * freq_hz * delay_seconds[fpga, receiver])
                        for raw_pol in range(2):
                            label = "XX" if canonical_pols[fpga, receiver, raw_pol] == 0 else "YY"
                            raw[fpga, receiver, raw_pol] = (
                                shifted[canonical_pols[fpga, receiver, raw_pol]]
                                * carrier * measured[(antenna, label)]
                            )
                raw += (rng.standard_normal(raw.shape) + 1j * rng.standard_normal(raw.shape)) * 1.0
                quantized = np.empty((n, 4, 10, 2, 2), dtype=np.int8)
                quantized[..., 0] = np.clip(np.rint(raw.real.transpose(3, 0, 1, 2)), -127, 127)
                quantized[..., 1] = np.clip(np.rint(raw.imag.transpose(3, 0, 1, 2)), -127, 127)
                packets = quantized.reshape(packet_count, 64, 4, 10, 2, 2).transpose(0, 2, 1, 3, 4, 5)
                for packet in range(packet_count):
                    relative_sample = sample_start + packet * 64
                    sample = SAMPLE_COUNT_BASE + relative_sample
                    timestamp = epoch.timestamp() + relative_sample * SAMPLE_SECONDS
                    seconds = int(timestamp)
                    micros = int((timestamp - seconds) * 1_000_000)
                    for fpga in range(4):
                        frame = (
                            frame_prefix[fpga]
                            + struct.pack("<QIHQ", sample, fpga, args.min_channel + coarse, 0)
                            + scales + packets[packet, fpga].tobytes()
                        )
                        output.write(struct.pack("<IIII", seconds, micros, len(frame), len(frame)))
                        output.write(frame)
            if packet_start % (args.chunk_packets * 256) == 0:
                print(f"  {packet_start}/{total_packets_per_stream} sample groups", flush=True)

    truth = {
        "pcap": str(pcap_path),
        "steering_utc": args.steering_utc,
        "ra_deg": target["ra_deg"], "dec_deg": target["dec_deg"],
        "alt_deg": alt_deg, "az_deg": az_deg,
        "enu_direction": direction.tolist(),
        "period_seconds": args.period,
        "duration_seconds": sample_count_total * SAMPLE_SECONDS,
        "sample_seconds": SAMPLE_SECONDS,
        "sample_count_base": SAMPLE_COUNT_BASE,
        "packets": expected_packets,
        "seed": args.seed,
        "min_channel": args.min_channel,
        "coarse_channels": args.coarse_channels,
        "delay_seconds_by_fpga_receiver": delay_seconds.tolist(),
    }
    truth_path.write_text(json.dumps(truth, indent=2) + "\n")
    print(f"Wrote {pcap_path}, {gains_path}, {ephemeris_path}, {truth_path}")


if __name__ == "__main__":
    main()

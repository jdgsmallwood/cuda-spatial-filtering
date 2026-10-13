#!/usr/bin/env python3
"""Independent NumPy reference for the single-FPGA CODIF Starspin replay.

Models unity gains, zero delays, identity stream order, 256-packet blocks,
the 32-bin/16-tap PFB, retained bins 3..29, and FP16 voltage rounding.
Requires NumPy and h5dump (both present in jay/cuda-env).
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import struct
import subprocess

import numpy as np

from check_adaptive_detection_parity import (
    h5dump_attr, h5dump_executable, load_dataset_with_h5dump,
    parse_h5dump_attr_scalar,
)


def read_pcap(path):
    data = path.read_bytes()
    if len(data) < 24 or data[:4] != b"\xd4\xc3\xb2\xa1" or struct.unpack_from("<I", data, 20)[0] != 1:
        raise ValueError("Expected little-endian Ethernet PCAP")
    packets, sources, exponents = {}, set(), set()
    offset = 24
    while offset < len(data):
        if len(data) - offset < 16:
            raise ValueError("Truncated PCAP record")
        size, original = struct.unpack_from("<II", data, offset + 8)
        packet = data[offset + 16:offset + 16 + size]
        offset += 16 + size
        if size != 2666 or original != size or len(packet) != size:
            raise ValueError("Expected complete 2666-byte LAMBDA CODIF frames")
        if packet[12:15] != b"\x08\x00\x45" or packet[23] != 17:
            raise ValueError("Expected untagged Ethernet/IPv4/UDP without IP options")
        sources.add(packet[28])
        header = packet[42:106]
        frame, seconds = struct.unpack_from("<II", header)
        if (header[9] != 8 or header[10] & 0xf7 != 4 or
                struct.unpack_from("<H", header, 14)[0] != 27 or
                struct.unpack_from("<H", header, 16)[0] >= 512 or
                struct.unpack_from("<H", header, 44)[0] not in (0, 0x03bb) or
                struct.unpack_from("<HHI", header, 24) != (20, 5, 320) or
                struct.unpack_from("<Q", header, 32)[0] not in (390625, 25000000) or
                header[40:44] != b"\xfe\xca\xed\xfe" or frame >= 390625):
            raise ValueError("Unsupported CODIF geometry or invalid header")
        epoch = datetime(2000 + header[8] // 2, 1 + 6 * (header[8] % 2), 1)
        seconds += int((epoch - datetime(2000, 1, 1)).total_seconds())
        sequence = seconds * 25000000 // 27 + frame * 64
        channel = struct.unpack_from("<H", header, 16)[0]
        powers = [(header[46 + adc // 2] >> (adc % 2 * 4)) & 15 for adc in range(20)]
        exponents.update(powers)
        scale = 2 ** np.minimum(powers, 8)
        raw = np.frombuffer(packet, dtype=np.int8, offset=106).reshape(64, 20, 2)
        key = (sequence, channel)
        if key in packets:
            raise ValueError("Duplicate sequence/channel in PCAP")
        packets[key] = (raw[..., 0] + 1j * raw[..., 1]) * scale
    if not packets or len(sources) != 1:
        raise ValueError("Expected one FPGA stream")
    channels = sorted({channel for _, channel in packets})
    counts = sorted({sequence for sequence, _ in packets})
    if channels != list(range(channels[0], channels[0] + 8)) or counts != list(range(counts[0], counts[-1] + 1, 64)):
        raise ValueError("Expected eight consecutive channels and consecutive frames")
    if len(packets) != len(channels) * len(counts):
        raise ValueError("Missing channel/frame in PCAP")
    samples = np.stack([np.concatenate([packets[s, c] for s in counts]) for c in channels])
    metadata = dict(packets=len(packets), fpga_id=sources.pop(), first_sample=counts[0],
                    first_channel=channels[0], scale_exponents=sorted(exponents),
                    distinct_stream_waveforms=len({samples[:, :, a].tobytes() for a in range(20)}))
    return samples, metadata


def pfb_coefficients():
    # gpu-filter's unsigned 2/512 ramp width is zero: the frequency grid is a step.
    grid = (np.arange(513) < 16).astype(float)
    impulse = np.fft.ifft(np.r_[grid, np.zeros(1024), grid[-2:0:-1]]).real
    coefficients = np.r_[impulse[-511::2], impulse[1:512:2]] * np.kaiser(512, 9.0695)
    # gpu-filter normalizes to its centre tap, then to its historical 256-bin gain.
    return (coefficients / coefficients[255] * 8).astype(np.float32).reshape(16, 32)


def half_round(values):
    return (values.real.astype(np.float16).astype(float) +
            1j * values.imag.astype(np.float16).astype(float))


def correlate_block(samples, phase):
    data = half_round(samples[:, (np.arange(16384) + phase) % samples.shape[1]])
    # Production independently zero-pads the 480-sample look-ahead of every block.
    data = np.pad(data, ((0, 0), (0, 480), (0, 0))).reshape(8, 527, 32, 20)
    filtered = np.zeros((8, 512, 32, 20), dtype=complex)
    for tap, weights in enumerate(pfb_coefficients()):
        filtered += data[:, tap:tap + 512] * weights[:, None]
    fine = np.fft.fftshift(np.fft.fft(filtered, axis=2), axes=2)[:, :, 3:30] / 64
    fine = half_round(fine).transpose(0, 2, 1, 3).reshape(216, 512, 10, 2)
    return np.stack([
        np.einsum("ctp,ctq->cpq", fine[:, :, i], fine[:, :, j].conj())
        for j in range(10) for i in range(j + 1)
    ], axis=1) * 4096


def error_metrics(got, expected):
    pairs = [(i, j) for j in range(10) for i in range(j + 1)]
    power = np.stack([expected[:, b].real.diagonal(axis1=-2, axis2=-1)
                      for b, (i, j) in enumerate(pairs) if i == j], axis=1)
    if not np.isfinite(got).all() or not np.isfinite(expected).all() or np.any(power < 0) or not np.any(power > 0):
        raise ValueError("Non-finite visibilities, negative autopower, or empty reference")
    normalization = np.stack([np.sqrt(power[:, i, :, None] * power[:, j, None, :])
                              for i, j in pairs], axis=1)
    difference = np.abs(got - expected)
    # Autopower normalization preserves sensitivity on weak channels and cross-hands.
    scaled = difference / np.maximum(normalization, np.max(power) * 1e-12)
    return float(np.linalg.norm(difference) / np.linalg.norm(expected)), float(scaled.max())


def check(pcap, visibility, output):
    samples, metadata = read_pcap(pcap)
    text = subprocess.check_output([h5dump_executable(), "-d", "/audit/run_manifest_json",
                                    "-y", "-w", "0", str(visibility)], text=True)
    data_text = text.split("   DATA {\n", 1)[1].lstrip()
    manifest, _ = json.JSONDecoder().raw_decode(data_text[1:])
    args = manifest["command_line"]
    normalized = manifest["normalized"]
    delays = json.loads(manifest["input_files"]["fpga_delays"]["content"])
    if (normalized["selected_fpga_ids"] != [metadata["fpga_id"]] or
            normalized["min_frequency_channel"] != metadata["first_channel"] or
            any(value != 0 for value in delays.values()) or
            "-a" in args or "--apply-gains-to-vis" in args or
            any(manifest["input_files"][key]["path"] for key in ("deripple", "stream_antenna_map", "fine_delays")) or
            not any(arg in args for arg in ("--loop", "-l"))):
        raise ValueError("Reference requires looping one FPGA with unity gains, zero delays, and identity stream order")
    for name, value in [("fine_fft_channels", 32), ("fine_channels_per_coarse", 27), ("first_raw_fine_bin", 3)]:
        if parse_h5dump_attr_scalar(h5dump_attr(visibility, name)) != value:
            raise ValueError(f"Unsupported {name}")
    load = lambda name, dtype: load_dataset_with_h5dump(visibility, name, dtype)
    gpu = load("visibilities", np.float32)
    seqs = load("vis_seq_nums", np.uint64)
    missing = load("vis_missing_nums", np.float32)
    pairs = load("baseline_receiver_indices", np.int32)
    np.testing.assert_array_equal(pairs, [(i, j) for j in range(10) for i in range(j + 1)])
    if (gpu.shape != (len(seqs), 216, 55, 2, 2, 2) or len(seqs) == 0 or
            missing.shape != (len(seqs), 3) or np.any(missing[:, 0] != 0)):
        raise ValueError("Expected complete integrations with no missing packets")
    # The fixture repeats every 256 samples, so only distinct block phases need a PFB.
    cached, rows, reference_rows = {}, [], []
    for row, (start, end) in enumerate(seqs):
        count = int(end) - int(start) + 64
        if count <= 0 or count % 16384 or count // 64 * 8 != missing[row, 1]:
            raise ValueError("Inconsistent integration range or unsupported block geometry")
        expected = np.zeros((216, 55, 2, 2), dtype=complex)
        for block_start in range(int(start), int(end) + 1, 16384):
            phase = (block_start - metadata["first_sample"]) % samples.shape[1]
            if phase not in cached:
                cached[phase] = correlate_block(samples, phase)
            expected += cached[phase]
        got = gpu[row, ..., 0] + 1j * gpu[row, ..., 1]
        rms, maximum = error_metrics(got, expected)
        rows.append(dict(row=row, blocks=count // 16384, relative_rms=rms,
                         max_autopower_normalized_error=maximum))
        reference_rows.append(expected)
    # Fixed tolerances allow FP32 FFT/FIR and accumulation rounding without hiding layout errors.
    passed = all(row["relative_rms"] < 1e-3 and row["max_autopower_normalized_error"] < 2e-3 for row in rows)
    controls = {"conjugated": error_metrics(reference_rows[0].conj(), reference_rows[0])[1],
                "missing_scale": error_metrics(reference_rows[0] / 16384, reference_rows[0])[1]}
    if any(value <= 2e-3 for value in controls.values()):
        raise ValueError("Capture cannot discriminate the negative controls")
    report = dict(passed=passed, pcap_sha256=hashlib.sha256(pcap.read_bytes()).hexdigest(),
                  visibility_file=str(visibility), packet_metadata=metadata,
                  complex_visibilities_compared=len(seqs) * 216 * 55 * 4,
                  max_relative_rms=max(row["relative_rms"] for row in rows),
                  max_autopower_normalized_error=max(row["max_autopower_normalized_error"] for row in rows),
                  negative_controls=controls, integrations=rows)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "python_visibilities.npz", visibilities=np.stack(reference_rows), vis_seq_nums=seqs)
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "integrations"}, indent=2))
    if not passed:
        raise AssertionError("CODIF visibility comparison failed; see comparison.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcap", type=Path, default=Path("tests/fixtures/codif_lambda.pcap"))
    parser.add_argument("--visibilities", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    check(args.pcap, args.visibilities, args.output_dir)

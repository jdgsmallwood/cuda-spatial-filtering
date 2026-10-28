#!/usr/bin/env python3
"""Compare a finite synthetic CODIF replay with DiFX 2.9.0 zerocorr.

Without --binary, runs the CPU compatibility gate only, never a Starspin pass.
Requires numpy and h5py. See docs/difx-validation.md for scope and scaling.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

import h5py
import numpy as np

from check_codif_visibility_parity import error_metrics, read_pcap
from generate_codif_fixture import fixture_bytes

DIFX_COMMIT = "39401aa88b24d3084fe683f52d23975a7b5da3d4"
BLOCK = 16384
FFT = 64
SCALE = 128
FORMAT = "VDIFC_128-390625m27-1-8"
PAIRS = [(i, j) for j in range(10) for i in range(j + 1)]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def generate_fixture(output, integrations, seed=20261019):
    if not 1 <= integrations <= 100:
        raise ValueError("integrations must be in 1..100")
    rng = np.random.default_rng(seed)
    count = (integrations + 2) * BLOCK  # Guard blocks for startup and EOF.
    common = rng.normal(size=(8, count, 1)) + 1j * rng.normal(size=(8, count, 1))
    independent = rng.normal(size=(8, count, 20)) + 1j * rng.normal(size=(8, count, 20))
    gains = np.linspace(2, 5, 20) * np.exp(1j * np.arange(20) * 0.31)
    voltage = common * gains + independent * 2
    iq = np.stack([np.rint(voltage.real), np.rint(voltage.imag)], axis=-1)
    if np.max(np.abs(iq)) > 127:
        raise ValueError("Fixture would clip")
    iq = iq.astype(np.int8)
    np.save(output / "canonical_iq.npy", iq)
    template = fixture_bytes()
    with (output / "input.pcap").open("xb") as file:
        file.write(template[:24])
        for frame in range(count // 64):
            for channel in range(8):
                record = bytearray(template[24 + channel * 2682:24 + (channel + 1) * 2682])
                struct.pack_into("<I", record, 16 + 42, frame)
                record[16 + 106:] = iq[channel, frame * 64:(frame + 1) * 64].tobytes()
                file.write(record)
    samples, metadata = read_pcap(output / "input.pcap")
    np.testing.assert_array_equal(samples, (iq[..., 0].astype(float) + 1j * iq[..., 1]) * SCALE)
    save_json(output / "fixture.json", dict(seed=seed, integrations=integrations,
              samples=count, sample_rate_hz="25000000/27", voltage_scale=SCALE,
              pcap_sha256=sha256(output / "input.pcap"),
              canonical_sha256=sha256(output / "canonical_iq.npy"), packet_metadata=metadata))
    return samples


def write_vdif(path, samples, first_sample=0):
    """One channel, complex offset-binary 8-bit, 64 samples/frame, epoch 2000."""
    iq = np.stack([samples.real, samples.imag], axis=-1) / SCALE
    if (first_sample < 0 or first_sample % 64 or iq.ndim != 2 or len(iq) < 128 or
            len(iq) % 64 or (first_sample + len(iq)) // 64 >= 390625 or
            not np.isfinite(iq).all() or np.any(iq != np.rint(iq)) or
            np.any(iq < -128) or np.any(iq > 127)):
        raise ValueError("VDIF requires exact int8 IQ and complete frames within one 27s period")
    encoded = (iq + 128).astype(np.uint8)
    with path.open("xb") as file:
        for frame, payload in enumerate(encoded.reshape(-1, 128)):
            file.write(struct.pack("<8I", 0, frame + first_sample // 64, 20,
                                   (1 << 31) | (7 << 26) | 0x5353, 0, 0, 0, 0))
            file.write(payload.tobytes())


def run_zerocorr(binary, first, second, start, count, prefix):
    if start < 0 or start % 64 or count <= 0 or count % FFT:
        raise ValueError("zerocorr interval must contain complete 64-sample frames")
    lines = []
    for path in (first, second):
        if any(c.isspace() for c in str(path)):
            raise ValueError("zerocorr input paths cannot contain whitespace")
        lines.extend([str(path), FORMAT, "0", str(start // 64 * 160), str(FFT), "0", str(FFT)])
    vis = prefix.with_suffix(".vis")
    config = prefix.with_suffix(".input")
    lines.extend([str(vis), str(prefix.with_suffix(".lag")), str(count // FFT)])
    config.write_text("\n".join(lines) + "\n")
    env = os.environ.copy()
    env.pop("SYMMETRIC_8_BIT_VDIF", None)
    log_path = prefix.with_suffix(".log")
    with log_path.open("w") as log:
        subprocess.run([str(binary), str(config)], stdout=log, stderr=subprocess.STDOUT,
                       env=env, timeout=60, check=True)
    output = log_path.read_text()
    processed = re.findall(r"(\d+) FFTs processed", output)
    if (processed != [str(count // FFT)] or "VDIF validate[" in output or
            "Skipping invalid frame" in output or "Skipping frame with" in output):
        raise ValueError(f"Incomplete/invalid DiFX decode: {prefix}")
    spectrum = np.loadtxt(vis)
    if spectrum.shape != (FFT, 8) or not np.isfinite(spectrum).all():
        raise ValueError(f"Invalid DiFX output: {vis}")
    np.testing.assert_array_equal(spectrum[:, 0], np.arange(FFT))
    # DiFX: (byte-128)/3.3, FFT * 0.5/L, then average FFT products.
    # Parseval over ALL L bins restores the unnormalized time-domain sum.
    factor = 4 * count * 3.3**2 * SCALE**2
    return np.array([np.sum(spectrum[:, 2] + 1j * spectrum[:, 3]),
                     np.sum(spectrum[:, 6]), np.sum(spectrum[:, 7])]) * factor


def numpy_visibilities(samples):
    data = samples.reshape(8, -1, 10, 2)
    return np.stack([np.einsum("ctp,ctq->cpq", data[:, :, i], data[:, :, j].conj())
                     for i, j in PAIRS], axis=1)


def metrics(got, expected):
    if got.shape != expected.shape:
        raise ValueError("Visibility shapes differ")
    rms, maximum = error_metrics(got, expected)
    power = np.stack([expected[:, b].real.diagonal(axis1=-2, axis2=-1)
                      for b, (i, j) in enumerate(PAIRS) if i == j], axis=1)
    normalization = np.stack([np.sqrt(power[:, i, :, None] * power[:, j, None, :])
                              for i, j in PAIRS], axis=1)
    strong = np.abs(expected) > 0.01 * np.maximum(normalization, np.max(power) * 1e-12)
    ratio = got[strong] / expected[strong]
    return dict(relative_rms=rms, max_autopower_normalized_error=maximum,
                strong_products=int(np.count_nonzero(strong)),
                max_strong_amplitude_ratio_error=float(np.max(np.abs(np.abs(ratio) - 1))),
                max_strong_phase_error_deg=float(np.max(np.abs(np.angle(ratio, deg=True)))),
                passed=rms < 1e-3 and maximum < 2e-3)


def run_starspin(binary, output):
    for name, data in [("config", {"frequency_plan": {"base_frequency_hz": 0,
                        "channel_bandwidth_hz": 781250}}), ("gains", {}),
                       ("eigenvalues", {}), ("delays", {"0": 0})]:
        save_json(output / f"{name}.json", data)
    command = [str(binary), "--packet-format", "codif", "-p", str(output / "input.pcap"),
               "-i", "lo", "-f", "110", "-c", str(output / "config.json"),
               "-g", str(output / "gains.json"), "-e", str(output / "eigenvalues.json"),
               "--delay-file", str(output / "delays.json"), "-v", str(output / "visibilities.hdf5"),
               "--accumulation-length", "1", "--obs-length", "10"]
    save_json(output / "starspin-command.json", command)
    with (output / "starspin.log").open("w") as log:
        subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)


def read_starspin(path, sample_count):
    with h5py.File(path) as file:
        if file.attrs["fine_fft_channels"] != 1 or file.attrs["fine_channels_per_coarse"] != 1:
            raise ValueError("Requires starspin_1_8_coarse")
        np.testing.assert_array_equal(file["baseline_receiver_indices"][:], PAIRS)
        seqs = file["vis_seq_nums"][:]
        missing = file["vis_missing_nums"][:]
        data = file["visibilities"][:]
    if data.shape != (len(seqs), 8, 55, 2, 2, 2) or missing.shape != (len(seqs), 3):
        raise ValueError("Unexpected HDF5 shape")
    intervals = []
    previous_end = -1
    for row, (start, end) in enumerate(seqs):
        start, stop = int(start), int(end) + 64
        if (start < 0 or start % 64 or stop - start != BLOCK or stop > sample_count or
                start <= previous_end or missing[row, 0] != 0 or missing[row, 2] != 0 or
                missing[row, 1] != (stop - start) // 64 * 8):
            raise ValueError("Incomplete, overlapping, or out-of-range Starspin integration")
        previous_end = stop - 1
        intervals.append((start, stop))
    return intervals, data[..., 0].astype(float) + 1j * data[..., 1]


def plot_report(output, rows, has_gpu):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    comparisons = ["difx_vs_numpy"] + (["starspin_vs_difx"] if has_gpu else [])
    for axis, key, threshold, title in zip(axes,
            ["relative_rms", "max_autopower_normalized_error"], [1e-3, 2e-3],
            ["Relative complex RMS", "Maximum autopower-normalized error"]):
        for comparison in comparisons:
            values = [max(row[comparison][key], np.finfo(float).tiny) for row in rows]
            axis.semilogy(range(len(rows)), values, "o-", label=comparison.replace("_", " "))
        axis.axhline(threshold, color="red", linestyle="--", label="acceptance limit")
        axis.set(xlabel="Integration index", ylabel=title)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    figure.suptitle("DiFX zerocorr coarse-channel comparison")
    figure.savefig(output / "comparison.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zerocorr", type=Path, required=True)
    parser.add_argument("--difx-source", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--integrations", type=int, default=10)
    args = parser.parse_args()
    zerocorr = args.zerocorr.resolve(strict=True)
    revision = subprocess.check_output(["git", "-C", str(args.difx_source), "rev-parse", "HEAD"], text=True).strip()
    if revision != DIFX_COMMIT:
        raise ValueError(f"Scaling verified only for DiFX commit {DIFX_COMMIT}")
    subprocess.run(["git", "-C", str(args.difx_source), "diff", "--exit-code", "HEAD", "--",
                    "libraries/mark5access"], check=True, stdout=subprocess.DEVNULL)
    binary = args.binary.resolve(strict=True) if args.binary else None
    output = args.output_dir.resolve()
    if any(c.isspace() for c in str(output)):
        raise ValueError("zerocorr output paths cannot contain whitespace")
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "provenance.json", dict(difx_commit=revision, zerocorr_sha256=sha256(zerocorr),
              starspin_binary=str(binary), starspin_sha256=sha256(binary) if binary else None,
              python=sys.version, numpy=np.__version__, h5py=h5py.__version__,
              runner_sha256=sha256(Path(__file__)),
              decoder_sha256=sha256(Path(__file__).with_name("check_codif_visibility_parity.py")),
              generator_sha256=sha256(Path(__file__).with_name("generate_codif_fixture.py")),
              difx_component="mark5access/zerocorr", format=FORMAT,
              normalization="sum(all FFT bins) * 4 * sample_count * 3.3**2 * 128**2"))
    samples = generate_fixture(output, args.integrations)
    gpu = None
    if binary:
        run_starspin(binary, output)
        intervals, gpu = read_starspin(output / "visibilities.hdf5", samples.shape[1])
        if len(intervals) < args.integrations:
            raise ValueError(f"Only {len(intervals)} complete GPU integrations; need {args.integrations}")
    else:
        intervals = [(i * BLOCK, (i + 1) * BLOCK) for i in range(args.integrations)]
    rows, difx_rows = [], []
    controls = None
    for row, (start, stop) in enumerate(intervals):
        directory = output / f"difx-{row:03}"
        directory.mkdir()
        # DiFX 2.9 short-file seeks can hit premature EOF; slice before packing.
        # Decoder reads ahead at frame boundaries: two zero guard frames are excluded.
        for channel in range(8):
            for stream in range(20):
                write_vdif(directory / f"c{channel}-s{stream}.vdif",
                           np.r_[samples[channel, start:stop, stream], np.zeros(128)], start)
        expected = numpy_visibilities(samples[:, start:stop])
        got = np.empty_like(expected)
        auto_error = 0.0
        for channel in range(8):
            for baseline, (i, j) in enumerate(PAIRS):
                for p in range(2):
                    for q in range(2):
                        a, b = i * 2 + p, j * 2 + q
                        result = run_zerocorr(zerocorr, directory / f"c{channel}-s{a}.vdif",
                            directory / f"c{channel}-s{b}.vdif", 0, stop - start,
                            directory / f"c{channel}-b{baseline}-p{p}{q}")
                        got[channel, baseline, p, q] = result[0]
                        powers = np.sum(np.abs(samples[channel, start:stop, [a, b]])**2, axis=1)
                        auto_error = max(auto_error, float(np.max(np.abs(result[1:].real / powers - 1))))
        record = dict(start_sample=start, stop_sample_exclusive=stop,
                      difx_vs_numpy=metrics(got, expected), max_autopower_relative_error=auto_error)
        if gpu is not None:
            record["starspin_vs_difx"] = metrics(gpu[row], got)
            record["starspin_vs_numpy"] = metrics(gpu[row], expected)
        if controls is None:
            controls = {"conjugation": metrics(got.conj(), expected),
                        "missing_scale": metrics(got / SCALE**2, expected),
                        "polarization_swap": metrics(got[..., ::-1, ::-1], expected)}
            if any(value["passed"] for value in controls.values()):
                raise ValueError("Fixture cannot distinguish a negative control")
        rows.append(record)
        difx_rows.append(got)
        print(json.dumps(dict(integration=row, **record)), flush=True)
    cpu_passed = all(r["difx_vs_numpy"]["passed"] and r["max_autopower_relative_error"] < 1e-3 for r in rows)
    gpu_passed = (all(r["starspin_vs_difx"]["passed"] and r["starspin_vs_numpy"]["passed"] for r in rows)
                  if gpu is not None else None)
    report = dict(scope="coarse-only synthetic single-FPGA, unity gains, zero delays; DiFX zerocorr",
                  cpu_compatibility_passed=cpu_passed, starspin_vs_difx_passed=gpu_passed,
                  complex_visibilities_compared=len(rows) * 8 * 55 * 4,
                  negative_controls=controls, integrations=rows)
    np.savez_compressed(output / "difx_visibilities.npz", visibilities=np.stack(difx_rows), intervals=intervals)
    save_json(output / "comparison.json", report)
    plot_report(output, rows, gpu is not None)
    if not cpu_passed or gpu_passed is False:
        raise AssertionError(f"Comparison failed: {output / 'comparison.json'}")


if __name__ == "__main__":
    main()

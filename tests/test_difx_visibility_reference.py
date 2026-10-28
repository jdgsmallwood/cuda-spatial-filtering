"""Portable checks plus opt-in tests against the actual pinned DiFX executable."""
import os
from pathlib import Path
import struct
import subprocess
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_difx_visibility_parity import (
    BLOCK, PAIRS, SCALE, generate_fixture, metrics, numpy_visibilities,
    read_starspin, run_zerocorr, write_vdif,
)


def test_fixture_roundtrip_and_negative_controls(tmp_path):
    samples = generate_fixture(tmp_path, 1)
    assert samples.shape == (8, 3 * BLOCK, 20)
    expected = numpy_visibilities(samples[:, :BLOCK])
    assert metrics(expected, expected)["passed"]
    for bad in (expected.conj(), expected / SCALE**2, expected[..., ::-1, ::-1]):
        assert not metrics(bad, expected)["passed"]
    assert len({samples[:, :, stream].tobytes() for stream in range(20)}) == 20


def test_vdif_encoding_and_rejects_requantization(tmp_path):
    samples = np.tile(np.array([-128 + 127j, 0 + 1j, 127 - 128j, -1j]), 32) * SCALE
    path = tmp_path / "samples.vdif"
    write_vdif(path, samples)
    data = path.read_bytes()
    for frame in range(2):
        header = struct.unpack_from("<8I", data, frame * 160)
        assert header[:3] == (0, frame, 20)
        assert header[3] >> 31 == 1
        payload = np.frombuffer(data, np.uint8, 128, frame * 160 + 32).astype(float).reshape(64, 2) - 128
        np.testing.assert_array_equal((payload[:, 0] + 1j * payload[:, 1]) * SCALE,
                                      samples[frame * 64:(frame + 1) * 64])
    for bad in (samples + 0.5, samples[:63], samples * 2, samples * np.nan):
        with pytest.raises(ValueError):
            write_vdif(tmp_path / "bad.vdif", bad)
    with pytest.raises(FileExistsError):
        write_vdif(path, samples)


def write_hdf5(path):
    with h5py.File(path, "w") as file:
        file.attrs["fine_fft_channels"] = 1
        file.attrs["fine_channels_per_coarse"] = 1
        file["baseline_receiver_indices"] = PAIRS
        file["vis_seq_nums"] = [[0, BLOCK - 64]]
        file["vis_missing_nums"] = [[0, BLOCK // 64 * 8, 0]]
        file["visibilities"] = np.zeros((1, 8, 55, 2, 2, 2))


@pytest.mark.parametrize("mutation", ["fine", "missing", "range", "mapping", "count"])
def test_rejects_incompatible_gpu_capture(tmp_path, mutation):
    path = tmp_path / "vis.hdf5"
    write_hdf5(path)
    assert read_starspin(path, BLOCK)[0] == [(0, BLOCK)]
    with h5py.File(path, "r+") as file:
        if mutation == "fine":
            file.attrs["fine_fft_channels"] = 32
        elif mutation == "missing":
            file["vis_missing_nums"][0, 0] = 1
        elif mutation == "range":
            file["vis_seq_nums"][0, 1] += 64
        elif mutation == "mapping":
            file["baseline_receiver_indices"][1] = [1, 0]
        else:
            file["vis_missing_nums"][0, 1] -= 1
    with pytest.raises((ValueError, AssertionError)):
        read_starspin(path, BLOCK)


@pytest.mark.skipif(not os.environ.get("DIFX_ZEROCORR"), reason="set DIFX_ZEROCORR for external compatibility gate")
def test_actual_difx_phase_scale_offset_and_environment(tmp_path, monkeypatch):
    # An integer quarter-turn gives exact nonzero imaginary correlation.
    rng = np.random.default_rng(5)
    first = (rng.integers(-20, 21, 1024) + 1j * rng.integers(-20, 21, 1024)) * SCALE
    second = first * 1j
    a, b = tmp_path / "a.vdif", tmp_path / "b.vdif"
    write_vdif(a, first)
    write_vdif(b, second)
    monkeypatch.setenv("SYMMETRIC_8_BIT_VDIF", "1")
    got = run_zerocorr(Path(os.environ["DIFX_ZEROCORR"]), a, b, 128, 512, tmp_path / "check")
    power = np.sum(np.abs(first[128:640])**2)
    np.testing.assert_allclose(got, [-1j * power, power, power], rtol=1e-6, atol=1e-6)
    # Late integrations are sliced before packing to avoid DiFX's short-file seek EOF.
    late = tmp_path / "late.vdif"
    write_vdif(late, np.r_[first[768:], np.zeros(128)], first_sample=98304)
    got = run_zerocorr(Path(os.environ["DIFX_ZEROCORR"]), late, late, 0, 256, tmp_path / "late-check")
    power = np.sum(np.abs(first[768:])**2)
    np.testing.assert_allclose(got, [power, power, power], rtol=1e-6)


def test_nonfinite_visibilities_rejected():
    expected = np.ones((8, 55, 2, 2), complex)
    with pytest.raises(ValueError):
        metrics(expected * np.nan, expected)


@pytest.mark.parametrize("timeout", [False, True])
def test_failed_difx_keeps_diagnostic_log(tmp_path, monkeypatch, timeout):
    binary = tmp_path / "zerocorr"
    binary.write_text(f"#!{sys.executable}\n"
                      "import sys, time\n"
                      "print('decode started', flush=True)\n"
                      "print('decoder diagnostic', file=sys.stderr, flush=True)\n"
                      + ("time.sleep(60)\n" if timeout else "sys.exit(2)\n"))
    binary.chmod(0o755)
    real_run = subprocess.run

    def short_run(*args, **kwargs):
        kwargs["timeout"] = 1
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", short_run)
    error = subprocess.TimeoutExpired if timeout else subprocess.CalledProcessError
    with pytest.raises(error):
        run_zerocorr(binary, tmp_path / "a.vdif", tmp_path / "b.vdif",
                     0, 64, tmp_path / "failed")
    log = (tmp_path / "failed.log").read_text()
    assert "decode started" in log
    assert "decoder diagnostic" in log

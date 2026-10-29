"""Optional GPU application seam; set ADAPTIVE_BEAM_BINARY to the built app."""
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest


@pytest.mark.skipif(not os.environ.get("ADAPTIVE_BEAM_BINARY"), reason="GPU app not selected")
@pytest.mark.parametrize("mode,protection,calibration", [("weights-only", "off", "unity"), ("fine", "rank-one", "unity"), ("coarse", "rank-one", "unity"), ("fine", "rank-one", "gains"), ("coarse", "rank-one", "gains")])
def test_replay_writes_paired_complex_voltages(tmp_path, mode, protection, calibration):
    root = Path(__file__).resolve().parents[1]
    binary = Path(os.environ["ADAPTIVE_BEAM_BINARY"]).resolve()
    sys.path.insert(0, str(root / "scripts"))
    from generate_codif_fixture import fixture_bytes
    capture = tmp_path / "input.pcap"
    capture.write_bytes(fixture_bytes(frames=1024))
    for name, data in {
        "config.json": {"frequency_plan": {"base_frequency_hz": 0, "channel_bandwidth_hz": 781250},
                        "array_location": {"latitude_deg": -30.31, "longitude_deg": 149.55, "height_m": 240},
                        "antenna_positions": {"0": {str(i): {"east": float(i), "north": 0, "up": 0} for i in range(10)}}},
        "delays.json": {"0": 0},
    }.items():
        (tmp_path / name).write_text(json.dumps(data))
    if calibration == "gains":
        gains = {"channel_axis": "fine_processing_index", "weights": {
            str(c): {pol: {str(r): {"real": 1.0, "imag": 0.5} for r in range(10)
                          if not (pol == "XX" and r == 1 and c == 110)} for pol in ("XX", "YY")}
            for c in range(110, 334)}}
        (tmp_path / "gains.json").write_text(json.dumps(gains))
    command = [str(binary), "--decomposition", mode, "--calibration", calibration,
               "--sun-protection", protection, "--steering-utc", "2026-10-29T02:00:00Z", "--packet-format", "codif",
               "-p", str(capture), "--loop",
               "-i", "lo", "-f", "110", "-c", str(tmp_path / "config.json"),
               "-g", str(tmp_path / "gains.json"), "-e", str(tmp_path / "eigenvalues.json"),
               "--delay-file", str(tmp_path / "delays.json"),
               "--beam-output-file", str(tmp_path / "beam.h5"), "--obs-length", "2"]
    if mode != "weights-only":
        command += ["--record-visibilities", "--vis_output_file", str(tmp_path / "vis.h5")]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=180)
    (tmp_path / "run.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    with h5py.File(tmp_path / "beam.h5") as output:
        assert output.attrs["elapsed_seconds"] > 0
        assert output.attrs["successful_input_payload_bytes"] > 0
        assert output.attrs["input_GB_per_s"] > 0
        assert output.attrs["decomposition"] == mode
        assert output.attrs["calibration_mode"] == calibration
        if calibration == "gains":
            exclusions = json.loads(output.attrs["calibration_exclusions_json"])
            assert len(exclusions) == (28 if mode == "coarse" else 1)
            assert all(e["receiver"] == 1 and e["polarization"] == "XX" for e in exclusions)
        assert output.attrs["sun_protection"] == protection
        # The writer stores IEEE half-precision bit patterns as uint16.
        beam = output["beam_data"][:2].view(np.float16)
        assert beam.shape[0] > 0, result.stdout + result.stderr
        assert np.isfinite(beam).all()
        assert np.any(beam != 0)
        if mode == "weights-only":
            np.testing.assert_array_equal(beam[:, :, :, 0], beam[:, :, :, 1])
    if mode != "weights-only":
        with h5py.File(tmp_path / "vis.h5") as output:
            assert output.attrs["decomposition"] == mode
            assert output.attrs["covariance_stage"] == "before-whitening"
            vis = output["visibilities"][:2]
            assert vis.shape[0] > 0
            channels = 8 if mode == "coarse" else beam.shape[1]
            assert vis.shape[1:] == (channels, 55, 2, 2, 2)
            assert np.isfinite(vis).all()
            assert np.any(vis != 0)
            for receiver in range(10):
                baseline = receiver * (receiver + 1) // 2 + receiver
                for pol in range(2):
                    auto = vis[:, :, baseline, pol, pol]
                    assert (auto[..., 0] >= 0).all()
                    np.testing.assert_allclose(auto[..., 1], 0, atol=1e-3)


@pytest.mark.skipif(not os.environ.get("ADAPTIVE_BEAM_BINARY"), reason="GPU app not selected")
def test_unattended_missing_calibration_requires_explicit_choice(tmp_path):
    binary = Path(os.environ["ADAPTIVE_BEAM_BINARY"]).resolve()
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "delays.json").write_text('{"0": 0}')
    result = subprocess.run(
        [str(binary), "-i", "lo", "-f", "110", "-c", str(tmp_path / "config.json"),
         "--delay-file", str(tmp_path / "delays.json"),
         "-g", str(tmp_path / "missing.json"), "--beam-output-file", str(tmp_path / "beam.h5")],
        cwd=tmp_path, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert "Choose --calibration" in result.stderr
    assert not (tmp_path / "beam.h5").exists()


@pytest.mark.skipif(not os.environ.get("ADAPTIVE_BEAM_BINARY"), reason="GPU app not selected")
def test_interactive_missing_calibration_can_abort_without_creating_output(tmp_path):
    import pty
    master, slave = pty.openpty()
    try:
        binary = Path(os.environ["ADAPTIVE_BEAM_BINARY"]).resolve()
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "delays.json").write_text('{"0": 0}')
        os.write(master, b"3\n")
        result = subprocess.run(
            [str(binary), "-i", "lo", "-f", "110", "-c", str(tmp_path / "config.json"),
             "--delay-file", str(tmp_path / "delays.json"), "-g", str(tmp_path / "missing.json"),
             "--beam-output-file", str(tmp_path / "beam.h5")],
            cwd=tmp_path, stdin=slave, capture_output=True, text=True, timeout=20)
        assert result.returncode != 0
        assert "Observation cancelled" in result.stderr
        assert not (tmp_path / "beam.h5").exists()
    finally:
        os.close(master)
        os.close(slave)

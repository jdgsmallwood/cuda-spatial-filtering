"""Optional 8 GB GPU regression for full-array fine-channel allocation."""
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest


@pytest.mark.skipif(not os.environ.get("ADAPTIVE_BEAM_FULL_BINARY"), reason="full-array app not selected")
@pytest.mark.parametrize("mode,protection,buffers", [("coarse", "off", 1), ("coarse", "rank-one", 1), ("coarse", "rank-one", 3), ("fine", "off", 1), ("fine", "rank-one", 1), ("weights-only", "off", 3)])
def test_full_array_replay_fits_and_writes_finite_voltages(tmp_path, mode, protection, buffers):
    binary = Path(os.environ["ADAPTIVE_BEAM_FULL_BINARY"]).resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from generate_codif_fixture import fixture_bytes
    (tmp_path / "input.pcap").write_bytes(fixture_bytes(frames=2, channels=40, fpgas=4))
    (tmp_path / "config.json").write_text(json.dumps({
        "frequency_plan": {"base_frequency_hz": 0, "channel_bandwidth_hz": 781250},
        "array_location": {"latitude_deg": -30.31, "longitude_deg": 149.55, "height_m": 240},
        "network_interfaces": {str(f): "lo" for f in range(4)},
        "antenna_positions": {str(f): {str(f * 10 + r): {"east": float(f * 10 + r), "north": 0, "up": 0}
                                      for r in range(10)} for f in range(4)}}))
    (tmp_path / "delays.json").write_text(json.dumps({str(f): 0 for f in range(4)}))
    result = subprocess.run(
        [str(binary), "--decomposition", mode, "--gpu-buffers", str(buffers), "--null-rank", "2", "--calibration", "unity",
         "--sun-protection", protection, "--steering-utc", "2026-10-29T02:00:00Z",
         "--packet-format", "codif", "-p", str(tmp_path / "input.pcap"), "--loop",
         "-i", "0,1,2,3", "-f", "110", "-c", str(tmp_path / "config.json"),
         "--delay-file", str(tmp_path / "delays.json"),
         "--beam-output-file", str(tmp_path / "beam.h5"), "--obs-length", "1"],
        cwd=tmp_path, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    with h5py.File(tmp_path / "beam.h5") as output:
        assert output.attrs["eigensolver_failures"] == 0
        assert output.attrs["gpu_buffers"] == buffers
        beam = output["beam_data"][:1].view(np.float16)
        assert beam.shape == (1, 1120, 2, 2, 512, 2)
        assert np.isfinite(beam).all()
        assert np.any(beam != 0)
        assert output.attrs["successful_input_payload_bytes"] > 0

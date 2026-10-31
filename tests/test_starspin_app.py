"""Optional application startup regression; set STARSPIN_BINARY to the built app."""
import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.skipif(not os.environ.get("STARSPIN_BINARY"), reason="GPU app not selected")
def test_correlation_starts_without_gains_or_eigenvalue_files(tmp_path):
    binary = Path(os.environ["STARSPIN_BINARY"]).resolve()
    capture = Path(__file__).parent / "fixtures" / "codif_synthetic.pcap"
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "delays.json").write_text(json.dumps({"0": 0}))
    output = tmp_path / "vis.h5"
    result = subprocess.run(
        [str(binary), "--packet-format", "codif", "-p", str(capture.resolve()),
         "-i", "lo", "-f", "110", "-c", str(tmp_path / "config.json"),
         "-g", str(tmp_path / "absent-gains.json"),
         "-e", str(tmp_path / "absent-eigenvalues.json"),
         "--delay-file", str(tmp_path / "delays.json"),
         "--vis_output_file", str(output), "--obs-length", "1"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.exists()

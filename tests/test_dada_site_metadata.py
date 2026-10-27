from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/generate_fine_dada_header.py'


def test_telescope_metadata_is_explicit(tmp_path):
    output = tmp_path / 'header.hdr'
    command = [sys.executable, str(SCRIPT), '--output', str(output)]
    missing = subprocess.run(command, capture_output=True, text=True)
    assert missing.returncode != 0
    assert '--telescope' in missing.stderr
    subprocess.run(command + ['--telescope', 'SYNTHETIC'], check=True, capture_output=True)
    fields = dict(line.split(maxsplit=1) for line in output.read_bytes().rstrip(b'\0').decode().splitlines())
    assert fields['TELESCOPE'] == 'SYNTHETIC'

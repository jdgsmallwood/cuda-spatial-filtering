"""Benchmark planning uses the real CLI without starting CUDA or making captures."""
import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark/adaptive_beam.py"


def invoke(tmp_path, *args):
    return subprocess.run([sys.executable, str(SCRIPT), "/missing/app", str(tmp_path / "output"),
                           "--dry-run", *args], capture_output=True, text=True)


def test_targeted_coarse_comparison_needs_no_temporary_runner(tmp_path):
    result = invoke(tmp_path, "--modes", "coarse", "--ranks", "2", "--skip-visibilities")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [["coarse", 2, False, False], ["coarse", 2, True, False]]
    assert not (tmp_path / "output").exists()


def test_default_matrix_and_weights_only_remain_available(tmp_path):
    result = invoke(tmp_path)
    assert result.returncode == 0, result.stderr
    cases = json.loads(result.stdout)
    assert len(cases) == 19
    assert cases[0] == ["weights-only", 0, False, False]
    assert cases[-2:] == [["coarse", 2, True, True], ["fine", 2, True, True]]
    result = invoke(tmp_path, "--weights-only")
    assert json.loads(result.stdout) == [["weights-only", 0, False, False]]


def test_reject_nonfinite_duration_before_creating_artifacts(tmp_path):
    result = invoke(tmp_path, "--seconds", "nan")
    assert result.returncode == 2
    assert "finite" in result.stderr
    assert not (tmp_path / "output").exists()

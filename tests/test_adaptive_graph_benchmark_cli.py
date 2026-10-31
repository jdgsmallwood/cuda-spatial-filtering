"""Exercise the benchmark runner CLI with an executable test double."""
import json
from pathlib import Path
import subprocess
import sys

RUNNER = Path(__file__).resolve().parents[1] / "scripts/benchmark/adaptive_graphs.py"


def test_graph_and_eager_runs_require_distinct_capture_evidence(tmp_path):
    binary = tmp_path / "harness"
    binary.write_text('''#!/usr/bin/env python3
import os,sys
n,b,mode=sys.argv[1:4]
graphs=b if os.environ.get("SPATIAL_REQUIRE_CUDA_GRAPH") else "0"
assert bool(os.environ.get("SPATIAL_DISABLE_CUDA_GRAPH")) != bool(os.environ.get("SPATIAL_REQUIRE_CUDA_GRAPH"))
print(f"mode={mode} protected=1 channels=40 fine_bins=28 gpu_buffers={b} captured_graphs={graphs} input_bytes_per_block=104857600 blocks={n} seconds=2 input_GB_per_s={int(n)*104857600/2/1e9}")
''')
    binary.chmod(0o755)
    out = tmp_path / "results"
    result = subprocess.run([sys.executable, str(RUNNER), str(binary), str(out),
        "--modes", "coarse", "--buffers", "2", "--protection", "rank-one", "--repeats", "1"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rows = json.loads((out / "runs.json").read_text())
    assert {r['execution']:r['captured_graphs'] for r in rows} == {'eager':0,'graph':2}
    assert all(r['status']=='ok' for r in rows)


def test_silent_eager_fallback_is_a_failed_graph_benchmark(tmp_path):
    binary = tmp_path / "fallback"
    binary.write_text('''#!/usr/bin/env python3
import sys
n,b,mode=sys.argv[1:4]
print(f"mode={mode} protected=1 gpu_buffers={b} captured_graphs=0 blocks={n} input_bytes_per_block=104857600 seconds=2 input_GB_per_s={int(n)*104857600/2/1e9}")
''')
    binary.chmod(0o755)
    out=tmp_path/'results'
    result=subprocess.run([sys.executable,str(RUNNER),str(binary),str(out),
        '--modes','fine','--buffers','1','--protection','rank-one','--repeats','1'],capture_output=True,text=True)
    assert result.returncode != 0
    rows=json.loads((out/'runs.json').read_text())
    assert next(r for r in rows if r['execution']=='graph')['status']=='failed'
    assert next(r for r in rows if r['execution']=='eager')['status']=='ok'

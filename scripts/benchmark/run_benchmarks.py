#!/usr/bin/env python3
"""Benchmark sweep for the cuda-spatial-filtering pipeline.

Runs the per-component throughput microbenchmarks built in `apps/`:

  - bench_capture   -- raw kernel-socket packet capture (recvmmsg ingestion)
  - bench_processor -- ring-buffer reassembly, 3 configs (1ch/1fpga, 8ch/1fpga, 8ch/4fpga)
  - gpu_benchmark   -- GPU correlate+beamform pipeline (LambdaCorrBeamOnlyGPUPipeline)
  - bench_writers   -- HDF5 visibilities/eigendata writer throughput, 3 configs

and prints a summary table of packets/sec and GB/sec for each stage so the
slowest stage (the bottleneck) is obvious at a glance.

Optionally (--with-observe) also runs the full `observe` pipeline against a
looping multi-FPGA PCAP replay, as an end-to-end comparison point against the
isolated per-stage numbers above.

Example:

    python3 scripts/benchmark/run_benchmarks.py \\
        --build-dir /workspace/build_apps --duration 10 \\
        --interfaces enp134s0np0,enp175s0np0,enp216s0np0,enp220s0np0
"""

import argparse
import datetime
import json
import os
import re
import shlex
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_/]*)=([-+0-9.eE]+)")
PROCESSOR_HDR_RE = re.compile(
    r"\[Processor ch=(\d+) fpga=(\d+) rx=(\d+)\]"
)
VIS_WRITER_HDR_RE = re.compile(
    r"\[Vis Writer ch=(\d+) fpga=(\d+) rx=(\d+)\]"
)
EIGEN_WRITER_HDR_RE = re.compile(
    r"\[Eigen Writer ch=(\d+) fpga=(\d+) rx=(\d+)\]"
)


def parse_kv_line(line: str) -> dict:
    """Parse a `key=value key2=value2 ...` summary line into a dict of floats."""
    out = {}
    for key, value in KV_RE.findall(line):
        try:
            out[key] = float(value)
        except ValueError:
            pass
    return out


def find_summary_lines(stdout: str, prefix: str) -> list:
    return [line for line in stdout.splitlines() if line.startswith(prefix)]


def run(cmd, cwd=None, env=None, timeout=None, label=None):
    print(f"\n=== {label or cmd[0]} ===")
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}")
    try:
        result = subprocess.run(
            cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        print(f"!! {label or cmd[0]} timed out after {timeout}s")
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        print(stdout)
        print(stderr, file=sys.stderr)
        return stdout
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        print(f"!! {label or cmd[0]} exited with code {result.returncode}")
    return result.stdout


def collect_meta() -> dict:
    """Collect git commit, branch, hostname, and timestamp for the result JSON."""
    meta = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "hostname": socket.gethostname(),
        "git_commit": None,
        "git_branch": None,
    }
    try:
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        meta["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        pass
    return meta


def _ensure_loopback_pcap(pcap_path: Path) -> bool:
    """Ensure a capture file exists at pcap_path for loopback replay.

    Prefers the repo's example_packets.pcapng (a real LAMBDA capture), then
    falls back to generating a synthetic one with create_pcap.py.
    Returns True if a file is ready, False if neither worked.
    """
    if pcap_path.exists():
        return True
    # Use the real capture file if present in the repo root
    example = REPO_ROOT / "example_packets.pcapng"
    if example.exists():
        print(f"Using {example} for loopback capture (symlinking to {pcap_path})")
        pcap_path.parent.mkdir(parents=True, exist_ok=True)
        pcap_path.symlink_to(example)
        return True
    print(f"\n=== Generating loopback PCAP at {pcap_path} ===")
    gen_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "create_pcap.py"),
        "--number_packets", "500",
        "--number_channels", "8",
        "--number_receivers", "10",
        "--output", str(pcap_path),
    ]
    try:
        subprocess.check_call(gen_cmd)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"!! create_pcap.py failed (exit {exc.returncode}); skipping bench_capture")
        return False


def bench_capture(args, env, results):
    binary = args.build_dir / "apps" / "bench_capture"

    # On loopback (default interface "lo"), auto-generate a pcap if none supplied.
    loopback_pcap = args.capture_loopback_pcap
    if loopback_pcap is None and args.interfaces == "lo":
        loopback_pcap = Path("/tmp/bench_loopback.pcap")

    if loopback_pcap is not None and not _ensure_loopback_pcap(loopback_pcap):
        return

    cmd = [
        str(binary),
        "-i", args.interfaces,
        "-p", str(args.port),
        "--duration", str(args.capture_duration),
        "--recv-buffer-size", str(args.recv_buffer_size),
    ]

    loopback_proc = None
    if loopback_pcap is not None:
        sender = args.build_dir / "apps" / "udp_sender"
        loop_cmd = (
            f"while true; do {shlex.quote(str(sender))} "
            f"{shlex.quote(str(loopback_pcap))} "
            f"{shlex.quote(str(args.port))} 127.0.0.1 >/dev/null 2>&1; done"
        )
        loopback_proc = subprocess.Popen(
            ["bash", "-c", loop_cmd], preexec_fn=os.setsid
        )
        time.sleep(0.5)

    try:
        stdout = run(
            cmd, env=env, timeout=args.capture_duration + 30, label="bench_capture"
        )
    finally:
        if loopback_proc is not None:
            os.killpg(os.getpgid(loopback_proc.pid), signal.SIGTERM)
            loopback_proc.wait()

    for line in find_summary_lines(stdout, "[Capture TOTAL]"):
        results["capture"] = parse_kv_line(line)


def bench_processor(args, env, results):
    binary = args.build_dir / "apps" / "bench_processor"
    cmd = [str(binary), "--duration", str(args.processor_duration)]
    # 3 configs each running for processor_duration
    timeout = args.processor_duration * 3 + 60
    stdout = run(cmd, env=env, timeout=timeout, label="bench_processor")

    processor_rows = []
    for line in stdout.splitlines():
        m = PROCESSOR_HDR_RE.match(line)
        if m:
            row = {
                "nr_channels": int(m.group(1)),
                "nr_fpga_sources": int(m.group(2)),
                "nr_receivers": int(m.group(3)),
            }
            row.update(parse_kv_line(line))
            processor_rows.append(row)

    if processor_rows:
        results["processor"] = processor_rows
    else:
        # Backward compatibility: single-config old format
        for line in find_summary_lines(stdout, "[Packet Processor]"):
            results["processor"] = parse_kv_line(line)


def gpu_benchmark(args, env, results):
    binary = args.build_dir / "apps" / "gpu_benchmark"
    cmd = [str(binary), "--duration", str(args.gpu_duration)]
    stdout = run(cmd, env=env, timeout=args.gpu_duration + 60, label="gpu_benchmark")
    gpu_rows = []
    for line in stdout.splitlines():
        if line.startswith("[GPU Pipeline]"):
            row = parse_kv_line(line)
            gpu_rows.append(row)
    if gpu_rows:
        results["gpu_pipeline"] = gpu_rows[0] if len(gpu_rows) == 1 else gpu_rows
    else:
        print("!! gpu_benchmark: no '[GPU Pipeline]' summary line found -- the "
              "process likely crashed before finishing a full run (see "
              "scripts/benchmark/README.md, 'Known issues')")


def bench_writers(args, env, results):
    binary = args.build_dir / "apps" / "bench_writers"
    out_dir = args.writer_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary),
        "--duration", str(args.writer_duration),
        "--num-blocks", str(args.writer_num_blocks),
        "-o", str(out_dir),
    ]
    # 3 configs × 2 writers each running for writer_duration
    timeout = args.writer_duration * 6 + 60
    stdout = run(cmd, env=env, timeout=timeout, label="bench_writers")

    vis_rows = []
    eigen_rows = []
    for line in stdout.splitlines():
        vm = VIS_WRITER_HDR_RE.match(line)
        em = EIGEN_WRITER_HDR_RE.match(line)
        if vm:
            row = {
                "nr_channels": int(vm.group(1)),
                "nr_fpga_sources": int(vm.group(2)),
                "nr_receivers": int(vm.group(3)),
            }
            row.update(parse_kv_line(line))
            vis_rows.append(row)
        elif em:
            row = {
                "nr_channels": int(em.group(1)),
                "nr_fpga_sources": int(em.group(2)),
                "nr_receivers": int(em.group(3)),
            }
            row.update(parse_kv_line(line))
            eigen_rows.append(row)

    if vis_rows:
        results["vis_writer"] = vis_rows
    else:
        for line in find_summary_lines(stdout, "[Visibilities Writer]"):
            results["vis_writer"] = parse_kv_line(line)

    if eigen_rows:
        results["eigen_writer"] = eigen_rows
    else:
        for line in find_summary_lines(stdout, "[Eigen Writer]"):
            results["eigen_writer"] = parse_kv_line(line)


def pcap_packet_size(pcap_path: Path) -> int:
    """Read the on-the-wire length of the first packet record in a pcap file."""
    with open(pcap_path, "rb") as f:
        global_header = f.read(24)
        (magic,) = struct.unpack("I", global_header[:4])
        endian = "<" if magic == 0xA1B2C3D4 else ">"
        rec_header = f.read(16)
        _, _, incl_len, _ = struct.unpack(endian + "IIII", rec_header)
        return incl_len


def observe_benchmark(args, env, results):
    pcap_path = args.observe_pcap
    if not pcap_path.exists():
        # Prefer the real capture file in the repo root
        example = REPO_ROOT / "example_packets.pcapng"
        if example.exists():
            print(f"\n=== Using {example} for observe replay ===")
            pcap_path.parent.mkdir(parents=True, exist_ok=True)
            pcap_path.symlink_to(example)
        else:
            print(f"\n=== Generating end-to-end PCAP replay file at {pcap_path} ===")
            gen_cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "create_pcap.py"),
                "--number_channels", str(args.observe_channels),
                "--number_receivers", "10",
                "--number_packets", str(args.observe_packets),
                "--output", str(pcap_path),
            ]
            run(gen_cmd, label="create_pcap")

    pkt_bytes = pcap_packet_size(pcap_path)

    binary = args.build_dir / "apps" / "observe"
    cmd = [
        str(binary),
        "-p", str(pcap_path),
        "-l",
        "-i", args.observe_ifnames,
        "-c", str(REPO_ROOT / "config.json"),
        "-g", str(REPO_ROOT / "weights.json"),
        "-e", str(REPO_ROOT / "nr-signal-eigenvalues.json"),
        "-y", str(REPO_ROOT / "alveo_delays.json"),
    ]
    print(f"\n=== observe (end-to-end PCAP replay) ===")
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.time() + args.observe_duration
    lines = []
    try:
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            print(line, end="")
            lines.append(line)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            remaining, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            remaining, _ = proc.communicate()
        print(remaining, end="")
        lines.extend(remaining.splitlines(keepends=True))

    stats_re = re.compile(
        r"Stats: Received=(\d+), Processed=(\d+), Missing=(\d+), Discarded=(\d+)"
    )
    stats = [stats_re.search(line) for line in lines]
    stats = [m for m in stats if m]
    if len(stats) >= 2:
        first, last = stats[0], stats[-1]
        elapsed = (len(stats) - 1) * 5.0  # printed every 5s
        processed_delta = int(last.group(2)) - int(first.group(2))
        pps = processed_delta / elapsed if elapsed > 0 else 0.0
        results["observe_e2e"] = {
            "packets/sec": pps,
            "GB/sec": pps * pkt_bytes / 1e9,
            "elapsed": elapsed,
            "packets_missing": float(last.group(3)),
            "packets_discarded": float(last.group(4)),
        }
    else:
        print("!! observe: not enough 'Stats:' lines captured to compute a rate")


def _config_label(row: dict) -> str:
    return f"{int(row.get('nr_channels', '?'))}ch/{int(row.get('nr_fpga_sources', '?'))}fpga"


def print_summary(results):
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    header = f"{'stage':<20}{'config':<15}{'packets|blocks/sec':>22}{'GB/sec':>15}"
    print(header)
    print("-" * len(header))

    rate_keys = ("packets/sec", "blocks/sec", "runs/sec", "packets|blocks/sec")
    gb_key_candidates = ("GB/sec", "output_GB/sec")

    rows_for_bottleneck = []

    def emit_row(name, config_label, metrics):
        rate = next((metrics[k] for k in rate_keys if k in metrics), None)
        gb = next((metrics[k] for k in gb_key_candidates if k in metrics), None)
        rate_str = f"{rate:,.2f}" if rate is not None else "n/a"
        gb_str = f"{gb:.4f}" if gb is not None else "n/a"
        print(f"{name:<20}{config_label:<15}{rate_str:>22}{gb_str:>15}")
        if gb is not None:
            rows_for_bottleneck.append((f"{name}[{config_label}]", gb))

    for name, val in results.items():
        if name == "meta" or not val:
            continue
        if isinstance(val, list):
            for row in val:
                emit_row(name, _config_label(row), row)
        else:
            emit_row(name, "—", val)

    if rows_for_bottleneck:
        bottleneck = min(rows_for_bottleneck, key=lambda r: r[1])
        print("-" * len(header))
        print(f"Bottleneck (lowest GB/sec): {bottleneck[0]} @ {bottleneck[1]:.4f} GB/sec")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--build-dir", type=Path, default=REPO_ROOT / "build_apps",
        help="CMake build directory containing apps/bench_* binaries"
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Default duration (seconds) for each microbenchmark stage"
    )
    parser.add_argument(
        "--cuda-home", default="/opt/conda/targets/x86_64-linux",
        help="CUDA_HOME for NVRTC (needed by gpu_benchmark/observe)"
    )
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help="Write the full results dict as JSON to this path (includes git metadata)"
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        choices=["capture", "processor", "gpu", "writers", "observe"],
        help="Stages to skip"
    )

    cap = parser.add_argument_group("bench_capture")
    cap.add_argument("--interfaces", default="lo",
                      help="Comma-separated NICs to capture on (or 'lo' for loopback self-test)")
    cap.add_argument("--port", type=int, default=36001)
    cap.add_argument("--recv-buffer-size", type=int, default=64 * 1024 * 1024)
    cap.add_argument("--capture-duration", type=float, default=None)
    cap.add_argument("--capture-loopback-pcap", type=Path, default=None,
                      help="If set, continuously replay this pcap over 'lo' with "
                           "udp_sender for the duration of bench_capture")

    proc = parser.add_argument_group("bench_processor")
    proc.add_argument("--processor-duration", type=float, default=None,
                       help="Duration per config (binary runs 3 configs sequentially)")

    gpu = parser.add_argument_group("gpu_benchmark")
    gpu.add_argument("--gpu-duration", type=float, default=None)

    wr = parser.add_argument_group("bench_writers")
    wr.add_argument("--writer-duration", type=float, default=None,
                     help="Duration per config per writer (binary runs 3 configs × 2 writers)")
    wr.add_argument("--writer-num-blocks", type=int, default=100)
    wr.add_argument("--writer-output-dir", type=Path,
                     default=Path("/tmp/bench_writers_out"))

    obs = parser.add_argument_group("observe (end-to-end, optional)")
    obs.add_argument("--with-observe", action="store_true",
                      help="Also run the full `observe` pipeline against a looping "
                           "multi-FPGA PCAP replay")
    obs.add_argument("--observe-duration", type=float, default=30.0)
    obs.add_argument("--observe-pcap", type=Path,
                      default=Path("/tmp/bench_observe_4fpga.pcap"))
    obs.add_argument("--observe-channels", type=int, default=8)
    obs.add_argument("--observe-packets", type=int, default=300)
    obs.add_argument("--observe-ifnames",
                      default="enp216s0np0,enp175s0np0,enp134s0np0,enp000s0np0",
                      help="Comma-separated fake/real ifnames, one per FPGA")

    args = parser.parse_args()

    for name in ("capture_duration", "processor_duration", "gpu_duration", "writer_duration"):
        if getattr(args, name) is None:
            setattr(args, name, args.duration)

    env = os.environ.copy()
    env.setdefault("CUDA_HOME", args.cuda_home)

    results = {"meta": collect_meta()}

    if "capture" not in args.skip:
        bench_capture(args, env, results)
    if "processor" not in args.skip:
        bench_processor(args, env, results)
    if "gpu" not in args.skip:
        gpu_benchmark(args, env, results)
    if "writers" not in args.skip:
        bench_writers(args, env, results)
    if args.with_observe and "observe" not in args.skip:
        observe_benchmark(args, env, results)

    print_summary(results)

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"\nWrote results to {args.json_out}")


if __name__ == "__main__":
    main()

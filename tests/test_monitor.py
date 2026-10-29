"""CPU-only C++ monitor checks; optional compiler/headers, no CUDA or real Redis.

Run integration mode separately with monitoring/check_stack.py.
"""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def monitor_binary(tmp_path_factory):
    def unavailable(reason):
        if os.environ.get("SPATIAL_REQUIRE_MONITOR_TESTS") == "1":
            pytest.fail(reason)
        pytest.skip(reason)

    compiler = shutil.which("g++")
    if not compiler:
        unavailable("g++ unavailable")
    probe = subprocess.run([compiler, "-std=c++17", "-x", "c++", "-fsyntax-only", "-"],
                           input="#include <hiredis/hiredis.h>\n#include <nlohmann/json.hpp>\n", text=True,
                           capture_output=True)
    if probe.returncode:
        unavailable("install libhiredis-dev and nlohmann-json3-dev for monitor checks\n" + probe.stderr)
    binary = tmp_path_factory.mktemp("monitor") / "test-monitor"
    subprocess.run([compiler, "-std=c++17", "-O1", "-pthread", "-I", str(ROOT / "include"),
                    str(ROOT / "tests/test_monitor.cpp"), "-lhiredis", "-o", str(binary)], check=True)
    return binary


def test_monitor_math_and_proc(monitor_binary):
    subprocess.run([monitor_binary], check=True, timeout=10)


def test_missing_redis_never_blocks(monitor_binary):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]  # bound but not listening
        result = subprocess.run([monitor_binary, "publisher", str(port)], capture_output=True,
                                text=True, check=True, timeout=6)
    assert "failures=0" not in result.stdout
    assert "capture continues" in result.stderr


def test_unresponsive_server_and_recovery(monitor_binary):
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        server.settimeout(0.1)
        port = server.getsockname()[1]
        done = threading.Event()
        commands = []
        started = time.monotonic()

        def serve():
            while not done.is_set():
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                with connection:
                    connection.settimeout(0.1)
                    stream = connection.makefile("rb")
                    try:
                        while not done.is_set():
                            line = stream.readline()
                            if not line:
                                break
                            count = int(line[1:])
                            command = []
                            for _ in range(count):
                                size = int(stream.readline()[1:])
                                command.append(stream.read(size))
                                assert stream.read(2) == b"\r\n"
                            commands.append(command)
                            # First second: accept but never reply. Then recover.
                            if time.monotonic() - started >= 1:
                                connection.sendall(b"+OK\r\n")
                    except (OSError, ValueError):
                        pass
                    finally:
                        stream.close()
        thread = threading.Thread(target=serve)
        thread.start()
        try:
            result = subprocess.run([monitor_binary, "publisher", str(port)], text=True,
                                    capture_output=True, check=True, timeout=6)
        finally:
            done.set()
            thread.join(timeout=2)
        assert not thread.is_alive()
        assert "failures=0" not in result.stdout
        assert "published=0 " not in result.stdout
        assert any(command[-1] == b"299" for command in commands)
        assert sum(len(command) > 1 and command[1] == b"monitor-test:setup" for command in commands) >= 2


def test_dashboard_and_optional_stack_contract():
    dashboard = json.loads((ROOT / "monitoring/grafana/dashboards/starspin.json").read_text())
    assert dashboard["refresh"] == "5s"
    assert {v["name"] for v in dashboard["templating"]["list"]} == {"run", "input", "baseline"}
    assert any(p["type"] == "heatmap" for p in dashboard["panels"])
    for panel in dashboard["panels"]:
        if panel["type"] == "xychart":
            assert panel["pluginVersion"] == "12.2.1"  # prevent migration as a pre-11.1 chart
            assert panel["options"]["series"][0]["frame"]["matcher"]["id"] == "byIndex"
        for target in panel["targets"]:
            assert target["command"] in {"ts.range", "ts.mrange", "json.get"}
            assert "$run" in target.get("keyName", target.get("filter", ""))


def test_shutdown_interrupts_a_continuously_trickling_reply(monitor_binary):
    # A socket timeout alone is insufficient if each fragment arrives in time.
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        server.settimeout(5)
        port = server.getsockname()[1]
        done = threading.Event()

        def trickle():
            connection, _ = server.accept()
            with connection:
                try:
                    connection.sendall(b"$1000000\r\n")
                    while not done.wait(0.02):
                        connection.sendall(b"x")
                except OSError:
                    pass
        thread = threading.Thread(target=trickle)
        thread.start()
        try:
            subprocess.run([monitor_binary, "publisher", str(port)], check=True, timeout=6,
                           capture_output=True)
        finally:
            done.set()
            thread.join(timeout=2)
        assert not thread.is_alive()

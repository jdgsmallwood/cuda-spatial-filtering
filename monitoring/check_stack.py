#!/usr/bin/env python3
"""Read-only assertions against a running test-monitor publish run and Grafana.

Use the CPU-only test fixture, not an observing run: expected power is 4, 8.
"""
import argparse
import json
from pathlib import Path
import socket
import time
import urllib.request


def redis_command(port, *args):
    with socket.create_connection(("127.0.0.1", port), timeout=2) as connection:
        connection.sendall(b"*%d\r\n" % len(args) + b"".join(
            b"$%d\r\n%s\r\n" % (len(str(arg).encode()), str(arg).encode()) for arg in args))
        with connection.makefile("rb") as stream:
            def read():
                line = stream.readline()
                kind, value = line[:1], line[1:-2]
                if kind == b"-":
                    raise RuntimeError(value.decode())
                if kind == b"+":
                    return value.decode()
                if kind == b":":
                    return int(value)
                if kind == b"$":
                    if int(value) == -1:
                        return None
                    data = stream.read(int(value))
                    assert stream.read(2) == b"\r\n"
                    return data.decode()
                if kind == b"*":
                    return [read() for _ in range(int(value))]
                raise RuntimeError(f"Invalid RESP reply: {line!r}")
            return read()


def check(port, grafana):
    run = redis_command(port, "GET", "starspin:latest")
    assert run, "No monitoring sample published"
    prefix = "starspin:" + run
    spectrum = json.loads(redis_command(port, "JSON.GET", prefix + ":spectrum:rx0_p0", "."))
    assert [float(row["power"]) for row in spectrum] == [4, 8]
    cross = json.loads(redis_command(port, "JSON.GET", prefix + ":spectrum:rx0_rx1_p0", "."))
    assert float(cross[0]["amplitude"]) == 5
    assert cross[1]["phase_deg"] == "NaN"
    assert 0 < redis_command(port, "TTL", prefix + ":spectrum:rx0_p0") <= 86400
    assert float(redis_command(port, "TS.GET", prefix + ":power_rx0_p0")[1]) == 6
    assert float(redis_command(port, "TS.GET", prefix + ":process_rss_bytes")[1]) > 0
    history = json.loads(redis_command(port, "JSON.GET", prefix + ":waterfall", "."))
    assert 0 < len(history) <= 240
    assert {float(row["power"]) for row in history} == {4, 8}
    dashboard = json.loads((Path(__file__).parent / "grafana/dashboards/starspin.json").read_text())
    count = 0
    for panel in dashboard["panels"]:
        for target in panel["targets"]:
            target = json.loads(json.dumps(target).replace("$run", run).replace("$input", "rx0_p0").replace("$baseline", "rx0_rx1_p0"))
            target.update(datasource={"uid": "starspin-redis", "type": "redis-datasource"}, intervalMs=1000, maxDataPoints=1000)
            payload = {"from": str(int(time.time() * 1000) - 3600000), "to": str(int(time.time() * 1000)), "queries": [target]}
            request = urllib.request.Request(grafana + "/api/ds/query", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            response = json.load(urllib.request.urlopen(request, timeout=10))
            for result in response["results"].values():
                assert not result.get("error"), (panel["title"], result)
                assert result.get("frames"), panel["title"]
                assert any(frame["data"]["values"] and frame["data"]["values"][0] for frame in result["frames"]), panel["title"]
            count += 1
    print(json.dumps({"run": run, "dashboard_queries_passed": count, "known_spectra_passed": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-port", type=int, default=6380)
    parser.add_argument("--grafana", default="http://127.0.0.1:3001")
    args = parser.parse_args()
    check(args.redis_port, args.grafana)

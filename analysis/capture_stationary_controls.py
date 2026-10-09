#!/usr/bin/env python3
"""Same-boot, same-tuning PN repeats and antenna-off raw packet controls.

This changes ADC-box source selection on all four boards, preserves channel
tuning, and restores gain 20 / antennas on / PN off in a finally block.
Run only after checking the current boot and packet delay solution.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


CONSOLE = "/data/BLACKMESA_0/lambda/lambda_firmware/control_software/.venv/bin/lambda_uconsole"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--config", type=Path, default=Path("config.json"))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    nics = json.loads(a.config.read_text())["network_interfaces"]
    events = []

    def record(**data):
        events.append(dict(utc=datetime.now(timezone.utc).isoformat(), **data))
        (a.output / "manifest.json").write_text(json.dumps(dict(
            gain_setting=20, tuning_changed=False, expected_channels=[400, 439],
            events=events), indent=2) + "\n")
        print(events[-1], flush=True)

    def adc(label, pn, antennas):
        for board in range(4):
            result = subprocess.run([CONSOLE, "-k", "lambda", f"10.0.{board}.10"],
                input=f"adc\n20\n31\n{'Y' if pn else 'N'}\n{antennas}\nY\nr\nq\n",
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=50)
            (a.output / f"{label}_fpga{board}.log").write_text(result.stdout)
            expected = f"Last ADC config: PN seq {'ON' if pn else 'OFF'}, antennas=0b{antennas:04b}"
            record(operation="adc", label=label, board=board, pn=pn,
                   antennas=antennas, exit_code=result.returncode,
                   software_readback_matches=expected in result.stdout)
            if result.returncode or expected not in result.stdout:
                raise RuntimeError(f"ADC command/readback failed on FPGA {board}")

    def capture(label, first, count, packets):
        directory = a.output / label
        directory.mkdir()
        channels = range(first, first + count)
        expression = "udp port 36001 and (" + " or ".join(
            f"(udp[20] = {ch & 255} and udp[21] = {ch >> 8})" for ch in channels) + ")"
        procs = []
        try:
            for board in range(4):
                log = (directory / f"dumpcap{board}.log").open("w")
                proc = subprocess.Popen(["dumpcap", "-q", "-i", nics[str(board)],
                    "-f", expression, "-c", str(packets),
                    "-w", str(directory / f"fpga{board}.pcapng")], stdout=log, stderr=log)
                procs.append((proc, log))
            for proc, _ in procs:
                if proc.wait(timeout=50):
                    raise RuntimeError("Packet capture failed")
        finally:
            for proc, log in procs:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                log.close()
        record(operation="capture", label=label, first_channel=first,
               channel_count=count, packets_per_board=packets)

    try:
        adc("off_control", False, 0)
        capture("antennas_off_430", 430, 8, 20000)
        adc("pn_first", True, 0)
        capture("pn_first", 400, 40, 200000)
        adc("pn_reset", False, 0)
        adc("pn_repeat", True, 0)
        capture("pn_repeat", 400, 40, 200000)
    finally:
        # Attempt every board even if one restore fails.
        errors = []
        for board in range(4):
            try:
                result = subprocess.run([CONSOLE, "-k", "lambda", f"10.0.{board}.10"],
                    input="adc\n20\n31\nN\n15\nY\nr\npp\nr freq\nq\nq\n", text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=50)
                (a.output / f"restored_fpga{board}.log").write_text(result.stdout)
                good = result.returncode == 0 and "Last ADC config: PN seq OFF, antennas=0b1111" in result.stdout
                record(operation="restore", board=board, software_readback_matches=good)
                if not good:
                    errors.append(board)
            except Exception as error:
                record(operation="restore_error", board=board, error=str(error))
                errors.append(board)
        if errors:
            raise RuntimeError(f"Restoration needs attention on boards {errors}")
    capture("sky_after_430", 430, 8, 20000)


if __name__ == "__main__":
    main()

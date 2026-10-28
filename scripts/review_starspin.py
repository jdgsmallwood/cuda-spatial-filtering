#!/usr/bin/env python3
"""Run the bundled CODIF fixture through Starspin and its independent reference.

No network capture or telescope access. The output directory must not exist.
"""
import argparse
import json
from pathlib import Path
import subprocess

from check_codif_visibility_parity import check

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/apps/starspin_1_8')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--monitor', action='store_true', help='Enable optional Redis monitoring')
    parser.add_argument('--monitor-port', type=int, default=6380)
    parser.add_argument('--monitor-interval', type=float, default=1)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = {'frequency_plan': {'base_frequency_hz': 0, 'channel_bandwidth_hz': 781250}}
    for name, data in [('config.json', config), ('gains.json', {}),
                       ('eigenvalues.json', {}), ('delays.json', {'0': 0})]:
        (output / name).write_text(json.dumps(data) + '\n')
    pcap = ROOT / 'tests/fixtures/codif_synthetic.pcap'
    visibility = output / 'visibilities.hdf5'
    command = [str(binary), '--packet-format', 'codif', '-p', str(pcap), '--loop',
               '-i', 'lo', '-f', '110', '-c', str(output / 'config.json'),
               '-g', str(output / 'gains.json'), '-e', str(output / 'eigenvalues.json'),
               '--delay-file', str(output / 'delays.json'), '-v', str(visibility),
               '--accumulation-length', '4', '--obs-length', '3']
    if args.monitor:
        command += ['--monitor', '--monitor-port', str(args.monitor_port),
                    '--monitor-interval', str(args.monitor_interval)]
    (output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (output / 'starspin.log').open('w') as log:
        subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=180)
    check(pcap, visibility, output / 'reference')


if __name__ == '__main__':
    main()

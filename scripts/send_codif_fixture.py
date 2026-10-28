#!/usr/bin/env python3
"""Send the bundled CODIF fixture over loopback UDP with advancing frame counters."""
import argparse
from pathlib import Path
import socket
import struct
import time

from check_codif_visibility_parity import read_pcap

FIXTURE = Path(__file__).resolve().parents[1] / 'tests/fixtures/codif_synthetic.pcap'


def load_payloads(path):
    # The numerical reference validates the supported, complete fixture geometry.
    read_pcap(path)
    data = path.read_bytes()
    payloads = []
    offset = 24
    while offset < len(data):
        size = struct.unpack_from('<I', data, offset + 8)[0]
        payloads.append(data[offset + 16 + 42:offset + 16 + size])
        offset += 16 + size
    return payloads


def advance_payload(payload, frames):
    payload = bytearray(payload)
    frame, seconds = struct.unpack_from('<II', payload)
    periods, frame = divmod(frame + frames, 390625)
    struct.pack_into('<II', payload, 0, frame, seconds + periods * 27)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pcap', type=Path, default=FIXTURE)
    parser.add_argument('--port', type=int, default=36001)
    parser.add_argument('--packets', type=int, default=100000)
    parser.add_argument('--pps', type=float, default=5000)
    args = parser.parse_args()
    if args.packets <= 0 or not 0 < args.pps <= 100000 or not 0 < args.port < 65536:
        parser.error('require positive packets, 0 < pps <= 100000 and a valid port')
    payloads = load_payloads(args.pcap)
    frames_per_loop = len(payloads) // 8
    start = time.monotonic()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(('127.0.0.1', 0))  # FPGA ID is the source IP's third octet (zero).
        for i in range(args.packets):
            loop, index = divmod(i, len(payloads))
            sock.sendto(advance_payload(payloads[index], loop * frames_per_loop),
                        ('127.0.0.1', args.port))
            # Pace one fixture-sized burst; Python scheduling is not a line-rate clock.
            if (i + 1) % len(payloads) == 0:
                time.sleep(max(0, start + (i + 1) / args.pps - time.monotonic()))
    elapsed = time.monotonic() - start
    print(f'Sent {args.packets} UDP packets in {elapsed:.3f}s ({args.packets / elapsed:.0f} packets/s)')


if __name__ == '__main__':
    main()

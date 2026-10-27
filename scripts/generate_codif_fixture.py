#!/usr/bin/env python3
"""Generate public test packets from deterministic integers; no observing data."""
from pathlib import Path
import struct


def fixture_bytes():
    result = bytearray(struct.pack('<IHHIIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1))
    for frame in range(4):
        for channel in range(110, 118):
            header = bytearray(64)
            struct.pack_into('<II', header, 0, frame, 0)
            header[9:11] = bytes([8, 4])
            struct.pack_into('<HH', header, 14, 27, channel)
            struct.pack_into('<HHI', header, 24, 20, 5, 320)
            struct.pack_into('<Q', header, 32, 390625)
            header[40:44] = bytes.fromhex('fecaedfe')
            struct.pack_into('<H', header, 44, 0x03bb)
            header[46:56] = bytes([0x77] * 10)
            data = bytearray()
            for t in range(frame * 64, (frame + 1) * 64):
                for adc in range(20):
                    real = (t * (adc % 3 + 1) + channel * 3) % 31 - 15
                    imag = (t * (adc % 3 + 2) + channel + adc % 3 * 7) % 29 - 14
                    data.extend(struct.pack('bb', real, imag))
            udp = struct.pack('!HHHH', 36000, 36001, 8 + len(header) + len(data), 0)
            ip = bytearray(struct.pack('!BBHHHBBH4s4s', 0x45, 0, 20 + len(udp) + len(header) + len(data),
                                       0, 0, 64, 17, 0, bytes([127, 0, 0, 1]), bytes([127, 0, 0, 1])))
            checksum = sum(struct.unpack('!10H', ip))
            while checksum >> 16:
                checksum = (checksum & 0xffff) + (checksum >> 16)
            struct.pack_into('!H', ip, 10, (~checksum) & 0xffff)
            ethernet = bytes.fromhex('0200000000020200000000010800')
            packet = ethernet + ip + udp + header + data
            result.extend(struct.pack('<IIII', 0, 0, len(packet), len(packet)))
            result.extend(packet)
    return bytes(result)


if __name__ == '__main__':
    output = Path(__file__).resolve().parents[1] / 'tests/fixtures/codif_synthetic.pcap'
    output.write_bytes(fixture_bytes())
    print(output)

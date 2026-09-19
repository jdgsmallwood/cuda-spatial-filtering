#!/usr/bin/env python3
"""
Generate a PCAP with exactly one active receiver slot per packet.

The active receiver carries real=1 imag=0 for every time step and polarization;
all other receivers are zero.  Two (or more) FPGA IDs emit packets with identical
sample_count values so the pipeline reassembly stage pairs them correctly.

Default parameters match the LambdaConfig used in tests/test_antenna_ordering.cu:
  NR_CHANNELS=1, NR_FPGA_SOURCES=2, NR_TIME_STEPS_PER_PACKET=8,
  NR_RECEIVERS_PER_PACKET=2, NR_POLARIZATIONS=2, NR_PACKETS_FOR_CORRELATION=1
"""

import argparse
import socket
import struct
import time


def pcap_global_header() -> bytes:
    return struct.pack("IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)


def pcap_packet_header(packet_len: int) -> bytes:
    ts = int(time.time())
    return struct.pack("IIII", ts, 0, packet_len, packet_len)


def ethernet_header(dst_mac: bytes, src_mac: bytes, eth_type: int) -> bytes:
    return dst_mac + src_mac + struct.pack("!H", eth_type)


def ipv4_header(src_ip: str, dst_ip: str, payload_len: int) -> bytes:
    total_length = 20 + 8 + payload_len
    return struct.pack(
        "!BBHHHBBH4s4s",
        0x45, 0, total_length, 0, 0, 18, 17, 0,
        socket.inet_aton(src_ip),
        socket.inet_aton(dst_ip),
    )


def udp_header(src_port: int, dst_port: int, payload_len: int) -> bytes:
    return struct.pack("!HHHH", src_port, dst_port, 8 + payload_len, 0)


def custom_header(sample_count: int, fpga_id: int, freq_channel: int) -> bytes:
    # Matches C struct: uint64 sample_count, uint32 fpga_id, uint16 freq_channel, uint8[8] padding
    return struct.pack("<QIHQ", sample_count, fpga_id, freq_channel, 0)


def make_scale_factors(nr_receivers: int, nr_pol: int) -> bytes:
    # int16_t[nr_receivers][nr_pol], all = 1
    n = nr_receivers * nr_pol
    return struct.pack(f"<{n}H", *([1] * n))


def make_sample_data(nr_time: int, nr_recv: int, nr_pol: int, active_recv: int) -> bytes:
    # complex<int8_t>[nr_time][nr_recv][nr_pol]
    # active receiver: real=1 imag=0; all others: 0+0j
    out = bytearray()
    active_pair = b'\x01\x00'
    zero_pair = b'\x00\x00'
    for _ in range(nr_time):
        for r in range(nr_recv):
            sample = active_pair if r == active_recv else zero_pair
            for _ in range(nr_pol):
                out.extend(sample)
    return bytes(out)


def build_packet(
    fpga_id: int,
    sample_count: int,
    freq_channel: int,
    nr_time: int,
    nr_recv: int,
    nr_pol: int,
    active_recv: int,
    dst_ip: str,
    src_port: int,
    dst_port: int,
) -> bytes:
    # Source IP encodes the FPGA ID in the third octet for OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET mode
    src_ip = f"10.0.{fpga_id}.10"

    udp_payload = (
        custom_header(sample_count, fpga_id, freq_channel)
        + make_scale_factors(nr_recv, nr_pol)
        + make_sample_data(nr_time, nr_recv, nr_pol, active_recv)
    )

    dst_mac = b"\xff\xff\xff\xff\xff\xff"
    src_mac = b"\x00\x0a\x95\x9d\x68\x16"

    return (
        ethernet_header(dst_mac, src_mac, 0x0800)
        + ipv4_header(src_ip, dst_ip, len(udp_payload))
        + udp_header(src_port, dst_port, len(udp_payload))
        + udp_payload
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nr-receivers-per-packet", type=int, default=2,
                        metavar="N", help="receivers per FPGA packet (default 2)")
    parser.add_argument("--nr-polarizations", type=int, default=2,
                        metavar="N", help="polarizations per receiver (default 2)")
    parser.add_argument("--nr-time-steps", type=int, default=8,
                        metavar="N", help="time samples per packet (default 8)")
    parser.add_argument("--nr-channels", type=int, default=1,
                        metavar="N", help="frequency channels (default 1)")
    parser.add_argument("--nr-packets", type=int, default=1,
                        metavar="N", help="number of correlation packets (default 1)")
    parser.add_argument("--fpga-ids", type=int, nargs="+", default=[0, 1],
                        metavar="ID", help="FPGA IDs to emit (default 0 1)")
    parser.add_argument("--active-receiver", type=int, default=0,
                        metavar="R", help="receiver index to fill with 1s (default 0)")
    parser.add_argument("--output", default="antenna_test.pcap",
                        help="output file name (default antenna_test.pcap)")
    args = parser.parse_args()

    dst_ip = "192.168.1.255"
    src_port = 36001
    dst_port = 36001

    frames = []
    for pkt_num in range(args.nr_packets):
        sample_count = pkt_num * args.nr_time_steps
        for ch in range(args.nr_channels):
            for fpga_id in args.fpga_ids:
                frames.append(build_packet(
                    fpga_id=fpga_id,
                    sample_count=sample_count,
                    freq_channel=ch,
                    nr_time=args.nr_time_steps,
                    nr_recv=args.nr_receivers_per_packet,
                    nr_pol=args.nr_polarizations,
                    active_recv=args.active_receiver,
                    dst_ip=dst_ip,
                    src_port=src_port,
                    dst_port=dst_port,
                ))

    with open(args.output, "wb") as f:
        f.write(pcap_global_header())
        for frame in frames:
            f.write(pcap_packet_header(len(frame)))
            f.write(frame)

    scales_bytes = args.nr_receivers_per_packet * args.nr_polarizations * 2
    data_bytes = args.nr_time_steps * args.nr_receivers_per_packet * args.nr_polarizations * 2
    print(
        f"Wrote {len(frames)} packets to '{args.output}'\n"
        f"  FPGAs: {args.fpga_ids}  channels: {args.nr_channels}  "
        f"correlation packets: {args.nr_packets}\n"
        f"  Per-packet payload: 22B header + {scales_bytes}B scales + {data_bytes}B samples\n"
        f"  Active receiver: {args.active_receiver} of {args.nr_receivers_per_packet} "
        f"(1+0j per time/pol, rest zero)"
    )


if __name__ == "__main__":
    main()

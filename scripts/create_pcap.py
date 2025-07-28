import argparse
import socket
import struct
import time
from typing import Union

import numpy as np

g_FILTERBANKS = 5
SCALE_FACTOR_SIZE = 2
SAMPLES_PER_ADC = 64
N_SAMPLES_PER_PACKET = 64
BYTES_PER_SAMPLE = 2

def pcap_global_header() -> bytes:
    return struct.pack(
        "IHHIIII",
        0xA1B2C3D4,  # Magic
        2,
        4,  # Version
        0,
        0,  # Thiszone, sigfigs
        65535,  # Snaplen
        1,  # Linktype: Ethernet
    )


def ethernet_header(dst_mac: bytes, src_mac: bytes, eth_type: int) -> bytes:
    return dst_mac + src_mac + struct.pack("!H", eth_type)


def ipv4_header(src_ip: str, dst_ip: str, payload_len: int) -> bytes:
    """Should be a total of 20 bytes

    :param src_ip: source IP address to pack.
    :type src_ip: str
    :param dst_ip: destination IP address to pack.
    :type dst_ip: str
    :param payload_len: Length of data packet.
    :type payload_len: int
    :return: Packet header in bytes.
    :rtype: bytes
    """
    version_ihl = 0x45
    total_length = 20 + 8 + payload_len
    return struct.pack(
        "!BBHHHBBH4s4s",
        version_ihl,
        0,
        total_length,
        0,
        0,
        18,
        17,
        0,
        socket.inet_aton(src_ip),
        socket.inet_aton(dst_ip),
    )


def udp_header(src_port: int, dst_port: int, payload_len: int) -> bytes:
    """should be a total of 8 bytes

    :param src_port: Source port number to packetize.
    :type src_port: int
    :param dst_port: Destination port number to packetize
    :type dst_port: int
    :param payload_len: Length of data in packet
    :type payload_len: int
    :return: Header in bytes
    :rtype: bytes
    """
    udp_len = 8 + payload_len
    return struct.pack("!HHHH", src_port, dst_port, udp_len, 0)


def custom_header(
    sample_count: int, fpga_id: int, freq_channel: int, magic_number: int
) -> bytes:
    return struct.pack("<QIHQ", sample_count, fpga_id, freq_channel, magic_number)


def scale_factors() -> bytes:
    return b"".join(struct.pack("<H", 1 + i) for i in range(g_FILTERBANKS * 4))


def filterbank_data(
    noise_sigma: float = 0.0,
    tone_freq: Union[float, list[float]] = 1,
    tone_amp: Union[float, list[float]] = 3.0,
    tone_phase: float = 0.0,
    common_noise_sigma: float = 0.0,
    n_receivers: int = 20,
    n_channels: int = 8,
    n_samples: int = SAMPLES_PER_ADC,
    sampling_rate_hz: float = 1000,
) -> np.ndarray:

    if not isinstance(tone_freq, list):
        tone_freq = [tone_freq for _ in range(n_channels)]

    if not isinstance(tone_amp, list):
        tone_amp = [tone_amp for _ in range(n_channels)]

    if len(tone_freq) != n_channels or len(tone_amp) != n_channels:
        raise ValueError(
            f"Tone frequency or amplitude lists are the wrong shape for the data. Len(tone_freq) = {len(tone_freq)}, len(tone_amp) = {len(tone_amp)} whereas n_channels is {n_channels}."
        )

    # Generate common noise if specified (same for all channels per receiver)
    common_noise_real = (
        np.random.normal(0, common_noise_sigma, (n_receivers, n_samples))
        if common_noise_sigma > 0
        else 0
    )
    common_noise_imag = (
        np.random.normal(0, common_noise_sigma, (n_receivers, n_samples))
        if common_noise_sigma > 0
        else 0
    )

    common_noise = common_noise_real + 1j * common_noise_imag

    # Independent Gaussian noise for each channel per receiver.
    noise_real = np.random.normal(0, noise_sigma, (n_channels, n_receivers, n_samples))
    noise_imag = np.random.normal(0, noise_sigma, (n_channels, n_receivers, n_samples))

    noise = noise_real + 1j * noise_imag

    data_np = np.zeros((n_channels, n_receivers, n_samples), dtype=np.complex64)

    t = np.linspace(0, n_samples / sampling_rate_hz - 1 / n_samples, n_samples)
    for channel in range(n_channels):
        # Tone signal
        tone = tone_amp[channel] * np.exp(
            2 * np.pi * 1j * tone_freq[channel] * t + tone_phase
        )

        data_np[channel, :, :] = noise[channel] + common_noise + tone

    return data_np


def packetize_data(data: np.ndarray) -> bytes:
    # This data will be n_receivers, n_samples
    # so tranpose it to be n_samples x n_receivers
    data = data.copy().transpose()

    output_data = bytearray()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            output_data.extend(quantize_data(data[i, j].real, data[i, j].imag))
    return output_data


def quantize_data(total_real: float, total_imag: float) -> bytes:
    real_quantized = np.clip(np.round(total_real), -128, 127).astype(int)
    imag_quantized = np.clip(np.round(total_imag), -128, 127).astype(int)

    packed = struct.pack("bb", real_quantized, imag_quantized)
    return packed


def pcap_packet_header(packet_len: int) -> bytes:
    ts = int(time.time())
    ts_usec = int((time.time() - ts) * 1_000_000)
    return struct.pack("IIII", ts, ts_usec, packet_len, packet_len)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate custom PCAP packet stream.")
    parser.add_argument(
        "--sigma_common", type=float, default=2.0, help="common noise sigma"
    )
    parser.add_argument(
        "--sigma_independent", type=float, default=1.0, help="independent noise sigma"
    )
    parser.add_argument(
        "--tone_freq", type=float, default=440.0, help="tone frequency in Hz"
    )
    parser.add_argument("--tone_amp", type=float, default=1.0, help="tone amplitude")
    parser.add_argument(
        "--tone_phase", type=float, default=0.0, help="tone phase in radians"
    )
    parser.add_argument(
        "--number_packets", type=int, default=100, help="number of packets to generate"
    )
    parser.add_argument(
        "--output", default="output.pcap", type=str, help="name of output file"
    )
    parser.add_argument(
        "--number_receivers", type=int, default=20, help="number of receivers"
    )
    parser.add_argument(
        "--number_channels", type=int, default=8, help="number of channels to generate"
    )

    args = parser.parse_args()
    # Define Constants
    dst_mac = b"\xff\xff\xff\xff\xff\xff"  # Example MAC address.
    src_mac = b"\x00\x0a\x95\x9d\x68\x16"  # Example MAC address.
    eth_type = 0x0800  # IPv4
    src_ip = "192.168.1.100"
    dst_ip = "192.168.1.255"
    src_port = 36001
    dst_port = 36001

    # Get arguments.
    independent_noise_sigma = args.sigma_independent
    common_noise_sigma = args.sigma_common
    tone_freq = args.tone_freq
    tone_amp = args.tone_amp
    tone_phase = args.tone_phase
    number_packets = args.number_packets
    output_file = args.output
    N_CHANNELS = args.number_channels
    N_RECEIVERS = args.number_receivers

    total_data = filterbank_data(
        noise_sigma=independent_noise_sigma,
        tone_freq=tone_freq,
        tone_amp=tone_amp,
        tone_phase=tone_phase,
        common_noise_sigma=common_noise_sigma,
        n_receivers=N_RECEIVERS,
        n_channels=N_CHANNELS,
        n_samples=number_packets * N_SAMPLES_PER_PACKET,
    )

    packets = []

    for packet_num in range(number_packets):
        for channel in range(N_CHANNELS):

            data_to_packetize = total_data[
                channel,
                :,
                packet_num
                * N_SAMPLES_PER_PACKET : (packet_num + 1)
                * N_SAMPLES_PER_PACKET,
            ]

            data_packet = packetize_data(data_to_packetize)

            # Build packet
            data_section = scale_factors() + data_packet
            custom = custom_header(
                sample_count=packet_num * N_SAMPLES_PER_PACKET,
                fpga_id=0xABCD1234,
                freq_channel=channel,
                magic_number=0xDEADBEEFCAFECAFE,
            )
            udp_payload = custom + data_section

            packet = (
                ethernet_header(dst_mac, src_mac, eth_type)
                + ipv4_header(src_ip, dst_ip, len(udp_payload))
                + udp_header(src_port, dst_port, len(udp_payload))
                + udp_payload
            )
            packets.append(packet)

    # === Write PCAP ===
    with open(output_file, "wb") as f:
        f.write(pcap_global_header())
        for packet in packets:
            #assert len(packet) == 2664
            f.write(pcap_packet_header(len(packet)))
            f.write(packet)

    print(f"✅ PCAP file '{output_file}' written.")

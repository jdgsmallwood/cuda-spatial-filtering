from pathlib import Path
import socket
import struct
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from send_codif_fixture import FIXTURE, advance_payload, load_payloads


def test_udp_payload_and_counter_rollover():
    payloads = load_payloads(FIXTURE)
    assert len(payloads) == 32
    assert all(len(p) == 2624 for p in payloads)
    original = bytearray(payloads[0])
    struct.pack_into('<II', original, 0, 390624, 100)
    advanced = advance_payload(original, 2)
    assert struct.unpack_from('<II', advanced) == (1, 127)
    assert advanced[8:] == original[8:]
    assert struct.unpack_from('<II', original) == (390624, 100)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(('127.0.0.1', 0))
        receiver.settimeout(1)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(advanced, receiver.getsockname())
        received, _ = receiver.recvfrom(4096)
    assert received == advanced

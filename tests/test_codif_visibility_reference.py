"""Checks for the independent Python CODIF wire decoder."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_codif_visibility_parity import read_pcap


class CodifVisibilityReferenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ROOT / "tests/fixtures/codif_lambda.pcap"

    def decode(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packet.pcap"
            path.write_bytes(data)
            return read_pcap(path)

    def test_supplied_capture_geometry_and_epoch(self):
        samples, metadata = read_pcap(self.fixture)
        self.assertEqual(samples.shape, (8, 256, 20))
        self.assertEqual(metadata["packets"], 32)
        self.assertEqual(metadata["first_sample"], 782244644444444)
        self.assertEqual(metadata["first_channel"], 110)
        self.assertEqual(metadata["fpga_id"], 0)
        self.assertEqual(metadata["scale_exponents"], [7])

    def test_signed_iq_nibble_order_and_scale_clamping(self):
        data = bytearray(self.fixture.read_bytes())
        frame = 24 + 16
        data[frame + 42 + 46] = 0x91
        struct.pack_into("bbbb", data, frame + 106, 3, -4, 5, -6)
        samples, _ = self.decode(data)
        np.testing.assert_array_equal(samples[0, 0, :2], [6 - 8j, 1280 - 1536j])

    def test_standard_metadata_preserves_decoded_data_and_ip_routing(self):
        expected, expected_metadata = read_pcap(self.fixture)
        data = bytearray(self.fixture.read_bytes())
        for offset in range(24, len(data), 16 + 2666):
            header = offset + 16 + 42
            struct.pack_into("<HH2s", data, header + 18, 2, 0, b"At")
            struct.pack_into("<H", data, header + 44, 0x03bb)
        samples, metadata = self.decode(data)
        np.testing.assert_array_equal(samples, expected)
        self.assertEqual(metadata, expected_metadata)

    def test_unknown_metadata_and_out_of_range_thread_are_rejected(self):
        for offset, value in ((44, 1), (44, 0xbb03), (16, 512), (16, 65535)):
            with self.subTest(offset=offset, value=value):
                data = bytearray(self.fixture.read_bytes())
                struct.pack_into("<H", data, 24 + 16 + 42 + offset, value)
                with self.assertRaises(ValueError):
                    self.decode(data)

    def test_incomplete_or_invalid_capture_is_rejected(self):
        original = self.fixture.read_bytes()
        bad_sync = bytearray(original)
        bad_sync[24 + 16 + 42 + 40] ^= 1
        for data in (original[:-1], original[:-2682], bytes(bad_sync)):
            with self.subTest(size=len(data)):
                with self.assertRaises(ValueError):
                    self.decode(data)


if __name__ == "__main__":
    unittest.main()

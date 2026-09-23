"""Pure exporter-axis tests; uses synthetic gains, never firmware coefficients."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from coarse_pfb_deripple import (load_deripple, validate_calibration_capture,
                                 validate_calibration_manifest)


class CoarsePfbDerippleTests(unittest.TestCase):
    def test_capture_audit_requires_exact_table(self):
        deripple = {"response_id": "synthetic", "voltage_gains": [1.0, 1.02]}
        manifest = {"input_files": {"deripple": {
            "present": True, "content": json.dumps(deripple)}}}
        validate_calibration_manifest(manifest, deripple)
        with self.assertRaises(ValueError):
            validate_calibration_manifest(manifest, {**deripple, "response_id": "other"})
        manifest["input_files"]["deripple"]["present"] = False
        with self.assertRaises(ValueError):
            validate_calibration_manifest(manifest, deripple)

    def test_capture_hdf5_loader_when_h5py_available(self):
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py unavailable in this Python environment")
        deripple = {"response_id": "synthetic", "voltage_gains": [1.0, 1.02]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "capture.h5"
            manifest = {"input_files": {"deripple": {
                "present": True, "content": json.dumps(deripple)}}}
            with h5py.File(path, "w") as file:
                file.create_dataset("audit/run_manifest_json", data=json.dumps(manifest))
            validate_calibration_capture(path, deripple)
            with self.assertRaises(ValueError):
                validate_calibration_capture(path, {**deripple, "response_id": "other"})
            manifest["input_files"]["deripple"]["present"] = False
            with h5py.File(path, "w") as file:
                file.create_dataset("audit/run_manifest_json", data=json.dumps(manifest))
            with self.assertRaises(ValueError):
                validate_calibration_capture(path, deripple)

    def test_rejects_bad_runtime_file(self):
        document = {
            "schema_version": 1,
            "kind": "coarse_pfb_amplitude_deripple",
            "fine_channels": 6,
            "edge_trim": 1,
            "normalization": "coarse_bin_centre",
            "response_id": "synthetic",
            "voltage_gains": [1.0, 1.02, 1.05, 1.1],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps(document))
            self.assertEqual(load_deripple(path, 6, 1), document)
            for replacement in ([1.0], [1.0, 1.02, 1.05, float("nan")],
                                [1.0, 1.02, 1.05, 1.2]):
                document["voltage_gains"] = replacement
                path.write_text(json.dumps(document))
                with self.assertRaises(ValueError):
                    load_deripple(path, 6, 1)


if __name__ == "__main__":
    unittest.main()

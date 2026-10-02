"""Runtime coarse-PFB amplitude table helpers; no production taps live here."""

import json
import math
import re
from pathlib import Path


def load_deripple(path: Path, fine_channels: int, edge_trim: int) -> dict:
    document = json.loads(path.read_text())
    required = {
        "schema_version": 1,
        "kind": "coarse_pfb_amplitude_deripple",
        "fine_channels": fine_channels,
        "edge_trim": edge_trim,
        "normalization": "coarse_bin_centre",
    }
    for key, expected in required.items():
        if document.get(key) != expected:
            raise ValueError(f"de-ripple {key} must be {expected!r}")
    response_id = document.get("response_id")
    if not isinstance(response_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", response_id):
        raise ValueError("de-ripple response_id must be 1-64 safe ASCII characters")
    gains = document.get("voltage_gains")
    if not isinstance(gains, list) or len(gains) != fine_channels - 2 * edge_trim:
        raise ValueError("de-ripple voltage_gains length mismatch")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
           not math.isfinite(x) or not 0 < x <= 1.1 for x in gains):
        raise ValueError("de-ripple voltage_gains must be finite and in (0, 1.1]")
    return document


def validate_calibration_manifest(manifest: dict, deripple: dict) -> None:
    """Require a Starspin audit embedding the exact correction table used."""
    captured_config = manifest["input_files"]["deripple"]
    if not captured_config.get("present"):
        raise ValueError("calibration visibilities were not captured with de-ripple")
    if json.loads(captured_config["content"]) != deripple:
        raise ValueError("calibration visibility de-ripple table differs")


def validate_calibration_capture(path: Path, deripple: dict) -> None:
    import h5py

    with h5py.File(path, "r") as visibility_file:
        raw = visibility_file["audit/run_manifest_json"][()]
    manifest = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    validate_calibration_manifest(manifest, deripple)

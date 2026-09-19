#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from adaptive_detection_reference import (
    apply_projection_to_weights,
    beamform,
    build_null_projection,
    detect_k,
)

try:
    import h5py  # type: ignore
except ModuleNotFoundError:
    h5py = None


def uint16_complex_half_to_complex64(raw: np.ndarray) -> np.ndarray:
    viewed = np.asarray(raw, dtype=np.uint16).view(np.float16).astype(np.float32)
    return viewed[..., 0] + 1j * viewed[..., 1]


def last_dim_pair_to_complex(arr: np.ndarray, dtype=np.float64) -> np.ndarray:
    data = np.asarray(arr)
    return data[..., 0].astype(dtype) + 1j * data[..., 1].astype(dtype)


def run_fixture_generator(test_bin: Path, output_path: Path) -> None:
    env = os.environ.copy()
    env["ADAPTIVE_PARITY_OUTPUT"] = str(output_path)
    subprocess.run(
        [
            str(test_bin),
            "--gtest_filter=AdaptivePipelineTest.DetectionModeWritesParityFixtureToHDF5",
        ],
        check=True,
        env=env,
    )


def h5dump_header(path: Path, dataset: str) -> str:
    return subprocess.check_output(
        ["h5dump", "-H", "-d", f"/{dataset}", str(path)], text=True
    )


def h5dump_attr(path: Path, attr: str) -> str:
    return subprocess.check_output(["h5dump", "-a", attr, str(path)], text=True)


def parse_h5dump_dims(header: str) -> tuple[int, ...]:
    marker = "DATASPACE  SIMPLE {"
    start = header.index(marker) + len(marker)
    rest = header[start:]
    open_paren = rest.index("(")
    close_paren = rest.index(")")
    dims_text = rest[open_paren + 1 : close_paren]
    return tuple(int(part.strip()) for part in dims_text.split(",") if part.strip())


def parse_h5dump_attr_scalar(text: str) -> float:
    for line in text.splitlines():
        line = line.strip().rstrip(",;")
        if ":" in line:
            line = line.split(":", 1)[1].strip()
        try:
            return float(line)
        except ValueError:
            continue
    raise ValueError(f"could not parse scalar attribute from:\n{text}")


def load_dataset_with_h5dump(path: Path, dataset: str, dtype: np.dtype) -> np.ndarray:
    dims = parse_h5dump_dims(h5dump_header(path, dataset))
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            [
                "h5dump",
                "-d",
                f"/{dataset}",
                "-y",
                "-b",
                "NATIVE",
                "-o",
                str(tmp_path),
                str(path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        data = np.fromfile(tmp_path, dtype=dtype)
    finally:
        tmp_path.unlink(missing_ok=True)
    return data.reshape(dims)


def load_fixture(path: Path) -> dict[str, np.ndarray | float]:
    if h5py is not None:
        with h5py.File(path, "r") as handle:
            return {
                "delta": float(handle.attrs["adaptive_detection_delta"]),
                "eigenvalues": handle["eigenvalues"][:],
                "eigenvectors": handle["eigenvectors"][:],
                "counts": handle["nulled_eigenmode_counts"][:],
                "samples": handle["input_samples"][:],
                "weights": handle["beam_weights"][:],
                "beam_data": handle["beam_data"][:],
            }

    return {
        "delta": parse_h5dump_attr_scalar(h5dump_attr(path, "adaptive_detection_delta")),
        "eigenvalues": load_dataset_with_h5dump(path, "eigenvalues", np.float32),
        "eigenvectors": load_dataset_with_h5dump(path, "eigenvectors", np.float32),
        "counts": load_dataset_with_h5dump(path, "nulled_eigenmode_counts", np.int32),
        "samples": load_dataset_with_h5dump(path, "input_samples", np.int8),
        "weights": load_dataset_with_h5dump(path, "beam_weights", np.float32),
        "beam_data": load_dataset_with_h5dump(path, "beam_data", np.uint16),
    }


def compare_fixture(
    fixture: Path, projection_tol: float, beam_tol: float, power_tol: float
) -> None:
    fixture_data = load_fixture(fixture)
    delta = float(fixture_data["delta"])
    eigenvalues = np.asarray(fixture_data["eigenvalues"])
    eigenvectors = last_dim_pair_to_complex(fixture_data["eigenvectors"]).swapaxes(-1, -2)
    counts = np.asarray(fixture_data["counts"])
    samples = last_dim_pair_to_complex(fixture_data["samples"], dtype=np.float64)
    weights = last_dim_pair_to_complex(fixture_data["weights"], dtype=np.float64)
    beam_data = uint16_complex_half_to_complex64(fixture_data["beam_data"])

    failures: list[str] = []
    for channel in range(eigenvalues.shape[0]):
        for pol in range(eigenvalues.shape[1]):
            eigs = eigenvalues[channel, pol]
            vecs = eigenvectors[channel, pol]
            detected = detect_k(eigs, delta)
            stored = int(counts[channel, pol])
            if detected != stored:
                failures.append(
                    f"K mismatch at channel={channel} pol={pol}: python={detected} stored={stored}"
                )
                continue

            projection = build_null_projection(vecs, eigs, detected)
            hermitian_residual = np.max(np.abs(projection - projection.conj().T))
            if hermitian_residual > projection_tol:
                failures.append(
                    f"projection non-Hermitian at channel={channel} pol={pol}: {hermitian_residual}"
                )

            original_weights = weights[channel, pol]
            adapted_weights = apply_projection_to_weights(original_weights, projection)
            complex_samples = samples[channel, pol]
            python_original = beamform(complex_samples, original_weights)[0]
            python_mitigated = beamform(complex_samples, adapted_weights)[0]

            pipeline_original = beam_data[channel, pol, 0]
            pipeline_mitigated = beam_data[channel, pol, 1]

            if not np.allclose(python_original, pipeline_original, atol=beam_tol, rtol=0.0):
                diff = np.max(np.abs(python_original - pipeline_original))
                failures.append(
                    f"original beam mismatch at channel={channel} pol={pol}: max_abs={diff}"
                )
            if not np.allclose(
                python_mitigated, pipeline_mitigated, atol=beam_tol, rtol=0.0
            ):
                diff = np.max(np.abs(python_mitigated - pipeline_mitigated))
                failures.append(
                    f"mitigated beam mismatch at channel={channel} pol={pol}: max_abs={diff}"
                )

            python_power = float(np.sum(np.abs(python_mitigated) ** 2))
            pipeline_power = float(np.sum(np.abs(pipeline_mitigated) ** 2))
            if abs(python_power - pipeline_power) > power_tol:
                failures.append(
                    f"power mismatch at channel={channel} pol={pol}: python={python_power} pipeline={pipeline_power}"
                )

    if failures:
        raise AssertionError("\n".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive-test-bin", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--projection-tol", type=float, default=1e-5)
    parser.add_argument("--beam-tol", type=float, default=2e-1)
    parser.add_argument("--power-tol", type=float, default=2.0)
    args = parser.parse_args()

    fixture = args.fixture
    if fixture is None:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            fixture = Path(tmp.name)
        try:
            run_fixture_generator(args.adaptive_test_bin, fixture)
            compare_fixture(
                fixture, args.projection_tol, args.beam_tol, args.power_tol
            )
        finally:
            fixture.unlink(missing_ok=True)
    else:
        run_fixture_generator(args.adaptive_test_bin, fixture)
        compare_fixture(fixture, args.projection_tol, args.beam_tol, args.power_tol)

    print("adaptive detection parity passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


def percentile_linear(sorted_values: np.ndarray, quantile: float) -> float:
    values = np.asarray(sorted_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("expected a 1D eigenvalue array")
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return float(values[0])
    position = float(np.clip(quantile, 0.0, 1.0)) * (values.size - 1)
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    if lower == upper:
        return float(values[lower])
    frac = position - lower
    return float(values[lower] + frac * (values[upper] - values[lower]))


def detect_k(eigenvalues: np.ndarray, delta: float) -> int:
    eigs = np.asarray(eigenvalues, dtype=np.float64)
    p20 = percentile_linear(eigs, 0.2)
    p50 = percentile_linear(eigs, 0.5)
    p80 = percentile_linear(eigs, 0.8)
    sigma_noise = (p80 - p20) / (2.0 * 0.8416)
    threshold = p50 + delta * sigma_noise
    return int(np.count_nonzero(eigs > threshold))


def build_null_projection(
    evecs: np.ndarray, evals: np.ndarray, detected_k: int
) -> np.ndarray:
    values = np.asarray(evals)
    vecs = np.asarray(evecs, dtype=np.complex128)
    n = values.shape[0]
    if detected_k == 0:
        return np.eye(n, dtype=np.complex128)
    rfi = vecs[:, n - detected_k :]
    return np.eye(n, dtype=np.complex128) - rfi @ rfi.conj().T


def build_shrink_projection(
    evecs: np.ndarray, evals: np.ndarray, detected_k: int
) -> np.ndarray:
    values = np.asarray(evals, dtype=np.float64)
    vecs = np.asarray(evecs, dtype=np.complex128)
    n = values.shape[0]
    scales = np.ones(n, dtype=np.float64)
    if detected_k > 0:
        keep = n - detected_k
        lambda_bar = float(np.mean(values[:keep])) if keep > 0 else 0.0
        for idx in range(keep, n):
            scales[idx] = lambda_bar / values[idx] if values[idx] > 0.0 else 0.0
    scaled = vecs * scales[np.newaxis, :]
    return scaled @ vecs.conj().T


def apply_projection_to_weights(
    weights: np.ndarray, projection: np.ndarray
) -> np.ndarray:
    return np.asarray(weights, dtype=np.complex128) @ np.asarray(
        projection, dtype=np.complex128
    )


def beamform(samples: np.ndarray, adapted_weights: np.ndarray) -> np.ndarray:
    return np.asarray(adapted_weights, dtype=np.complex128) @ np.asarray(
        samples, dtype=np.complex128
    )


@dataclass
class SelfTestResult:
    name: str
    passed: bool


def run_self_tests() -> list[SelfTestResult]:
    tests: list[SelfTestResult] = []

    tests.append(
        SelfTestResult(
            "strict-threshold-equality",
            detect_k(np.array([0.0, 0.0, 100.0, 100.0]), 0.8416) == 0,
        )
    )
    tests.append(
        SelfTestResult(
            "zero-spectrum",
            detect_k(np.zeros(4), 0.5) == 0,
        )
    )
    tests.append(
        SelfTestResult(
            "rank1-detection",
            detect_k(np.array([0.0, 0.0, 0.0, 100.0]), 0.5) == 1,
        )
    )
    tests.append(
        SelfTestResult(
            "rank2-detection",
            detect_k(np.array([0.0, 0.0, 100.0, 100.0]), 0.5) == 2,
        )
    )

    identity = np.eye(4, dtype=np.complex128)
    null_projection = build_null_projection(identity, np.array([0, 0, 5, 10]), 1)
    tests.append(
        SelfTestResult(
            "null-projection-last-mode",
            np.allclose(null_projection, np.diag([1.0, 1.0, 1.0, 0.0])),
        )
    )

    shrink_projection = build_shrink_projection(
        identity, np.array([1.0, 2.0, 3.0, 12.0]), 1
    )
    tests.append(
        SelfTestResult(
            "shrink-scaling",
            np.allclose(shrink_projection, np.diag([1.0, 1.0, 1.0, 1.0 / 6.0])),
        )
    )
    return tests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if not args.self_test:
        parser.error("use --self-test or import this module")

    failed = [test.name for test in run_self_tests() if not test.passed]
    if failed:
        for name in failed:
            print(f"FAILED {name}")
        return 1
    print("adaptive_detection_reference self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

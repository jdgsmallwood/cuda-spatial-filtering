import unittest
import sys

import numpy as np


def null_projection_from_correlation(correlation, k):
    eigvals, eigvecs = np.linalg.eigh(correlation)
    if k == 0:
        return np.eye(correlation.shape[0], dtype=np.complex64), eigvals

    u = eigvecs[:, -k:]
    projection = np.eye(correlation.shape[0], dtype=np.complex64) - u @ u.conj().T
    return projection.astype(np.complex64), eigvals


def strong_noise_sample(time, receiver, polarization, channel=0):
    strong_re = [
        [18, -17, 15, -14, 12, -11, 9, -8],
        [-13, 16, -15, 11, -10, 14, -9, 12],
    ][channel % 2][time % 8]
    strong_im = [
        [7, 13, -9, -15, 11, 5, -13, -7],
        [-12, 6, 14, -8, 10, -16, 4, 9],
    ][channel % 2][(time * 3 + channel) % 8]
    noise_re = ((time * 17 + receiver * 13 + polarization * 7 + channel * 5) % 7) - 3
    noise_im = ((time * 11 + receiver * 5 + polarization * 3 + channel * 2) % 5) - 2
    return np.complex64(complex(strong_re + noise_re, strong_im + noise_im))


def emit_pipeline_strong_noise_reference():
    nr_receivers = 4
    nr_times = 64
    nr_channels = 2
    weights = np.ones(nr_receivers, dtype=np.complex64)
    payload = {"channels": []}

    for channel in range(nr_channels):
        channel_payload = {"polarizations": []}
        for polarization in range(2):
            samples = np.array(
                [
                    [
                        strong_noise_sample(time, receiver, polarization, channel)
                        for receiver in range(nr_receivers)
                    ]
                    for time in range(nr_times)
                ],
                dtype=np.complex64,
            )
            correlation = samples.T @ samples.conj()
            projection, eigvals = null_projection_from_correlation(correlation, k=1)

            original = samples @ weights
            nulled = samples @ (weights @ projection)

            def pairs(values):
                return [[float(v.real), float(v.imag)] for v in values]

            channel_payload["polarizations"].append(
                {
                    "original": pairs(original),
                    "nulled": pairs(nulled),
                    "eigenvalues": [float(v) for v in eigvals],
                }
            )
        payload["channels"].append(channel_payload)

    import json

    print(json.dumps(payload))


class AdaptiveNullingReferenceTest(unittest.TestCase):
    def test_projected_weights_match_projected_samples_for_strong_signal(self):
        nr_receivers = 4
        nr_times = 64

        signal = np.array([1.0 + 0.0j, 0.5 + 0.25j, -0.25 + 0.75j, -0.5 - 0.5j])
        signal = signal / np.linalg.norm(signal)
        amplitudes = (1000.0 + 0.5j) * np.ones(nr_times, dtype=np.complex64)
        samples = amplitudes[:, None] * signal[None, :]

        correlation = samples.T @ samples.conj()
        projection, eigvals = null_projection_from_correlation(correlation, k=1)

        # The dominant direction should be overwhelmingly larger than the rest.
        self.assertGreater(eigvals[-1], 1e6 * max(float(eigvals[-2]), 1e-20))

        weights = signal.conj() / nr_receivers
        projected_weights = weights @ projection

        beam_from_projected_weights = samples @ projected_weights
        beam_from_projected_samples = (samples @ projection.T) @ weights

        np.testing.assert_allclose(
            beam_from_projected_weights,
            beam_from_projected_samples,
            rtol=1e-5,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            beam_from_projected_weights,
            np.zeros_like(beam_from_projected_weights),
            rtol=0.0,
            atol=1e-4,
        )

    def test_k0_projection_leaves_beam_unchanged(self):
        nr_receivers = 4
        rng = np.random.default_rng(12345)
        samples = (
            rng.normal(size=(64, nr_receivers))
            + 1j * rng.normal(size=(64, nr_receivers))
        ).astype(np.complex64)
        weights = (
            rng.normal(size=nr_receivers) + 1j * rng.normal(size=nr_receivers)
        ).astype(np.complex64)

        correlation = samples.T @ samples.conj()
        projection, _ = null_projection_from_correlation(correlation, k=0)

        np.testing.assert_allclose(samples @ (weights @ projection), samples @ weights)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--emit-pipeline-strong-noise-reference":
        emit_pipeline_strong_noise_reference()
    else:
        unittest.main()

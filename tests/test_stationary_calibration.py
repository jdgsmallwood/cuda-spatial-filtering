"""Independent sky/stationary-signal invariants for offline calibration."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from test_stationary_projection import (fixed_mode, fit_motion, project,
                                        subtract_covariance_component)
from fit_sun_sky_stationary import fit_with_stationary


class StationaryCalibrationTests(unittest.TestCase):
    def test_white_noise_and_injected_sky_response(self):
        rng = np.random.default_rng(641)
        n = 12
        u = np.exp(1j * rng.uniform(-np.pi, np.pi, n)) / np.sqrt(n)
        sky = np.exp(1j * rng.uniform(-np.pi, np.pi, n))
        mode = np.outer(u, u.conj())
        covariance = np.outer(sky, sky.conj())
        mask = np.triu(np.ones((n, n), bool), 1)
        noise = np.eye(n)
        # White noise must not acquire a spurious negative cross-correlation.
        clean, amplitude = subtract_covariance_component(noise, u, mask)
        np.testing.assert_allclose(clean, noise, atol=1e-13)
        self.assertAlmostEqual(float(amplitude), 0.)
        # Freeze the learned direction, then vary interference in held data.
        learned, _ = fixed_mode(np.array([noise + 90 * mode, noise + 110 * mode]))
        dirty = noise + 160 * mode
        base, _ = subtract_covariance_component(dirty, learned, mask)
        injected, _ = subtract_covariance_component(dirty + .1 * covariance, learned, mask)
        response, _ = subtract_covariance_component(.1 * covariance, learned, mask)
        np.testing.assert_allclose(base, noise, atol=1e-12)
        np.testing.assert_allclose(injected - base, response, atol=1e-12)
        # A sky source in the interference direction is lost completely.
        lost, _ = subtract_covariance_component(mode, learned, mask)
        np.testing.assert_allclose(lost, 0, atol=1e-12)
        # The voltage projector must transform a forward sky model identically.
        p = np.eye(n) - mode
        np.testing.assert_allclose(project(covariance, u),
            np.outer(p @ sky, (p @ sky).conj()), atol=1e-12)

    def test_motion_fit_never_uses_held_values(self):
        phase = np.exp(1j * np.linspace(0, 4, 100))[:, None, None]
        data = (3 + 2j) + (7 - 1j) * phase
        train = np.arange(100) < 50
        pred, coeff, _ = fit_motion(data, phase, train)
        np.testing.assert_allclose(pred, data, atol=1e-12)
        altered = data.copy()
        altered[~train] *= 30j
        pred2, coeff2, _ = fit_motion(altered, phase, train)
        np.testing.assert_allclose(coeff2, coeff, atol=1e-12)
        np.testing.assert_allclose(pred2, pred, atol=1e-12)

    def test_stable_antenna_gains_with_large_additive_signal(self):
        rng = np.random.default_rng(187)
        n, nt = 10, 90
        t = np.linspace(0, 2, nt)
        steering = np.exp(1j * t[:, None] * np.arange(n)[None])
        model = steering[:, :, None] * steering[:, None].conj()
        gains = (1 + rng.random(n)) * np.exp(1j * rng.uniform(-np.pi, np.pi, n))
        pickup = 10 * np.exp(1j * rng.uniform(-np.pi, np.pi, n))
        stationary = np.outer(pickup, pickup.conj())
        data = model * gains[None, :, None] * gains[None, None].conj() + stationary
        train = np.arange(nt) < 45
        weights = 1 - np.eye(n)
        _, recovered, prediction = fit_with_stationary(data, model, train, weights)
        np.testing.assert_allclose(prediction[~train], data[~train], atol=1e-5)
        np.testing.assert_allclose(recovered, stationary, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

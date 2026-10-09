"""Synthetic check of the held-baseline solar amplitude solve."""

import numpy as np

from analysis.fit_sun_amplitude_selfcal import fit_antenna_amplitudes


def test_recovers_antenna_amplitudes_despite_outliers():
    antenna_a, antenna_b = np.triu_indices(12, k=1)
    true = np.exp(np.linspace(-0.25, 0.25, 12))
    measured = 3.7 * true[antenna_a] * true[antenna_b]
    measured *= np.exp(0.005 * np.sin(np.arange(len(measured))))
    measured[[2, 14, 36]] *= 2.5
    train = np.arange(len(measured))[np.arange(len(measured)) % 5 != 0]
    fitted, active = fit_antenna_amplitudes(
        measured, antenna_a, antenna_b, train, nant=12)
    np.testing.assert_array_equal(active, np.arange(12))
    target = true / np.median(true)
    assert np.median(np.abs(np.log(fitted / target))) < 0.015

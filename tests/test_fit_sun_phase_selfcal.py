"""Synthetic check of antenna phase recovery with withheld baselines."""

import numpy as np

from analysis.fit_sun_phase_selfcal import fit_antenna_phases


def test_recovers_broadband_antenna_phases_with_outlier_baselines():
    a, b = np.triu_indices(12, k=1)
    true = np.linspace(-.18, .22, 12)
    pair = np.exp(1j * (true[a] - true[b]))
    pair *= np.exp(.004j * np.sin(np.arange(len(pair))))
    pair[[2, 14, 36]] *= np.exp(1.2j)
    counts = np.full(len(pair), 100)
    train = np.arange(len(pair))[np.arange(len(pair)) % 5 != 0]
    fitted, active = fit_antenna_phases(pair, counts, a, b, train, nant=12)
    np.testing.assert_array_equal(active, np.arange(12))
    residual = np.angle(np.exp(1j * (fitted - true)))
    residual = np.angle(np.exp(1j * (residual - np.median(residual))))
    assert np.median(np.abs(residual)) < .015

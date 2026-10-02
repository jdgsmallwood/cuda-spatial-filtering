"""FFT-bin centres and 3+2 science-axis selection for the 32/27 PFB."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fine_frequency_axis import (COARSE_WIDTH_HZ, FINE_SPACING_HZ,
                                 processing_frequencies_hz, science_channel_indices,
                                 science_channel_mask)


def test_gpu_tone_bin_and_boundary_copies_have_expected_centres():
    freq = processing_frequencies_hz(360, 40).reshape(40, 28)
    centre = 360 * COARSE_WIDTH_HZ
    # A +8-bin tone in original FFT bin 24 appears at retained bin 22.
    np.testing.assert_allclose(freq[0, 22], centre + 8 * FINE_SPACING_HZ)
    np.testing.assert_allclose(freq[:-1, -1], freq[1:, 0], atol=1e-7)


def test_three_low_two_high_trim_is_unique_uniform_and_centered():
    freq = processing_frequencies_hz(360, 40)
    kept = science_channel_indices(40)
    selected = freq[kept]
    assert len(selected) == 1080
    np.testing.assert_allclose(np.diff(selected), FINE_SPACING_HZ, atol=1e-7)
    np.testing.assert_allclose(selected[0], 360 * COARSE_WIDTH_HZ - 13 * FINE_SPACING_HZ)
    np.testing.assert_allclose(selected[-1], 399 * COARSE_WIDTH_HZ + 13 * FINE_SPACING_HZ)
    assert 1080 * FINE_SPACING_HZ == 40 * COARSE_WIDTH_HZ
    mask = science_channel_mask(freq)
    np.testing.assert_array_equal(np.flatnonzero(mask), kept)

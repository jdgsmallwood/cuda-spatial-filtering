"""Frequency axes for Lambda's complex 32/27 fine channelizer.

Starspin and the science/DADA axis keep original FFT-shifted bins 3..29 from
every coarse channel: 27 distinct centres per coarse channel. Older Starspin
captures and the pulsar path keep bins 2..29 internally.
"""

import numpy as np

COARSE_WIDTH_HZ = 781_250.0
FINE_SPACING_HZ = COARSE_WIDTH_HZ / 27.0
INTERNAL_BINS_PER_COARSE = 28


def processing_frequencies_hz(min_coarse: int, coarse_channels: int) -> np.ndarray:
    """Return the 28-bin internal frequency axis in GPU processing order."""
    coarse = min_coarse + np.arange(coarse_channels, dtype=float)[:, None]
    raw_bin = np.arange(2, 30, dtype=float)[None, :]
    return (coarse * COARSE_WIDTH_HZ +
            (raw_bin - 16.0) * FINE_SPACING_HZ).ravel()


def unique_processing_frequencies_hz(min_coarse: int, coarse_channels: int) -> np.ndarray:
    """Return Starspin's 27 unique raw bins 3..29 per coarse channel."""
    return processing_frequencies_hz(min_coarse, coarse_channels)[
        science_channel_indices(coarse_channels)]


def science_channel_indices(coarse_channels: int) -> np.ndarray:
    """Indices of original GPU bins 3..29 in the 28-bin internal axis."""
    return (np.arange(coarse_channels)[:, None] * INTERNAL_BINS_PER_COARSE +
            np.arange(1, INTERNAL_BINS_PER_COARSE)[None, :]).ravel()


def science_channel_mask(frequencies_hz: np.ndarray) -> np.ndarray:
    """Select 3..29 on a 28-bin internal axis; accept small generic fixtures."""
    frequency = np.asarray(frequencies_hz)
    if len(frequency) % INTERNAL_BINS_PER_COARSE:
        return np.r_[True, np.diff(frequency) > 1]
    result = np.zeros(len(frequency), bool)
    result[science_channel_indices(len(frequency) // INTERNAL_BINS_PER_COARSE)] = True
    return result


def science_frequency_bounds_hz(min_coarse: int, coarse_channels: int) -> tuple[float, float]:
    frequency = processing_frequencies_hz(min_coarse, coarse_channels)
    selected = science_channel_indices(coarse_channels)
    return float(frequency[selected[0]]), float(frequency[selected[-1]])

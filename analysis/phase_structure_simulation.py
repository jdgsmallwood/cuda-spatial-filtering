"""Small phase model for the interactive coarse/fine-channel notebook.

This is an illustrative relative-voltage model, not an FPGA emulator. All
phases are relative to the first coarse-channel centre; a constant phase has
no effect on the slope comparisons.
"""

import numpy as np

COARSE_HZ = 781_250.0
FINE_HZ = COARSE_HZ / 27.0
PN_TOOTH_HZ = COARSE_HZ / 4.0
PN_OFFSETS_HZ = -329_589.84375 + PN_TOOTH_HZ * np.arange(4)


def phase_terms(channels, offsets_hz, input_delay_ns=0.0,
                coarse_term_ns=0.0, post_channel_delay_ns=0.0):
    """Return input, coarse-only, post-channel and summed phase in radians.

    Shape is (number of coarse channels, number of offsets). Positive phase
    slopes follow the convention used for the PN fits; conjugation reverses
    every displayed phase without changing the comparison.
    """
    channels = np.asarray(channels, dtype=int)
    offsets = np.asarray(offsets_hz, dtype=float)
    if channels.ndim != 1 or offsets.ndim != 1 or not len(channels) or not len(offsets):
        raise ValueError("channels and offsets_hz must be nonempty one-dimensional arrays")
    centres_from_ref = (channels - channels[0])[:, None] * COARSE_HZ
    local = offsets[None, :]
    input_phase = 2 * np.pi * (centres_from_ref + local) * input_delay_ns * 1e-9
    coarse_phase = 2 * np.pi * centres_from_ref * coarse_term_ns * 1e-9
    post_phase = 2 * np.pi * local * post_channel_delay_ns * 1e-9
    total = input_phase + coarse_phase + post_phase
    return {
        "input": np.broadcast_to(input_phase, total.shape),
        "coarse": np.broadcast_to(coarse_phase, total.shape),
        "post": np.broadcast_to(post_phase, total.shape),
        "total": total,
    }


def pn_tooth_phase(channels, input_delay_ns=0.0, coarse_term_ns=0.0,
                   post_channel_delay_ns=0.0, intrinsic_ripple_rad=0.0):
    """Illustrative PN comb phases on four measured teeth per coarse channel.

    The nine-tooth sinusoid is deliberately arbitrary: its purpose is to show
    that a test generator's Fourier coefficients need not have linear phase.
    It is not a model of the actual ADC-box PN generator.
    """
    channels = np.asarray(channels, dtype=int)
    components = phase_terms(channels, PN_OFFSETS_HZ, input_delay_ns,
                             coarse_term_ns, post_channel_delay_ns)
    tooth_number = 4 * (channels - channels[0])[:, None] + np.arange(4)[None, :]
    intrinsic = intrinsic_ripple_rad * np.sin(2 * np.pi * tooth_number / 9.0 + 0.4)
    return components["total"] + intrinsic, intrinsic


def apparent_delay_ns(frequency_hz, phase_rad):
    """Slope of wrapped phase after sequential unwrapping, expressed in ns.

    The result is an apparent delay: frequency spacing can alias true delays.
    """
    frequency = np.asarray(frequency_hz, dtype=float)
    phase = np.asarray(phase_rad, dtype=float)
    if frequency.ndim != 1 or phase.shape != frequency.shape or len(frequency) < 2:
        raise ValueError("frequency_hz and phase_rad must be matching 1D arrays")
    unwrapped = np.unwrap(np.angle(np.exp(1j * phase)))
    centred = frequency - np.mean(frequency)
    slope = np.dot(centred, unwrapped - np.mean(unwrapped)) / np.dot(centred, centred)
    return float(slope * 1e9 / (2 * np.pi))


def fitted_slopes(channels, fine_offsets_hz, input_delay_ns, coarse_term_ns,
                  post_channel_delay_ns, intrinsic_ripple_rad=0.0):
    """Measure centre-to-centre and within-channel apparent phase slopes."""
    channels = np.asarray(channels, dtype=int)
    fine_offsets = np.asarray(fine_offsets_hz, dtype=float)
    sky = phase_terms(channels, fine_offsets, input_delay_ns,
                      coarse_term_ns, post_channel_delay_ns)["total"]
    sky_centres = phase_terms(channels, np.array([0.0]), input_delay_ns,
                              coarse_term_ns, post_channel_delay_ns)["total"][:, 0]
    pn, _ = pn_tooth_phase(channels, input_delay_ns, coarse_term_ns,
                           post_channel_delay_ns, intrinsic_ripple_rad)
    pn_centres = np.angle(np.mean(np.exp(1j * pn), axis=1))
    pn_local = [apparent_delay_ns(PN_OFFSETS_HZ, p) for p in pn]
    sky_local = [apparent_delay_ns(fine_offsets, p) for p in sky]
    return {
        "sky_across_ns": apparent_delay_ns(channels * COARSE_HZ, sky_centres),
        "sky_within_ns": float(np.median(sky_local)),
        "pn_across_ns": apparent_delay_ns(channels * COARSE_HZ, pn_centres),
        "pn_within_ns": float(np.median(pn_local)),
    }

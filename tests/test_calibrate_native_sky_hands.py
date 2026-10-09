"""Check calibration hand selection when either end of a baseline is swapped."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate_native_sky import select_parallel_hand  # noqa: E402


def test_select_parallel_hand_with_two_independent_receptor_swaps():
    # Antennas are zero-based here: A8 and A12 have reversed recorded hands.
    a = np.array([0, 7, 0, 7, 11])
    b = np.array([1, 9, 11, 11, 14])
    swapped = np.zeros(36, bool)
    swapped[[7, 11]] = True
    block = np.empty((3, len(a), 2, 2, 2), dtype=float)
    for fi in range(3):
        for bi in range(len(a)):
            for p in range(2):
                for q in range(2):
                    block[fi, bi, p, q] = [100*fi + 10*bi + 2*p + q,
                                            -100*fi - 10*bi - 2*p - q]
    for hand in (0, 1):
        selected = select_parallel_hand(block, a, b, swapped, hand)
        expected = np.column_stack([
            block[:, bi, hand ^ int(swapped[ai]),
                  hand ^ int(swapped[ant_b]), 0]
            + 1j * block[:, bi, hand ^ int(swapped[ai]),
                         hand ^ int(swapped[ant_b]), 1]
            for bi, (ai, ant_b) in enumerate(zip(a, b))])
        np.testing.assert_array_equal(selected, expected)

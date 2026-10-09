"""Check the conceptual phase notebook against analytically known cases."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from phase_structure_simulation import FINE_HZ, fitted_slopes  # noqa: E402


class PhaseStructureSimulationTests(unittest.TestCase):
    def setUp(self):
        self.channels = np.arange(400, 408)
        self.fine_offsets = (np.arange(3, 30) - 16) * FINE_HZ

    def slopes(self, *terms):
        return fitted_slopes(self.channels, self.fine_offsets, *terms)

    def test_input_delay_has_the_same_coarse_and_fine_slope(self):
        result = self.slopes(350, 0, 0)
        for value in result.values():
            self.assertAlmostEqual(value, 350)

    def test_coarse_and_post_channel_terms_are_distinct(self):
        result = self.slopes(180, 250, 90)
        self.assertAlmostEqual(result["sky_across_ns"], 430)
        self.assertAlmostEqual(result["sky_within_ns"], 270)
        self.assertAlmostEqual(result["pn_across_ns"], 430)
        self.assertAlmostEqual(result["pn_within_ns"], 270)

    def test_pn_spectral_phase_does_not_change_sky(self):
        result = self.slopes(0, 0, 0, 1.2)
        self.assertAlmostEqual(result["sky_across_ns"], 0)
        self.assertAlmostEqual(result["sky_within_ns"], 0)
        self.assertGreater(abs(result["pn_within_ns"]), 50)

    def test_coarse_spacing_aliases_a_large_input_delay(self):
        result = self.slopes(700, 0, 0)
        self.assertAlmostEqual(result["sky_across_ns"], -580)
        self.assertAlmostEqual(result["sky_within_ns"], 700)


if __name__ == "__main__":
    unittest.main()

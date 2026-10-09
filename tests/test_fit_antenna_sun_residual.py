"""Synthetic check of the antenna phase/delay recovery used for Sun residuals."""

import unittest

import numpy as np

from analysis.fit_antenna_sun_residual import FREF, fit_gain, wrap
from analysis.evaluate_board_projected_sun import project_board_phase


class AntennaSunResidualTest(unittest.TestCase):
    def test_board_projection_preserves_internal_phase(self):
        antenna = np.arange(1, 9)
        board = np.repeat(np.arange(4), 2)
        frequency = FREF + np.arange(6).reshape(2, 3) * 1e5
        sky_phase = np.linspace(-.3, .4, len(antenna))
        sky_delay = np.linspace(-20., 20., len(antenna))
        board_shift = np.array([.31, -.27, .59, -1.01])
        pn_sky_phase = sky_phase + board_shift[board]
        pn_sky_delay = sky_delay.copy()
        pn_phase = np.zeros((len(antenna),) + frequency.shape)
        got, coherence = project_board_phase(
            antenna, board, frequency, sky_phase, sky_delay,
            pn_sky_phase, pn_sky_delay, pn_phase)
        expected = wrap(board_shift - board_shift[0])
        np.testing.assert_allclose(got,
                                   np.broadcast_to(expected[:, None, None], got.shape),
                                   atol=1e-12)
        np.testing.assert_allclose(coherence, 1., atol=1e-12)
        for a in range(0, len(antenna), 2):
            np.testing.assert_allclose(got[board[a]] - got[board[a + 1]], 0.)

    def test_recovers_relative_phase_delay_and_pn_sign(self):
        antenna = np.array([1, 2, 3, 4, 5])
        pair = np.array([(a, b) for a in antenna for b in antenna if a < b])
        freq = FREF + np.array([-.20, 0., .20, .58, .78, .98, 1.36, 1.56, 1.76]) * 1e6
        freq_grid = np.broadcast_to(freq[:, None], (len(freq), len(pair)))
        phase = np.array([0., .31, -.42, .59, -.12])
        delay_ns = np.array([0., -65., 43., 91., -22.])
        pn_phase_ant = np.array([0., .37, -.31, .19, -.41])
        pn_delay_ant = np.array([0., 230., -130., 92., -180.])
        observed = np.empty_like(freq_grid)
        pn = np.empty_like(freq_grid)
        for bi, (a, b) in enumerate(pair):
            a -= 1
            b -= 1
            offset = 2 * np.pi * (freq - FREF) * 1e-9
            pn[:, bi] = pn_phase_ant[a] - pn_phase_ant[b] + \
                offset * (pn_delay_ant[a] - pn_delay_ant[b])
            observed[:, bi] = wrap(pn[:, bi] + phase[a] - phase[b] +
                                     offset * (delay_ns[a] - delay_ns[b]))
        quality = np.zeros_like(observed, bool)
        quality[:6] = True
        quality[:, -1] = False  # leave one baseline out entirely
        ids, got_phase, got_delay, prediction, fit = fit_gain(
            pair, freq_grid, observed, pn, quality, np.full_like(observed, 10.), 1)
        self.assertTrue(fit.success)
        np.testing.assert_array_equal(ids, antenna)
        np.testing.assert_allclose(wrap(got_phase - phase), 0., atol=1e-5)
        np.testing.assert_allclose(got_delay, delay_ns, atol=1e-3)
        self.assertLess(np.max(abs(wrap(observed[6:] - prediction[6:]))), 1e-5)


if __name__ == "__main__":
    unittest.main()

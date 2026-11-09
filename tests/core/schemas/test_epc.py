"""Independent analytic checks for units, spectral moments and signed integration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.constants import physical_constants

from quacc.schemas.epc import CM1_PER_MEV, selected_spectrum


class TestEPC(unittest.TestCase):
    def test_einstein_and_large_lambda(self):
        ryK = (
            physical_constants["Rydberg constant times hc in J"][0]
            / physical_constants["Boltzmann constant"][0]
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a2F.dos"
            w0 = 0.004
            width = 0.00001
            lam = 3.0
            w = np.linspace(0.0038, 0.0042, 10001)
            a = (
                lam
                * w0
                / 2
                * np.exp(-0.5 * ((w - w0) / width) ** 2)
                / (width * np.sqrt(2 * np.pi))
            )
            np.savetxt(p, np.column_stack([w, a]))
            r, mev, _, cumulative = selected_spectrum(p)
            assert abs(r["lambda"] - lam) < 3e-5
            assert abs(r["omega_log_K"] - w0 * ryK) < 0.02
            assert abs(cumulative[-1] - r["lambda"]) < 1e-10
            mu = 0.1
            expected = (
                w0
                * ryK
                / 1.2
                * np.exp(-1.04 * (1 + lam) / (lam - mu * (1 + 0.62 * lam)))
            )
            assert abs(r["Tc"][0]["McMillan_omega_log_K"] - expected) < 0.02
            assert r["Tc"][0]["Allen_Dynes_K"] > expected
            assert mev[-1] > mev[0]

    def test_signed_spectrum_is_not_clipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a2F.dos"
            w = np.linspace(0.001, 0.006, 1001)
            a = np.exp(-(((w - 0.003) / 0.0003) ** 2)) - 0.05 * np.exp(
                -(((w - 0.005) / 0.0003) ** 2)
            )
            np.savetxt(p, np.column_stack([w, a]))
            r, _, _, curve = selected_spectrum(p)
            assert "negative_spectral_weight" in r["flags"]
            assert r["lambda"] < r["lambda_positive_part"]
            assert curve[-1] < curve.max()

    def test_frequency_conversion(self):
        assert abs(CM1_PER_MEV - 8.065543937) < 1e-8


if __name__ == "__main__":
    unittest.main()

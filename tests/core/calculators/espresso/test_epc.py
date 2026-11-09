"""Offline EPC regression against preserved QE7.5 MgB2 q4 spectra."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import physical_constants

from quacc.calculators.espresso.epc import (
    analyze_a2f,
    analyze_epc_directory,
    epc_file_manifest,
    read_a2f,
    read_matdyn_frequencies,
    read_phonon_dos,
)

DATA = Path(__file__).parent / "epc_data"


def test_mgb2_all_broadenings() -> None:
    # Broadening identities must use the actual Ry values, not ordinal labels.
    result = analyze_epc_directory(
        DATA,
        mu_stars=(0.10, 0.13, 0.15),
        broadenings_Ry=[0.005 * i for i in range(1, 11)],
    )
    expected = json.loads((DATA / "expected.json").read_text())
    assert len(result["spectra"]) == 10
    for saved in expected:
        spectrum = result["spectra"][int(saved["broadening_index"]) - 1]
        assert spectrum["lambda"] == pytest.approx(float(saved["lambda"]), rel=2e-7)
        assert spectrum["omega_log_K"] == pytest.approx(
            float(saved["omega_log_K"]), rel=2e-7
        )
        assert spectrum["omega_2_K"] == pytest.approx(
            float(saved["omega_2_K"]), rel=2e-7
        )
        for method, field in [
            ("modified_mcmillan_omega_log", "McMillan_omega_log_K"),
            ("allen_dynes_f1_f2", "Allen_Dynes_K"),
        ]:
            tc = next(
                r
                for r in spectrum["Tc"]
                if r["method"] == method and r["mu_star"] == float(saved["mu_star"])
            )
            assert tc["Tc_K"] == pytest.approx(float(saved[field]), rel=2e-7)
            assert tc["status"] == "estimated"
        assert len(spectrum["mode_indices"]) == 9
        assert np.min(spectrum["alpha2F"]) < 0
        assert "negative_spectral_weight" in spectrum["diagnostics"]
        assert spectrum["mode_sum_max_abs_error"] < 1e-5
    assert result["frequencies"]["number_of_modes"] == 9
    assert result["frequencies"]["number_of_qpoints"] == 1734
    assert result["instability"]["sampled_imaginary_modes"] is False
    assert result["phonon_dos"]["diagnostics"]["integrated_states"] == pytest.approx(
        9, abs=0.08
    )
    assert len(result["qe_qsum"]) == 10
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_reference_formula_values() -> None:
    """Method-matched constants from the independently executed reference class."""
    out = analyze_a2f(read_a2f(DATA / "a2F.dos5.gz"))
    assert out["lambda"] == pytest.approx(0.49899993668275183, rel=2e-7)
    assert out["omega_log_K"] == pytest.approx(776.942818, rel=2e-7)
    assert out["Tc"][0]["Tc_K"] == pytest.approx(9.369445815, rel=2e-7)
    assert out["Tc"][1]["Tc_K"] == pytest.approx(9.560764955, rel=2e-7)


def test_frequency_units(tmp_path: Path) -> None:
    raw = read_a2f(DATA / "a2F.dos5.gz")
    baseline = analyze_a2f(raw)
    ry_ev = physical_constants["Rydberg constant times hc in eV"][0]
    for unit, multiplier in [("eV", ry_ev), ("meV", ry_ev * 1000)]:
        converted = dict(
            raw,
            frequency=(np.array(raw["frequency"]) * multiplier).tolist(),
            frequency_unit=unit,
        )
        out = analyze_a2f(converted)
        assert out["lambda"] == pytest.approx(baseline["lambda"], rel=1e-12)
        assert out["omega_log_K"] == pytest.approx(baseline["omega_log_K"], rel=1e-12)
    with pytest.raises(ValueError, match="conflicts"):
        read_a2f(DATA / "a2F.dos5.gz", frequency_unit="meV")
    f = tmp_path / "unknown"
    f.write_text("1 2\n2 3\n3 4\n")
    with pytest.raises(ValueError, match="unavailable"):
        read_a2f(f)
    assert read_a2f(f, frequency_unit="Ry")["frequency_unit"] == "Ry"


@pytest.mark.parametrize("value", [0, -1])
def test_invalid_moments_not_zero_Tc(value: float) -> None:
    raw = read_a2f(DATA / "a2F.dos5.gz")
    raw["alpha2F"] = [value] * len(raw["frequency"])
    out = analyze_a2f(raw)
    assert out["status"] == "invalid"
    assert out["omega_log_K"] is None
    assert all(r["Tc_K"] is None and r["reason"] for r in out["Tc"])
    json.dumps(out, allow_nan=False)


def test_formula_domain_and_large_lambda() -> None:
    raw = read_a2f(DATA / "a2F.dos5.gz")
    out = analyze_a2f(raw, mu_stars=(1.0,))
    assert all(
        r["Tc_K"] is None and r["reason"] == "nonpositive_formula_denominator"
        for r in out["Tc"]
    )
    raw["alpha2F"] = (10 * np.array(raw["alpha2F"])).tolist()
    assert all(r["status"] == "estimated" for r in analyze_a2f(raw)["Tc"])
    with pytest.raises(ValueError, match="mu_stars"):
        analyze_a2f(raw, mu_stars=(float("nan"),))


def test_frequencies_path_and_overflow(tmp_path: Path) -> None:
    f = tmp_path / "freq"
    f.write_text(" &plot nbnd=3, nks=**** /\n0 0 0\n-20 1 2\n1 0 0\n3 4 5\n")
    result = read_matdyn_frequencies(f)
    assert result["weights"] == [None, None]
    assert result["frequencies"][0] == [-20, 1, 2]
    assert result["declared_nks"] is None
    assert result["number_of_qpoints"] == 2
    f.write_text(" &plot nbnd=3, nks=2 /\n0 0 0\n1 2\n")
    with pytest.raises(ValueError, match="Incomplete"):
        read_matdyn_frequencies(f)


def test_missing_and_unstable_outputs(tmp_path: Path) -> None:
    out = analyze_epc_directory(tmp_path, broadenings_Ry=(0.01, 0.02))
    assert out["status"] == "incomplete_or_invalid"
    assert out["instability"] is None
    assert [r["status"] for r in out["spectra"]] == ["unavailable", "unavailable"]
    for name in ["matdyn.dos", "a2F.dos1"]:
        (tmp_path / name).write_bytes(
            gzip.decompress((DATA / (name + ".gz")).read_bytes())
        )
    (tmp_path / "matdyn.freq").write_text(" &plot nbnd=3, nks=1 /\n0 0 0 1\n-20 1 2\n")
    partial = analyze_epc_directory(tmp_path, broadenings_Ry=(0.005, 0.01))
    assert partial["status"] == "incomplete_or_invalid"
    assert partial["spectra"][1]["status"] == "unavailable"
    out = analyze_epc_directory(tmp_path, broadenings_Ry=(0.005,))
    assert out["status"] == "sampled_instability"
    assert out["instability"]["below_tolerance_count"] == 1
    json.dumps(out, allow_nan=False)


def test_signed_dos_and_malformed_spectrum(tmp_path: Path) -> None:
    f = tmp_path / "dos"
    f.write_text("# Frequency[cm^-1] DOS PDOS\n-1 -2 -2\n0 0 0\n1 2 2\n")
    assert read_phonon_dos(f)["total_dos"] == [-2, 0, 2]
    f.write_text("# frequencies in Ry\n1 2\n1 3\n")
    with pytest.raises(ValueError, match="Non-increasing"):
        read_a2f(f)
    f.write_text("# frequencies in Ry\n1 NaN\n2 3\n")
    with pytest.raises(ValueError, match="nonfinite"):
        read_a2f(f)


def make_dfpt_source(path: Path) -> None:
    """Minimal file-layout fixture, not fake solver output used for physics."""
    path.mkdir(exist_ok=True)
    (path / "matdyn0").write_text("2 2 2\n2\n")
    for i in [1, 2]:
        (path / f"matdyn{i}").write_text("dynamical matrix fixture")
    elph = path / "elph_dir"
    elph.mkdir()
    for q in [1, 2]:
        (elph / f"elph.inp_lambda.{q}").write_text(
            "Gaussian Broadening: 0.005 Ry\nGaussian Broadening: 0.010 Ry\n"
        )
        for b in [51, 52]:
            (elph / f"a2Fq2r.{b}.{q}").write_text("fixture")


def test_staging_completeness_and_provenance(tmp_path: Path) -> None:
    make_dfpt_source(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = epc_file_manifest(tmp_path, "q2r")
    assert result["broadenings_Ry"] == [0.005, 0.01]
    assert result["qmesh"] == [2, 2, 2]
    assert all(
        p.read_bytes() == before[str(p)] for p in tmp_path.rglob("*") if p.is_file()
    )
    (tmp_path / "elph_dir/a2Fq2r.52.2").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        epc_file_manifest(tmp_path, "q2r")
    (tmp_path / "elph_dir/a2Fq2r.52.2").write_text("fixture")
    (tmp_path / "matdyn1").unlink()
    with pytest.raises(FileNotFoundError):
        epc_file_manifest(tmp_path, "q2r")


def test_missing_epc_and_force_constants(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing EPC"):
        epc_file_manifest(tmp_path, "q2r")
    make_dfpt_source(tmp_path)
    with pytest.raises(ValueError, match="Incomplete q2r"):
        epc_file_manifest(tmp_path, "matdyn")
    for i in [51, 52]:
        (tmp_path / f"elph_dir/a2Fmatdyn.{i}").write_text("fixture")
    with pytest.raises(FileNotFoundError):
        epc_file_manifest(tmp_path, "matdyn")
    (tmp_path / "q2r.fc").write_text("force constants")
    assert epc_file_manifest(tmp_path, "matdyn")["filenames"] == ["q2r.fc*", "elph_dir"]

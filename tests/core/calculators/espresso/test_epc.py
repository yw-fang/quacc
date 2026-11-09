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


@pytest.mark.parametrize("mu_star", [0.10, 0.13, 0.15])
def test_mgb2_spectrum_regression(mu_star: float) -> None:
    """One complete spectrum retains the physical quadrature regression."""
    spectrum = analyze_a2f(read_a2f(DATA / "a2F.dos5.gz"), mu_stars=(mu_star,))
    expected = next(
        row
        for row in json.loads((DATA / "expected.json").read_text())
        if float(row["mu_star"]) == mu_star
    )
    for key in ["lambda", "omega_log_K", "omega_2_K"]:
        assert spectrum[key] == pytest.approx(float(expected[key]), rel=2e-7)
    for tc, field in zip(
        spectrum["Tc"], ["McMillan_omega_log_K", "Allen_Dynes_K"], strict=True
    ):
        assert tc["Tc_K"] == pytest.approx(float(expected[field]), rel=2e-7)
        assert tc["status"] == "estimated"
        assert tc["mu_star"] == mu_star
    assert [tc["method"] for tc in spectrum["Tc"]] == [
        "modified_mcmillan_omega_log",
        "allen_dynes_f1_f2",
    ]
    assert len(spectrum["mode_indices"]) == 9
    assert np.min(spectrum["alpha2F"]) < 0
    assert "negative_spectral_weight" in spectrum["diagnostics"]
    assert spectrum["mode_sum_max_abs_error"] < 1e-5
    assert json.loads(json.dumps(spectrum, allow_nan=False)) == spectrum


def test_native_frequency_and_dos_excerpts() -> None:
    frequencies = read_matdyn_frequencies(DATA / "matdyn.freq")
    assert frequencies["number_of_modes"] == 9
    assert frequencies["number_of_qpoints"] == 2
    assert frequencies["weights"] == [0.000031, 0.000061]
    assert frequencies["frequencies"][0][-1] == 698.8808
    dos = read_phonon_dos(DATA / "matdyn.dos")
    assert dos["frequency_unit"] == "cm^-1"
    assert np.shape(dos["atom_projected_dos"]) == (3, 3)
    assert dos["total_dos"][1] == pytest.approx(3.5855890226e-8)


def test_frequency_units(tmp_path: Path) -> None:
    raw = read_a2f(DATA / "a2F.dos5.gz")
    baseline = analyze_a2f(raw)
    ry_ev = physical_constants["Rydberg constant times hc in eV"][0]
    for unit, multiplier in [
        ("eV", ry_ev),
        ("meV", ry_ev * 1000),
        (
            "cm^-1",
            ry_ev
            * physical_constants["electron volt-inverse meter relationship"][0]
            / 100,
        ),
    ]:
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
    (tmp_path / "matdyn.dos").write_text("# Frequency[cm^-1] DOS\n0 0\n1 1\n2 0\n")
    (tmp_path / "a2F.dos1").write_text("# Ry\n1 0.2\n2 0.4\n3 0.6\n")
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


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("", "nbnd"),
        (" &plot nbnd=0, nks=1 /\n", "nbnd"),
        (" &plot nbnd=3, nks=2 /\n0 0 0\n1 2 3\n", "nks disagrees"),
        (" &plot nbnd=3, nks=1 /\n0 0\n1 2 3\n", "three q coordinates"),
        (" &plot nbnd=3, nks=1 /\n0 0 0\n1 NaN 3\n", "nonfinite"),
        (" &plot nbnd=3, nks=1 /\n0 0 NaN\n1 2 3\n", "Nonfinite q"),
    ],
)
def test_invalid_frequency_files(tmp_path: Path, contents: str, message: str) -> None:
    f = tmp_path / "freq"
    f.write_text(contents)
    with pytest.raises(ValueError, match=message):
        read_matdyn_frequencies(f)


@pytest.mark.parametrize("unit", ["Ry", "eV", "meV", "cm^-1"])
def test_a2f_endpoints_and_mode_columns(tmp_path: Path, unit: str) -> None:
    f = tmp_path / "a2F.dos1"
    f.write_text(
        f"# frequencies in {unit}\n-1 -2 -1 -1\n0 0 0 0\n1 2 1 1\nlambda = 0.5D+00\n"
    )
    result = read_a2f(f)
    assert result["frequency"] == [-1, 0, 1]
    assert result["alpha2F"] == [-2, 0, 2]
    assert result["alpha2F_modes"] == [[-1, -1], [0, 0], [1, 1]]
    assert result["qe_footer_lambda"] == 0.5
    assert result["frequency_unit"] == unit
    assert result["source"]["bytes"] == f.stat().st_size
    assert Path(result["source"]["path"]).is_absolute()


def test_frequency_units_are_case_sensitive(tmp_path: Path) -> None:
    f = tmp_path / "spectrum"
    f.write_text("# frequencies in MeV\n1 2\n2 3\n3 4\n")
    with pytest.raises(ValueError, match="unavailable"):
        read_a2f(f)
    with pytest.raises(ValueError, match="Unsupported"):
        read_a2f(f, frequency_unit="THz")
    with pytest.raises(ValueError, match="Unsupported"):
        read_matdyn_frequencies(DATA / "matdyn.freq", frequency_unit="THz")


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("# Ry\n", "Missing"),
        ("# Ry\n1 2\n2 3 4\n", "inconsistent"),
        ("# Ry\n1 2\n2 bad\n", "Malformed"),
    ],
)
def test_malformed_a2f_tables(tmp_path: Path, contents: str, message: str) -> None:
    f = tmp_path / "a2F.dos1"
    f.write_text(contents)
    with pytest.raises(ValueError, match=message):
        read_a2f(f)


def test_three_point_moments(tmp_path: Path) -> None:
    """Simpson weights 1:4:1 give independent closed-form moments."""
    f = tmp_path / "a2F.dos1"
    f.write_text("# meV\n1 0.2 0.1 0.1\n2 0.4 0.2 0.2\n3 0.6 0.3 0.3\n")
    result = analyze_a2f(read_a2f(f))
    mev_to_K = 0.001 / physical_constants["Boltzmann constant in eV/K"][0]
    assert result["lambda"] == pytest.approx(0.8)
    assert result["mode_lambda"] == pytest.approx([0.4, 0.4])
    assert result["omega_log_K"] == pytest.approx(48 ** (1 / 6) * mev_to_K)
    assert result["omega_2_K"] == pytest.approx(np.sqrt(13 / 3) * mev_to_K)


@pytest.mark.parametrize(
    ("rows", "diagnostic"),
    [
        ("-1 2\n0 1\n1 1\n2 1\n", "insufficient_positive_frequency_points"),
        ("1 10\n2 1\n3 -10\n", "invalid_frequency_moments"),
    ],
)
def test_unavailable_moments(tmp_path: Path, rows: str, diagnostic: str) -> None:
    f = tmp_path / "a2F.dos1"
    f.write_text("# Ry\n" + rows)
    result = analyze_a2f(read_a2f(f))
    assert diagnostic in result["diagnostics"]
    assert all(tc["Tc_K"] is None for tc in result["Tc"])
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_broadening_series_diagnostics(tmp_path: Path) -> None:
    (tmp_path / "matdyn.freq").write_text(" &plot nbnd=3, nks=1 /\n0 0 0\n0 1 2\n")
    (tmp_path / "matdyn.dos").write_text("# cm^-1\n0 0\n1 3\n2 0\n")
    for i in [1, 3, 10]:
        (tmp_path / f"a2F.dos{i}").write_text("# Ry\n1 0.2\n2 0.4\n3 0.6\n")
    (tmp_path / "lambda").write_text(
        "Broadening 0.005 lambda 0.81 dos(Ef) 4.1 omega_ln [K] 750\n"
    )
    result = analyze_epc_directory(tmp_path)
    assert [s["index"] for s in result["spectra"]] == list(range(1, 11))
    assert result["spectra"][1]["status"] == "unavailable"
    assert result["spectra"][0]["qe_qsum"]["lambda"] == 0.81
    assert result["spectra"][0]["lambda"] == pytest.approx(0.8)
    assert result["spectra"][0]["broadening_Ry"] == 0.005
    assert result["phonon_dos"]["diagnostics"]["integrated_states"] == 4
    result = analyze_epc_directory(tmp_path, broadenings_Ry=(0.01,))
    assert result["spectra"][0]["reason"] == "broadening_identity_conflict"
    assert result["diagnostics"][0]["indices"] == [3, 10]
    (tmp_path / "a2F.dos3").write_text("# Ry\n1 bad\n")
    assert analyze_epc_directory(tmp_path)["spectra"][2]["status"] == "invalid"
    (tmp_path / "a2F.dos1.gz").write_bytes(b"ambiguous")
    with pytest.raises(ValueError, match="Ambiguous duplicate"):
        analyze_epc_directory(tmp_path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mu_stars": ()},
        {"mu_stars": (-0.1,)},
        {"broadenings_Ry": ()},
        {"broadenings_Ry": (float("inf"),)},
        {"imaginary_tolerance_cm1": -1},
        {"imaginary_tolerance_cm1": float("nan")},
    ],
)
def test_invalid_analysis_settings(tmp_path: Path, kwargs: dict) -> None:
    with pytest.raises(ValueError, match=r"finite|nonnegative|positive"):
        analyze_epc_directory(tmp_path, **kwargs)


@pytest.mark.parametrize(
    "problem", ["metadata", "qpoint", "header", "duplicate", "extra_qpoint"]
)
def test_invalid_dfpt_layout(tmp_path: Path, problem: str) -> None:
    make_dfpt_source(tmp_path)
    if problem == "metadata":
        (tmp_path / "elph_dir/elph.inp_lambda.2").write_text(
            "Gaussian Broadening: 0.03 Ry\n"
        )
    elif problem == "qpoint":
        (tmp_path / "elph_dir/elph.inp_lambda.2").unlink()
    elif problem == "header":
        (tmp_path / "matdyn0").write_text("0 1 1\n2\n")
    elif problem == "duplicate":
        (tmp_path / "elph_dir/a2Fq2r.51.1.gz").write_bytes(gzip.compress(b"duplicate"))
    else:
        (tmp_path / "elph_dir/a2Fq2r.51.3").write_text("extra")
    with pytest.raises(ValueError, match=r"Inconsistent|Incomplete|Invalid|Ambiguous"):
        epc_file_manifest(tmp_path, "q2r")


def test_invalid_manifest_options_and_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stage"):
        epc_file_manifest(tmp_path, "invalid")
    with pytest.raises(ValueError, match="basename"):
        epc_file_manifest(tmp_path, "q2r", "../matdyn")
    (tmp_path / "elph_dir").mkdir()
    with pytest.raises(FileNotFoundError, match="broadening metadata"):
        epc_file_manifest(tmp_path, "q2r")
    (tmp_path / "elph_dir/elph.inp_lambda.1").write_text(
        "Gaussian Broadening: -0.01 Ry\n"
    )
    with pytest.raises(ValueError, match="finite positive"):
        epc_file_manifest(tmp_path, "q2r")


def test_duplicate_transformed_epc_files(tmp_path: Path) -> None:
    make_dfpt_source(tmp_path)
    (tmp_path / "q2r.fc").write_text("fixture")
    for name in ["a2Fmatdyn.51", "a2Fmatdyn.51.gz"]:
        (tmp_path / "elph_dir" / name).write_bytes(gzip.compress(b"duplicate"))
    with pytest.raises(ValueError, match="Ambiguous duplicate"):
        epc_file_manifest(tmp_path, "matdyn")


def test_tc_underflow_is_unavailable(tmp_path: Path) -> None:
    f = tmp_path / "a2F.dos1"
    f.write_text("# Ry\n1 0.00001\n2 0.00002\n3 0.00003\n")
    result = analyze_a2f(read_a2f(f), mu_stars=(0.0,))
    assert result["lambda"] > 0
    assert all(tc["Tc_K"] is None for tc in result["Tc"])
    assert all(tc["reason"] == "Tc_numerically_unrepresentable" for tc in result["Tc"])
    json.dumps(result, allow_nan=False)


def test_one_bin_dos_is_not_zero_integrated_states(tmp_path: Path) -> None:
    (tmp_path / "matdyn.dos").write_text("# cm^-1\n1 2\n")
    diagnostics = analyze_epc_directory(tmp_path)["phonon_dos"]["diagnostics"]
    assert diagnostics["integrated_states"] is None
    assert diagnostics["status"] == "insufficient_grid_points"

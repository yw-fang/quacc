"""Offline EPC regressions against preserved QE 7.5 one-atom spectra."""

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
PB = DATA / "pb"

# Independent reference values from hyllios/utils autoph/allen_dynes.py
# (commit 4e86e82ae2eec3762f66e6228a70f67b69570789); see epc_data/README.md.
# The 2e-7 relative tolerance locks the same algorithm; it is not an accuracy bound.
PB_MOMENTS = {
    "lambda": 7.875436476796867,
    "omega_log_K": 64.2876372164752,
    "omega_2_K": 68.04225513743206,
}
PB_TC_K = {  # mu* -> (modified McMillan, Allen-Dynes)
    0.10: (15.095127126096326, 26.241952273952318),
    0.13: (14.6279463102742, 24.626938176283097),
    0.15: (14.311930611917951, 23.624487751414375),
}
AL_MOMENTS = {
    "lambda": 0.3707535899829175,
    "omega_log_K": 325.0235690931655,
    "omega_2_K": 344.6259412608493,
}
AL_TC_K = (0.8588558676754687, 0.8697846981057307)  # mu* = 0.10


def test_al_example03_spectrum_regression() -> None:
    """Official Al example covers weak coupling and signed mode columns."""
    spectrum = analyze_a2f(read_a2f(DATA / "al" / "a2F.dos10.gz"), mu_stars=(0.10,))
    for key, value in AL_MOMENTS.items():
        assert spectrum[key] == pytest.approx(value, rel=2e-7)
    for tc, value in zip(spectrum["Tc"], AL_TC_K, strict=True):
        assert tc["Tc_K"] == pytest.approx(value, rel=2e-7)
        assert tc["status"] == "estimated"
    assert 0 < spectrum["lambda"] < 0.5
    assert np.shape(spectrum["alpha2F_modes"]) == (50, 3)
    assert np.any(np.asarray(spectrum["alpha2F_modes"]) < 0)
    assert all(value >= 0 for value in spectrum["alpha2F"])
    assert "large_lambda_formula_regime" not in spectrum["warnings"]
    assert abs(spectrum["qe_midpoint_footer_difference"]) < 1e-3
    # Different coarse-bin quadratures remain distinct, with a diagnostic.
    assert (
        "Simpson_positive_frequency_vs_QE_midpoint_disagreement" in spectrum["warnings"]
    )
    assert json.loads(json.dumps(spectrum, allow_nan=False)) == spectrum


@pytest.mark.parametrize("mu_star", [0.10, 0.13, 0.15])
def test_pb_spectrum_regression(mu_star: float) -> None:
    """One complete spectrum retains the physical quadrature regression."""
    spectrum = analyze_a2f(read_a2f(PB / "a2F.dos1.gz"), mu_stars=(mu_star,))
    for key, value in PB_MOMENTS.items():
        assert spectrum[key] == pytest.approx(value, rel=2e-7)
    for tc, value in zip(spectrum["Tc"], PB_TC_K[mu_star], strict=True):
        assert tc["Tc_K"] == pytest.approx(value, rel=2e-7)
        assert tc["status"] == "estimated"
        assert tc["mu_star"] == mu_star
    assert [tc["method"] for tc in spectrum["Tc"]] == [
        "modified_mcmillan_omega_log",
        "allen_dynes_f1_f2",
    ]
    assert len(spectrum["mode_indices"]) == 3
    assert spectrum["lambda"] > 5
    assert "large_lambda_formula_regime" in spectrum["warnings"]
    assert abs(spectrum["qe_midpoint_footer_difference"]) < 1e-3
    # Independent E16.6 rounding of every column: sum their absolute error bounds,
    # including signed cancellation, rather than imposing a physical mismatch.
    modes = np.asarray(spectrum["alpha2F_modes"])
    total = np.asarray(spectrum["alpha2F"])
    rounding_bound = 5e-6 * (abs(total) + np.sum(abs(modes), axis=1)) + 1e-12
    assert np.all(abs(modes.sum(axis=1) - total) <= rounding_bound)
    assert json.loads(json.dumps(spectrum, allow_nan=False)) == spectrum


def test_native_frequency_and_dos_excerpts() -> None:
    frequencies = read_matdyn_frequencies(PB / "matdyn.freq")
    assert frequencies["number_of_modes"] == 3
    assert frequencies["number_of_qpoints"] == 2
    assert frequencies["weights"] == [0.001953, 0.015625]
    assert frequencies["frequencies"][0][-1] == 0.0
    dos = read_phonon_dos(PB / "matdyn.dos")
    assert dos["frequency_unit"] == "cm^-1"
    assert np.shape(dos["atom_projected_dos"]) == (3, 1)
    assert dos["total_dos"][1] == pytest.approx(6.1326590894e-05)


def test_frequency_units(tmp_path: Path) -> None:
    raw = read_a2f(PB / "a2F.dos1.gz")
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
            qe_footer_delta_frequency=raw["qe_footer_delta_frequency"] * multiplier,
        )
        out = analyze_a2f(converted)
        assert out["lambda"] == pytest.approx(baseline["lambda"], rel=1e-12)
        assert out["qe_midpoint_lambda"] == pytest.approx(
            baseline["qe_midpoint_lambda"], rel=1e-12
        )
        assert out["omega_log_K"] == pytest.approx(baseline["omega_log_K"], rel=1e-12)
    with pytest.raises(ValueError, match="conflicts"):
        read_a2f(PB / "a2F.dos1.gz", frequency_unit="meV")
    f = tmp_path / "unknown"
    f.write_text("1 2\n2 3\n3 4\n")
    with pytest.raises(ValueError, match="unavailable"):
        read_a2f(f)
    assert read_a2f(f, frequency_unit="Ry")["frequency_unit"] == "Ry"


@pytest.mark.parametrize("value", [0, -1])
def test_invalid_moments_not_zero_Tc(value: float) -> None:
    raw = read_a2f(PB / "a2F.dos1.gz")
    raw["alpha2F"] = [value] * len(raw["frequency"])
    out = analyze_a2f(raw)
    assert out["status"] == "invalid"
    assert out["omega_log_K"] is None
    assert all(r["Tc_K"] is None and r["reason"] for r in out["Tc"])
    json.dumps(out, allow_nan=False)


def test_formula_domain_and_large_lambda() -> None:
    raw = read_a2f(PB / "a2F.dos1.gz")
    out = analyze_a2f(raw, mu_stars=(2.0,))
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
    elph.mkdir(exist_ok=True)
    for q in [1, 2]:
        (elph / f"elph.inp_lambda.{q}").write_text(
            "Gaussian Broadening: 0.005 Ry\nGaussian Broadening: 0.010 Ry\n"
        )
        for b in [51, 52]:
            (elph / f"a2Fq2r.{b}.{q}").write_text(
                f"{0.005 * (b - 50):.16E} 0.5 4.0\nfixture"
            )


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
    (tmp_path / "elph_dir/a2Fq2r.52.2").write_text("0.01 0.5 4.0\nfixture")
    (tmp_path / "matdyn1").unlink()
    with pytest.raises(FileNotFoundError, match="Missing EPC file"):
        epc_file_manifest(tmp_path, "q2r")


def test_missing_epc_and_force_constants(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing EPC"):
        epc_file_manifest(tmp_path, "q2r")
    make_dfpt_source(tmp_path)
    (tmp_path / "q2r.fc").write_text("fixture")
    with pytest.raises(ValueError, match="Incomplete or incorrect q2r"):
        epc_file_manifest(tmp_path, "matdyn")
    for i in [61, 62]:
        (tmp_path / f"elph_dir/a2Fmatdyn.{i}").write_text(
            f"{0.005 * (i - 60)} 0.5 4.0\nfixture"
        )
    (tmp_path / "q2r.fc").unlink()
    with pytest.raises(FileNotFoundError, match="Missing EPC file"):
        epc_file_manifest(tmp_path, "matdyn")
    (tmp_path / "q2r.fc").write_text("force constants")
    assert set(epc_file_manifest(tmp_path, "matdyn")["filenames"]) == {
        "q2r.fc",
        "elph_dir/elph.inp_lambda.1",
        "elph_dir/elph.inp_lambda.2",
        "elph_dir/a2Fmatdyn.61",
        "elph_dir/a2Fmatdyn.62",
    }


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
        read_matdyn_frequencies(PB / "matdyn.freq", frequency_unit="THz")


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
    [("-1 2\n0 1\n1 1\n2 1\n", "insufficient_positive_frequency_points")],
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
    assert result["qe_qsum"][0]["lambda"] == 0.81
    assert result["spectra"][0]["lambda"] == pytest.approx(0.8)
    assert result["spectra"][0]["broadening_Ry"] is None
    assert result["phonon_dos"]["diagnostics"]["integrated_states"] == 4
    result = analyze_epc_directory(tmp_path, broadenings_Ry=(0.01,))
    assert result["spectra"][0]["reason"] == "broadening_identity_conflict"
    assert result["diagnostics"][0]["indices"] == [3, 10]
    (tmp_path / "a2F.dos3").write_text("# Ry\n1 bad\n")
    assert analyze_epc_directory(tmp_path)["spectra"][2]["status"] == "invalid"
    (tmp_path / "a2F.dos1.gz").write_bytes(b"ambiguous")
    assert analyze_epc_directory(tmp_path)["spectra"][0]["status"] == "invalid"


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
    message = (
        "mu_stars"
        if "mu_stars" in kwargs
        else "Broadenings"
        if "broadenings_Ry" in kwargs
        else "Imaginary-frequency tolerance"
    )
    with pytest.raises(ValueError, match=message):
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
    message = {
        "metadata": "Inconsistent/missing",
        "qpoint": "Incomplete/ambiguous",
        "header": "Invalid DFPT",
        "duplicate": "Ambiguous plain/gzip",
        "extra_qpoint": "Incomplete or incorrect",
    }[problem]
    with pytest.raises(ValueError, match=message):
        epc_file_manifest(tmp_path, "q2r")


def test_invalid_manifest_options_and_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stage"):
        epc_file_manifest(tmp_path, "invalid")
    with pytest.raises(ValueError, match="basename"):
        epc_file_manifest(tmp_path, "q2r", "../matdyn")
    (tmp_path / "elph_dir").mkdir()
    with pytest.raises(FileNotFoundError, match="broadening metadata"):
        epc_file_manifest(tmp_path, "q2r")
    make_dfpt_source(tmp_path)
    for q in [1, 2]:
        (tmp_path / f"elph_dir/a2Fq2r.51.{q}").write_text("-0.01 0.5 4.0\n")
    with pytest.raises(ValueError, match="finite positive"):
        epc_file_manifest(tmp_path, "q2r")


def test_duplicate_transformed_epc_files(tmp_path: Path) -> None:
    make_dfpt_source(tmp_path)
    (tmp_path / "q2r.fc").write_text("fixture")
    for name in ["a2Fmatdyn.61", "a2Fmatdyn.61.gz"]:
        (tmp_path / "elph_dir" / name).write_bytes(gzip.compress(b"duplicate"))
    with pytest.raises(ValueError, match="Ambiguous plain/gzip"):
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
    assert diagnostics["status"] == "insufficient_or_nonfinite_grid_integral"


@pytest.mark.parametrize("sigma", [0.0025, 0.0004])
def test_full_precision_broadening_headers(tmp_path: Path, sigma: float) -> None:
    make_dfpt_source(tmp_path)
    for q in [1, 2]:
        (tmp_path / f"elph_dir/elph.inp_lambda.{q}").write_text(
            "".join(
                f"     Gaussian Broadening: {i * sigma:7.3f} Ry, ngauss=   0\n"
                for i in [1, 2]
            )
        )
        for i in [1, 2]:
            (tmp_path / f"elph_dir/a2Fq2r.{50 + i}.{q}").write_text(
                f"{i * sigma:.16E} 0.5 4.0\n"
            )
    assert epc_file_manifest(tmp_path, "q2r")["broadenings_Ry"] == [sigma, 2 * sigma]


@pytest.mark.parametrize("stage", ["q2r", "matdyn"])
def test_exact_qe_file_number_sets(tmp_path: Path, stage: str) -> None:
    make_dfpt_source(tmp_path)
    if stage == "q2r":
        for q in [1, 2]:
            (tmp_path / f"elph_dir/a2Fq2r.52.{q}").rename(
                tmp_path / f"elph_dir/a2Fq2r.53.{q}"
            )
    else:
        (tmp_path / "q2r.fc").write_text("fixture")
        for i in [51, 99]:
            (tmp_path / f"elph_dir/a2Fmatdyn.{i}").write_text("0.0025 0.5 4.0\n")
    with pytest.raises(ValueError, match="Incomplete or incorrect"):
        epc_file_manifest(tmp_path, stage)


@pytest.mark.parametrize("name", ["matdyn0", "matdyn1", "q2r.fc"])
def test_ambiguous_staged_files(tmp_path: Path, name: str) -> None:
    make_dfpt_source(tmp_path)
    (tmp_path / "q2r.fc").write_text("fixture")
    (tmp_path / (name + ".gz")).write_bytes(gzip.compress(b"different"))
    with pytest.raises(ValueError, match="Ambiguous plain/gzip"):
        epc_file_manifest(tmp_path, "matdyn" if name == "q2r.fc" else "q2r")


@pytest.mark.parametrize("name", ["matdyn.freq", "matdyn.dos", "lambda", "a2F.dos1"])
def test_duplicate_output_is_a_component_diagnostic(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text("fixture")
    (tmp_path / (name + ".gz")).write_bytes(gzip.compress(b"different"))
    result = analyze_epc_directory(tmp_path, broadenings_Ry=(0.0025,))
    assert result["status"] == "incomplete_or_invalid"
    assert result["diagnostics"]
    assert all(tc["status"] != "estimated" for tc in result["spectra"][0]["Tc"])
    json.dumps(result, allow_nan=False)


def test_qsum_rows_are_matched_by_value(tmp_path: Path) -> None:
    for i in [1, 2, 3]:
        (tmp_path / f"a2F.dos{i}").write_text("# Ry\n1 .2\n2 .4\n3 .6\nlambda = .8\n")
    (tmp_path / "lambda").write_text(
        " Broadening 0.0075 lambda 0.83 dos(Ef)120.0000 omega_ln [K]700\n"
        " Broadening 0.0050 lambda 0.82 dos(Ef)4.0 omega_ln [K]NaN\n"
        " Broadening 0.0025 lambda 0.81 dos(Ef)4.0 omega_ln [K]750\n"
        " Broadening bad incomplete line\n"
    )
    result = analyze_epc_directory(tmp_path, broadenings_Ry=(0.0025, 0.005, 0.0075))
    assert [s["qe_qsum"]["lambda"] for s in result["spectra"]] == [0.81, 0.82, 0.83]
    assert result["spectra"][1]["qe_qsum"]["omega_log_K"] is None
    assert any(d["status"] == "malformed_row" for d in result["diagnostics"])
    assert any(d["status"] == "nonfinite_row" for d in result["diagnostics"])
    assert all(s["Tc"][0]["status"] == "estimated" for s in result["spectra"])
    unknown = analyze_epc_directory(tmp_path)
    assert all(
        s["broadening_Ry"] is None and "qe_qsum" not in s for s in unknown["spectra"]
    )
    json.dumps(result, allow_nan=False)


def test_identity_conflict_invalidates_tc(tmp_path: Path) -> None:
    (tmp_path / "a2F.dos1").write_text("# Ry\n1 .2\n2 .4\n3 .6\nlambda = .8\n")
    (tmp_path / "lambda").write_text(
        "Broadening .01 lambda .8 dos(Ef)4 omega_ln [K]750\n"
    )
    spectrum = analyze_epc_directory(tmp_path, broadenings_Ry=(0.0025,))["spectra"][0]
    assert spectrum["status"] == "invalid"
    assert spectrum["lambda"] is None
    assert all(
        tc["Tc_K"] is None and tc["reason"] == "broadening_identity_conflict"
        for tc in spectrum["Tc"]
    )


def test_totals_only_spectrum_and_method_domains(tmp_path: Path) -> None:
    f = tmp_path / "a2F.dos1"
    f.write_text("# Ry\n1 .2\n2 .4\n3 .6\n4 .8\n5 1.0\nlambda = 1.6\n")
    result = analyze_a2f(read_a2f(f))
    assert result["mode_lambda"] == []
    assert result["lambda"] == pytest.approx(1.6)
    f.write_text("# Ry\n1 10\n2 1\n3 -10\nlambda = 5.77\n")
    result = analyze_a2f(read_a2f(f))
    assert result["omega_log_K"] > 0
    assert result["omega_2_K"] is None
    assert result["Tc"][0]["status"] == "estimated"
    assert result["Tc"][1]["Tc_K"] is None
    assert "Tc_unavailable" in result["warnings"]


def test_fixed_width_imaginary_modes_and_overflow(tmp_path: Path) -> None:
    f = tmp_path / "matdyn.freq"
    f.write_text(
        " &plot nbnd=3, nks=1 /\n0 0 0\n"
        + "".join(f"{v:10.4f}" for v in [-1234.5678, -2345.6789, 3])
        + "\n"
    )
    assert read_matdyn_frequencies(f)["frequencies"][0] == [-1234.5678, -2345.6789, 3]
    f.write_text(" &plot nbnd=3, nks=1 /\n0 0 0\n**********    1.0000    2.0000\n")
    with pytest.raises(ValueError, match="frequency field overflow"):
        read_matdyn_frequencies(f)
    result = analyze_epc_directory(tmp_path)
    assert result["frequencies"] is None
    assert "overflow" in result["diagnostics"][0]["reason"]


def test_interrupted_tables_and_ambiguous_header_units(tmp_path: Path) -> None:
    f = tmp_path / "a2F.dos1"
    f.write_text("# Ry\n1 .2\nBROKEN ROW\n2 .4\n3 .6\n")
    with pytest.raises(ValueError, match="Non-numeric interruption"):
        read_a2f(f)
    f.write_text("# cm^-1 and Ry\n1 .2\n2 .4\n3 .6\n")
    with pytest.raises(ValueError, match="Ambiguous frequency units"):
        read_a2f(f)
    f.write_text("# RY\n1 .2\n2 .4\n3 .6\n# eV is only a footer annotation\n")
    raw = read_a2f(f)
    assert raw["frequency_unit"] == "Ry"
    assert "truncation" in raw["diagnostics"][0]


def test_nonfinite_derived_quantities_remain_json_serializable(tmp_path: Path) -> None:
    f = tmp_path / "a2F.dos1"
    f.write_text("# Ry\n1e-320 1e300 1e300\n2e-320 1e300 1e300\n3e-320 1e300 1e300\n")
    result = analyze_a2f(read_a2f(f))
    assert result["lambda"] is None
    assert result["mode_lambda"] == [None]
    assert "nonfinite_mode_integral" in result["warnings"]
    json.dumps(result, allow_nan=False)


def test_compression_independent_provenance(tmp_path: Path) -> None:
    from quacc.calculators.espresso.epc import file_provenance

    p = tmp_path / "data"
    p.write_text("same scientific bytes\n")
    first = file_provenance(p)
    p.with_name("data.gz").write_bytes(gzip.compress(p.read_bytes(), mtime=10))
    p.unlink()
    second = file_provenance(p)
    assert first["uncompressed_sha256"] == second["uncompressed_sha256"]
    assert first["sha256"] != second["sha256"]
    assert second["compression"] == "gzip"


def test_instability_units_and_tc_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quacc.calculators.espresso import epc

    (tmp_path / "matdyn.freq").write_text(" &plot nbnd=3, nks=1 /\n0 0 0\n-20 1 2\n")
    (tmp_path / "matdyn.dos").write_text("# cm^-1\n0 0\n1 2.25\n2 0\n")
    (tmp_path / "a2F.dos1").write_text("# Ry\n1 .2\n2 .4\n3 .6\nlambda = .8\n")
    native = epc.read_matdyn_frequencies(tmp_path / "matdyn.freq")
    cm_per_ev = physical_constants["electron volt-inverse meter relationship"][0] / 100
    native.update(
        frequencies=(np.array(native["frequencies"]) / cm_per_ev).tolist(),
        frequency_unit="eV",
    )
    monkeypatch.setattr(epc, "read_matdyn_frequencies", lambda _: native)
    result = epc.analyze_epc_directory(tmp_path, broadenings_Ry=(0.0025,))
    assert result["instability"]["minimum_frequency_cm1"] == pytest.approx(-20)
    assert result["status"] == "sampled_instability"
    assert "sampled_instability" in result["spectra"][0]["Tc"][0]["warnings"][0]
    native["frequencies"] = [[-1e308, 1e308, 0]]
    result = epc.analyze_epc_directory(tmp_path, broadenings_Ry=(0.0025,))
    assert result["instability"] is None
    assert result["status"] == "incomplete_or_invalid"
    json.dumps(result, allow_nan=False)


def test_qe_midpoint_anchor_is_distinct_from_positive_simpson(tmp_path: Path) -> None:
    f = tmp_path / "a2F.dos1"
    # Uniform bin-centred values include a nonpositive-frequency contribution.
    f.write_text("# Ry\n-1 .2\n1 .4\n3 .6\n5 .8\nlambda = 2.24 Delta = 2\n")
    result = analyze_a2f(read_a2f(f))
    assert result["qe_midpoint_lambda"] == pytest.approx(2.24)
    assert result["qe_midpoint_footer_difference"] == pytest.approx(0)
    assert (
        "Simpson_positive_frequency_vs_QE_midpoint_disagreement" in result["warnings"]
    )

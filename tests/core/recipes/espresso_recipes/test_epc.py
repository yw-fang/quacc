"""Exercise flow customization and internal staging without launching QE."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from quacc.calculators.espresso.espresso import EspressoTemplate
from quacc.recipes.espresso import epc, phonons
from quacc.utils.files import copy_decompress_files

DATA_DIR = Path(__file__).parent / "data"
# Saved Pb matdyn outputs shared with the offline parser tests.
EPC_OUTPUTS = Path(__file__).parents[2] / "calculators/espresso/epc_data/pb"


def test_analysis_without_plotting(monkeypatch: Any) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    result = epc.epc_analysis_job({}, {"dir_name": str(EPC_OUTPUTS)})
    assert sum("alpha2F" in spectrum for spectrum in result["spectra"]) == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("compressed", [False, True])
def test_epc_flow_staging_and_customization(
    tmp_path: Path, monkeypatch: Any, compressed: bool
) -> None:
    source = tmp_path / "completed_dfpt"
    source.mkdir()
    (source / "custom.dyn0").write_text("1 1 1\n1\n")
    (source / "custom.dyn1").write_text("fixture")
    elph = source / "elph_dir"
    elph.mkdir()
    (elph / "elph.inp_lambda.1").write_text(
        "Gaussian Broadening: 0.005 Ry\nGaussian Broadening: 0.010 Ry\n"
    )
    for i in [51, 52]:
        (elph / f"a2Fq2r.{i}.1").write_text(f"{0.005 * (i - 50):.16E} 0.5 4.0\nfixture")
    if compressed:
        for path in list(source.rglob("*")):
            if path.is_file():
                path.with_name(path.name + ".gz").write_bytes(
                    gzip.compress(path.read_bytes())
                )
                path.unlink()
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    calls = []

    def fake_run(
        template: EspressoTemplate,
        calc_defaults: Any,
        calc_swaps: Any,
        additional_fields: Any,
        copy_files: Any,
    ) -> dict[str, Any]:
        target = tmp_path / template.binary
        target.mkdir()
        for spec in copy_files:
            copy_decompress_files(spec["source"], spec["filenames"], target)
        calls.append(template.binary)
        assert (target / "elph_dir/elph.inp_lambda.1").is_file()
        if template.binary == "q2r":
            assert (target / "custom.dyn1").is_file()
            (target / "q2r.fc").write_text("fixture")
            assert calc_swaps["input_data"]["input"] == {
                "la2f": True,
                "zasr": "simple",
                "fildyn": "custom.dyn",
                "el_ph_nsigma": 2,
            }
            for i in [61, 62]:
                (target / f"elph_dir/a2Fmatdyn.{i}").write_text(
                    f"{0.005 * (i - 60):.16E} 0.5 4.0\nfixture"
                )
            if compressed:
                for path in list(target.rglob("*")):
                    if path.is_file():
                        path.with_name(path.name + ".gz").write_bytes(
                            gzip.compress(path.read_bytes())
                        )
                        path.unlink()
        else:
            assert (target / "q2r.fc").is_file()
            assert calc_swaps["input_data"]["input"]["nk1"] == 6
            (target / "matdyn.freq").write_text(
                " &plot nbnd=3, nks=1 /\n0 0 0\n0 1 2\n"
            )
            (target / "matdyn.dos").write_text(
                "# Frequency[cm^-1] DOS\n0 0\n1 2.25\n2 0\n"
            )
            for i, scale in [(1, 1), (2, 2)]:
                (target / f"a2F.dos{i}").write_text(
                    f"# frequencies in Ry\n1 {0.2 * scale}\n2 {0.4 * scale}\n3 {0.6 * scale}\nlambda = {0.8 * scale}\n"
                )
        (target / f"{template.binary}.in").write_text(str(calc_swaps))
        return {"dir_name": str(target), **additional_fields}

    def decorator(function: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            calls.append(function.__name__)
            return function(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(phonons, "run_and_summarize", fake_run)
    output = epc.epc_postprocess_flow(
        source,
        mu_stars=(0.13,),
        job_params={
            "q2r_job": {"input_data": {"input": {"fildyn": "custom.dyn"}}},
            "matdyn_job": {"input_data": {"input": {"nk1": 6}}},
        },
        job_decorators={"epc_analysis_job": decorator},
    )
    assert calls == ["q2r", "matdyn", "epc_analysis_job"]
    result = output["epc_results"]
    assert len(result["spectra"]) == 2
    assert [s["broadening_Ry"] for s in result["spectra"]] == [0.005, 0.01]
    assert [s["lambda"] for s in result["spectra"]] == pytest.approx([0.8, 1.6])
    assert result["mu_stars"] == [0.13]
    assert "6" in result["settings"]["matdyn"]
    assert result["status"] == "analyzed"
    assert result["input_provenance"]["DFPT"]["qmesh"] == [1, 1, 1]
    json.dumps(result, allow_nan=False)
    assert all(p.read_bytes() == data for p, data in before.items())


def test_q2r_custom_prefix_is_not_overwritten(tmp_path: Path) -> None:
    template = EspressoTemplate("q2r")
    result = template._output_handler(
        {"input_data": {"input": {"fildyn": "qe.dyn"}}}, tmp_path
    )
    assert result["input_data"]["input"]["fildyn"] == "qe.dyn"
    assert result["input_data"]["input"]["flfrc"] == "q2r.fc"


@pytest.mark.parametrize("binary", ["q2r", "matdyn"])
def test_non_epc_source_and_legacy_copy_files(
    tmp_path: Path, monkeypatch: Any, binary: str
) -> None:
    calls = []

    def capture(**kwargs: Any) -> dict:
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(phonons, "run_and_summarize", capture)
    recipe = getattr(phonons, f"{binary}_job")
    extra_files = [{"source": tmp_path / "extra", "filenames": "input"}]
    # The original positional copy_files API still works without EPC validation.
    recipe(extra_files)
    assert calls[-1]["copy_files"] == extra_files
    recipe(copy_files=extra_files, source_directory=tmp_path)
    assert calls[-1]["copy_files"] == [
        *extra_files,
        {
            "source": tmp_path,
            "filenames": ["matdyn[0-9]*" if binary == "q2r" else "q2r.fc*"],
        },
    ]
    assert "epc_input_manifest" not in calls[-1]["additional_fields"]


@pytest.mark.parametrize("compressed_source", [False, True])
def test_native_pb_epc_postprocess(tmp_path: Path, compressed_source: bool) -> None:
    """Run only q2r/matdyn against the tiny, completed QE 7.5 Pb source."""
    import hashlib
    from shutil import copytree, which

    from monty.io import zopen
    from monty.os.path import zpath

    from quacc import change_settings, get_settings

    settings = get_settings()
    if any(
        which(str(settings.ESPRESSO_BIN_DIR / settings.ESPRESSO_BINARIES[binary]))
        is None
        for binary in ("q2r", "matdyn")
    ):
        pytest.skip("QE q2r/matdyn not installed")
    source = tmp_path / "readonly_dfpt"
    if compressed_source:
        copytree(DATA_DIR / "epc_test", source)
    else:
        copy_decompress_files(DATA_DIR / "epc_test", "*", source)
    for p in source.glob("matdyn*"):
        p.rename(p.with_name(p.name.replace("matdyn", "pb.dyn", 1)))
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*")
        if p.is_file()
    }
    for p in source.rglob("*"):
        p.chmod(0o555 if p.is_dir() else 0o444)
    source.chmod(0o555)
    try:
        with change_settings(
            {
                "RESULTS_DIR": tmp_path / "results",
                "SCRATCH_DIR": tmp_path / "scratch",
                "CREATE_UNIQUE_DIR": True,
            }
        ):
            output = epc.epc_postprocess_flow(
                source,
                mu_stars=(0.1, 0.13),
                job_params={
                    "q2r_job": {"input_data": {"input": {"fildyn": "pb.dyn"}}},
                    "matdyn_job": {
                        "input_data": {
                            "input": {"nk1": 8, "nk2": 8, "nk3": 8, "ndos": 120}
                        }
                    },
                },
            )
        result = output["epc_results"]
        assert [s["broadening_Ry"] for s in result["spectra"]] == [0.0025, 0.005]
        assert result["status"] in (
            "analyzed",
            "analyzed_with_warnings",
            "sampled_instability",
        )
        for spectrum in result["spectra"]:
            assert abs(spectrum["qe_midpoint_footer_difference"]) < 1e-3
            assert spectrum["qe_qsum"]["broadening_Ry"] == spectrum["broadening_Ry"]
            assert all(tc["status"] == "estimated" for tc in spectrum["Tc"])
        for binary in ("q2r", "matdyn"):
            with zopen(
                zpath(Path(output[f"{binary}_job"]["dir_name"]) / f"{binary}.out"), "rt"
            ) as f:
                assert "JOB DONE." in f.read()
        assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before} == before
        assert all(p.stat().st_mode & 0o222 == 0 for p in before)
        json.dumps(result, allow_nan=False)
    finally:
        # Let pytest remove its private read-only fixture copy on every platform.
        source.chmod(0o755)
        for p in source.rglob("*"):
            p.chmod(0o755 if p.is_dir() else 0o644)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mu_stars": (-0.1,)},
        {"imaginary_tolerance_cm1": -1},
        {"job_params": {"all": {"input_data": {}}}},
        {"job_params": {"epc_analysis_job": {"mu_stars": ()}}},
        {"job_params": {"q2r_job": {"input_data": {"la2f": True}}}},
        {"job_params": {"matdyn_job": {"input_data": {"input": {"dos": False}}}}},
    ],
)
def test_flow_settings_fail_before_solver_calls(
    tmp_path: Path, monkeypatch: Any, kwargs: dict
) -> None:
    def forbidden(**options: Any) -> None:
        pytest.fail("Solver called before settings validation")

    monkeypatch.setattr(phonons, "run_and_summarize", forbidden)
    expected = (
        "mu_stars"
        if "mu_stars" in kwargs or "epc_analysis_job" in kwargs.get("job_params", {})
        else "Imaginary-frequency"
        if "imaginary_tolerance_cm1" in kwargs
        else "unsupported"
        if "all" in kwargs.get("job_params", {})
        else "nested"
        if "q2r_job" in kwargs.get("job_params", {})
        else "dos=True"
    )
    with pytest.raises(ValueError, match=expected):
        epc.epc_postprocess_flow(tmp_path, **kwargs)


def test_extra_copies_cannot_replace_validated_inputs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from quacc.calculators.espresso.epc import epc_copy_files

    extra = tmp_path / "extra"
    (extra / "elph_dir").mkdir(parents=True)
    (extra / "elph_dir/a2Fq2r.51.1").write_text("replacement")
    with pytest.raises(ValueError, match="overlaps validated EPC inputs"):
        epc_copy_files(
            tmp_path, ["matdyn0", "elph_dir"], [{"source": extra, "filenames": "*"}]
        )


def test_shared_result_directory_is_rejected(tmp_path: Path) -> None:
    from quacc import change_settings

    with (
        change_settings({"CREATE_UNIQUE_DIR": False}),
        pytest.raises(ValueError, match="CREATE_UNIQUE_DIR"),
    ):
        epc.epc_postprocess_flow(tmp_path)


@pytest.mark.parametrize("binary", ["q2r", "matdyn"])
@pytest.mark.parametrize("data", [{"la2f": True}, {"input": {"la2f": ".false."}}])
def test_epc_boolean_inputs_are_explicit(binary: str, data: dict) -> None:
    message = "nested" if "la2f" in data else "Python boolean"
    with pytest.raises(ValueError, match=message):
        getattr(phonons, f"{binary}_job")(input_data=data)


def test_broadening_count_conflict_fails_before_qe() -> None:
    with pytest.raises(ValueError, match="el_ph_nsigma disagrees"):
        phonons.q2r_job(
            source_directory=DATA_DIR / "epc_test",
            input_data={"input": {"la2f": True, "el_ph_nsigma": 10}},
        )

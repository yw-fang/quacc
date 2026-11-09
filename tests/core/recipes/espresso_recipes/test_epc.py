"""Exercise flow customization and internal staging without launching QE."""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from ase.io.espresso import Namelist

from quacc.calculators.espresso.espresso import EspressoTemplate
from quacc.recipes.espresso import epc, phonons
from quacc.utils.files import copy_decompress_files

DATA = Path(__file__).parents[2] / "calculators/espresso/epc_data"


def test_analysis_without_plotting() -> None:
    script = """
import json, sys
sys.modules['matplotlib'] = None
from quacc.recipes.espresso.epc import epc_analysis_job
r = epc_analysis_job(
    {'epc_input_manifest': {'broadenings_Ry': [.005*i for i in range(1,11)]}},
    {'dir_name': sys.argv[1]},
)
json.dumps(r, allow_nan=False)
print(len(r['spectra']))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(DATA)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "10"


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
        (elph / f"a2Fq2r.{i}.1").write_text("fixture")
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
            for i in [51, 52]:
                (target / f"elph_dir/a2Fmatdyn.{i}").write_text("fixture")
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
            for name in ["matdyn.freq", "matdyn.dos", "a2F.dos1", "a2F.dos2"]:
                (target / name).write_bytes(
                    gzip.decompress((DATA / (name + ".gz")).read_bytes())
                )
        return {
            "dir_name": str(target),
            "parameters": Namelist({**calc_swaps, "source_path": target}),
            **additional_fields,
        }

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
    assert result["spectra"][0]["broadening_Ry"] == 0.005
    assert result["mu_stars"] == [0.13]
    assert result["settings"]["matdyn"]["source_path"] == str(tmp_path / "matdyn")
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

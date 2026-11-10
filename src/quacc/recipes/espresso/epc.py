"""QE interpolated electron-phonon postprocessing jobs and flow."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from monty.io import zopen

from quacc import flow, get_settings, job
from quacc.calculators.espresso.epc import (
    analyze_epc_directory,
    file_provenance,
    validate_epc_settings,
)
from quacc.utils.dicts import recursive_dict_merge
from quacc.wflow_tools.customizers import customize_jobs

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal

    from quacc.types import (
        EPCResultSchema,
        EspressoEPCSchema,
        RunSchema,
        SourceDirectory,
    )


@job
def epc_analysis_job(
    q2r_results: RunSchema,
    matdyn_results: RunSchema,
    mu_stars: tuple[float, ...] = (0.10,),
    imaginary_tolerance_cm1: float = 10.0,
    *,
    imaginary_mode_policy: Literal["warn", "skip_tc", "raise"] = "warn",
) -> EPCResultSchema:
    """Summarize completed EPC interpolation independently of figures.

    Parameters
    ----------
    q2r_results
        q2r_job result, including its DFPT input manifest.
    matdyn_results
        matdyn_job result with the completed spectral output directory.
    mu_stars
        Explicit Coulomb pseudopotentials for both formula-based Tc estimates.
    imaginary_tolerance_cm1
        Numerical threshold for sampled imaginary-mode diagnostics.
    imaginary_mode_policy
        "warn" retains diagnostic Tc for sampled instability or unknown stability;
        "skip_tc" suppresses Tc; "raise" raises ValueError before Tc evaluation.

    Returns
    -------
    EPCResultSchema
        JSON-serializable scientific result for every broadening, with signed
        spectra, named Tc methods, diagnostics, settings and source provenance.
        These estimates are not numerical Eliashberg solutions.
    """
    q2r_data: dict[str, Any] = dict(q2r_results)
    matdyn_data: dict[str, Any] = dict(matdyn_results)
    source = q2r_data.get("epc_input_manifest", {})
    result = analyze_epc_directory(
        matdyn_data["dir_name"],
        mu_stars=mu_stars,
        broadenings_Ry=source.get("broadenings_Ry"),
        imaginary_tolerance_cm1=imaginary_tolerance_cm1,
        imaginary_mode_policy=imaginary_mode_policy,
    )
    result["settings"] = {}
    result["input_provenance"] = {
        "DFPT": source,
        "q2r": matdyn_data.get("epc_input_manifest"),
        "written_inputs": {},
    }
    for name, data in [("q2r", q2r_data), ("matdyn", matdyn_data)]:
        try:
            provenance = file_provenance(Path(data["dir_name"]) / f"{name}.in")
            result["input_provenance"]["written_inputs"][name] = provenance
            with zopen(provenance["path"], "rt", encoding="utf-8") as stream:
                result["settings"][name] = stream.read()
        except (KeyError, OSError, ValueError) as exc:
            result["diagnostics"].append(
                {
                    "component": f"{name}_input",
                    "status": "unavailable",
                    "reason": str(exc),
                }
            )
    return result


@flow
def epc_postprocess_flow(
    source_directory: SourceDirectory,
    mu_stars: tuple[float, ...] = (0.10,),
    imaginary_tolerance_cm1: float = 10.0,
    job_params: dict[str, Any] | None = None,
    job_decorators: dict[str, Callable | None] | None = None,
    *,
    imaginary_mode_policy: Literal["warn", "skip_tc", "raise"] = "warn",
) -> EspressoEPCSchema:
    """Postprocess completed DFPT/EPC data.

    Consists of following jobs that can be modified:
    1. q2r_job: transform dynamical matrices and EPC coefficients.
    2. matdyn_job: interpolate frequencies, DOS and spectra.
    3. epc_analysis_job: integrate moments and estimate Tc per broadening.

    Parameters
    ----------
    source_directory
        Completed DFPT directory with dynamical matrices and elph_dir. QuAcc's
        matrix prefix is matdyn; another basename can be set with the q2r_job
        fildyn input. All needed files are validated and staged internally.
    mu_stars
        Coulomb pseudopotentials; results retain every broadening, not a best Tc.
    imaginary_tolerance_cm1
        Numerical instability threshold; does not establish physical stability.
    imaginary_mode_policy
        "warn" retains diagnostic Tc for sampled instability or unknown stability;
        "skip_tc" retains spectra/moments without evaluating Tc; "raise" raises
        ValueError in analysis. q2r/matdyn outputs remain saved for inspection.
    job_params
        Keyword overrides for q2r_job, matdyn_job and epc_analysis_job, following
        QuAcc's customize_jobs convention. Defaults use la2f, simple ASR and a
        32-cubed DOS interpolation grid. The heterogeneous jobs do not accept
        the shared "all" parameter override; customize each named job instead.
    job_decorators
        Per-job decorator overrides using the same three job names.

    Returns
    -------
    dict
        q2r_job and matdyn_job execution results and the independent epc_results
        scientific dictionary. Requires unique per-job result directories.
    """
    # Keep analysis-only imports usable with the numerical dependencies alone.
    from quacc.recipes.espresso.phonons import matdyn_job, q2r_job

    if not get_settings().CREATE_UNIQUE_DIR:
        raise ValueError("EPC postprocessing requires CREATE_UNIQUE_DIR=True")
    if "all" in (job_params or {}):
        raise ValueError(
            "EPC flow job_params['all'] is unsupported; customize named jobs"
        )
    validate_epc_settings(mu_stars, imaginary_tolerance_cm1, imaginary_mode_policy)

    defaults: dict[str, dict[str, Any]] = {
        "q2r_job": {"input_data": {"input": {"la2f": True, "zasr": "simple"}}},
        "matdyn_job": {
            "input_data": {
                "input": {
                    "la2f": True,
                    "asr": "simple",
                    "dos": True,
                    "nk1": 32,
                    "nk2": 32,
                    "nk3": 32,
                }
            }
        },
        "epc_analysis_job": {
            "mu_stars": mu_stars,
            "imaginary_tolerance_cm1": imaginary_tolerance_cm1,
            "imaginary_mode_policy": imaginary_mode_policy,
        },
    }
    effective = recursive_dict_merge(defaults, job_params or {})
    analysis_settings = effective["epc_analysis_job"]
    validate_epc_settings(
        analysis_settings["mu_stars"],
        analysis_settings["imaginary_tolerance_cm1"],
        analysis_settings["imaginary_mode_policy"],
    )
    for name in ("q2r_job", "matdyn_job"):
        data = effective[name].get("input_data", {})
        if "la2f" in data or data.get("input", {}).get("la2f") is not True:
            raise ValueError(f"{name} requires nested input_data['input']['la2f']=True")
    if effective["matdyn_job"]["input_data"]["input"].get("dos") is not True:
        raise ValueError("EPC postprocessing requires matdyn dos=True")
    fc_job, spectral_job, analysis_job = customize_jobs(
        {
            "q2r_job": q2r_job,
            "matdyn_job": matdyn_job,
            "epc_analysis_job": epc_analysis_job,
        },
        param_defaults=defaults,
        param_swaps=job_params,
        decorators=job_decorators,
    ).values()
    fc = fc_job(source_directory=source_directory)
    spectra = spectral_job(source_directory=fc["dir_name"])
    scientific = analysis_job(fc, spectra)
    return {"q2r_job": fc, "matdyn_job": spectra, "epc_results": scientific}

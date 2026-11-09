"""Formula-based EPC postprocessing, independent of electronic/plotting recipes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from monty.json import jsanitize

from quacc import flow, job
from quacc.calculators.espresso.epc import analyze_epc_directory
from quacc.wflow_tools.customizers import customize_jobs

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from quacc.types import EspressoEPCSchema, RunSchema, SourceDirectory


def _json_ready(value: Any) -> Any:
    # ASE Namelist is a UserDict, not a dict or an MSONable. Normalize mappings
    # before Monty handles Paths, NumPy values and other recognized metadata.
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return jsanitize(value, strict=True, enum_values=True)


@job
def epc_analysis_job(
    q2r_results: RunSchema,
    matdyn_results: RunSchema,
    mu_stars: tuple[float, ...] = (0.10,),
    imaginary_tolerance_cm1: float = 10.0,
) -> dict[str, Any]:
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

    Returns
    -------
    dict
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
    )
    result["settings"] = {
        "q2r": q2r_data.get("parameters"),
        "matdyn": matdyn_data.get("parameters"),
    }
    result["input_provenance"] = {
        "DFPT": source,
        "q2r": matdyn_data.get("epc_input_manifest"),
    }
    # QE parameters may contain Paths/NumPy scalars despite numeric arrays being
    # plain lists. Keep the scientific result portable with strict JSON encoders.
    return _json_ready(result)


@flow
def epc_postprocess_flow(
    source_directory: SourceDirectory,
    mu_stars: tuple[float, ...] = (0.10,),
    imaginary_tolerance_cm1: float = 10.0,
    job_params: dict[str, Any] | None = None,
    job_decorators: dict[str, Callable | None] | None = None,
) -> EspressoEPCSchema:
    """Postprocess completed DFPT/EPC: q2r → matdyn → scientific results.

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
    job_params
        Keyword overrides for q2r_job, matdyn_job and epc_analysis_job, following
        QuAcc's customize_jobs convention. Defaults use la2f, simple ASR and a
        32-cubed DOS interpolation grid. No PW or DFPT calculation is launched.
    job_decorators
        Per-job decorator overrides using the same three job names.

    Returns
    -------
    dict
        q2r_job and matdyn_job execution results and the independent epc_results
        scientific dictionary. No plotting, scheduler or notification is used.
    """
    from quacc.recipes.espresso.phonons import matdyn_job, q2r_job

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
        },
    }
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

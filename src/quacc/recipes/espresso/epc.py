"""Reusable electronic/phonon analysis of a completed QE EPC calculation.

These jobs use existing QuAcc runners; paths and numerical settings are supplied
by the project. Source SCF/force-constant directories are never modified.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from quacc import job
from quacc.wflow_tools.customizers import strip_decorator

_IMPORTED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _check(result, binary):
    text = (Path(result["dir_name"]) / f"{binary}.out").read_text()
    if "JOB DONE." not in text or "Error in routine" in text:
        raise RuntimeError(f"{binary}.x did not finish successfully")
    if "convergence NOT achieved" in text:
        raise RuntimeError(f"{binary}.x did not converge")
    result["execution"] = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "quantum_espresso_slurm_job_id": result.get("recovery", {}).get(
            "original_slurm_id", os.environ.get("SLURM_JOB_ID")
        ),
        "recipe_source_sha256": _IMPORTED_SOURCE_SHA256,
    }
    return result


def _path(atoms, path, npoints):
    bandpath = atoms.cell.bandpath(path=path, npoints=npoints)
    x, ticks, labels = bandpath.get_linear_kpoint_axis()
    return bandpath, {
        "path": bandpath.path,
        "kpoints_fractional": bandpath.kpts.tolist(),
        "axis_inverse_A": x.tolist(),
        "ticks": ticks.tolist(),
        "labels": labels,
    }


@job(data=True)
def electronic_bands_job(atoms, scf_dir, pw_kwargs, path=None, npoints=220):
    """Calculate bands in the original SCF cell without standardizing its basis."""
    from quacc.recipes.espresso.bands import bands_pw_job

    bandpath, info = _path(atoms, path, npoints)
    marker = Path.cwd() / "RECOVER_COMPLETED_BANDS.json"
    if marker.exists():
        from ase.io import read

        record = json.loads(marker.read_text())
        for name, digest in record["sha256"].items():
            if (
                Path(name).name != name
                or hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest
            ):
                raise ValueError("Completed-band recovery checksum mismatch")
        if json.loads(Path("bandpath.json").read_text()) != info:
            raise ValueError("Completed-band recovery path differs")
        atom = read("pw.out", format="espresso-out")
        result = {
            "dir_name": str(Path.cwd()),
            "results": dict(atom.calc.results),
            "atoms": atom,
            "bandpath": info,
            "reused_completed_QE": True,
            "recovery": record,
        }
        return _check(result, "pw")
    result = strip_decorator(bands_pw_job)(
        atoms,
        make_bandpath=False,
        copy_files=[{"source": scf_dir, "filenames": "pwscf.save"}],
        kpts=bandpath,
        **pw_kwargs,
    )
    _check(result, "pw")
    # ASE BandPath is not MSONable. Preserve its exact plain path metadata;
    # raw pw.in retains the numerical crystal_b coordinates used by QE.
    result["parameters"]["kpts"] = info
    result["bandpath"] = info
    (Path(result["dir_name"]) / "bandpath.json").write_text(json.dumps(info, indent=2))
    return result


@job(data=True)
def electronic_nscf_job(atoms, scf_dir, pw_kwargs, mesh=(24, 24, 24)):
    """Private NSCF wavefunctions for total and element-projected DOS."""
    from quacc.recipes.espresso.core import non_scf_job

    result = strip_decorator(non_scf_job)(
        atoms,
        preset=None,
        copy_files=[{"source": scf_dir, "filenames": "pwscf.save"}],
        kpts=tuple(mesh),
        koffset=(0, 0, 0),
        **pw_kwargs,
    )
    return _check(result, "pw")


@job(data=True)
def electronic_dos_job(nscf_dir, input_data):
    """Run native dos.x using a private copy of the NSCF save directory."""
    from quacc.recipes.espresso.dos import dos_job

    result = strip_decorator(dos_job)(
        copy_files=[{"source": nscf_dir, "filenames": "pwscf.save"}],
        input_data=input_data,
    )
    return _check(result, "dos")


@job(data=True)
def element_dos_job(nscf_dir, input_data):
    """Run native projwfc.x; retain individual atom/orbital PDOS files."""
    from quacc.recipes.espresso.dos import projwfc_job

    result = strip_decorator(projwfc_job)(
        copy_files=[{"source": nscf_dir, "filenames": "pwscf.save"}],
        input_data=input_data,
    )
    return _check(result, "projwfc")


@job(data=True)
def phonon_bands_job(atoms, force_constants_dir, path=None, npoints=220, asr="simple"):
    """Interpolate existing force constants on an explicitly saved fractional path."""
    from quacc.recipes.espresso.phonons import matdyn_job

    bandpath, info = _path(atoms, path, npoints)
    cards = [str(len(bandpath.kpts))] + [
        " ".join(f"{v:.12f}" for v in q) for q in bandpath.kpts
    ]
    result = strip_decorator(matdyn_job)(
        copy_files=[{"source": force_constants_dir, "filenames": "q2r.fc*"}],
        input_data={
            "input": {
                "asr": asr,
                "dos": False,
                "la2f": False,
                "q_in_cryst_coord": True,
                "q_in_band_form": False,
            }
        },
        additional_cards=cards,
    )
    _check(result, "matdyn")
    result["bandpath"] = info
    (Path(result["dir_name"]) / "bandpath.json").write_text(json.dumps(info, indent=2))
    return result


@job(data=True)
def epc_report_job(bands, nscf, dos, pdos, phonons, spectrum, metadata, output_dir):
    """Durably save scientific artifacts; notification delivery is a separate action."""
    from quacc.schemas.epc import save_epc_report

    return save_epc_report(
        bands, nscf, dos, pdos, phonons, spectrum, metadata, output_dir
    )


def epc_analysis_flow(
    atoms,
    scf_dir,
    force_constants_dir,
    spectrum,
    metadata,
    output_dir,
    pw_kwargs,
    dos_input,
    pdos_input,
    mesh=(24, 24, 24),
    path=None,
    npoints=220,
):
    """Build a Jobflow graph from reusable QuAcc jobs, with no scheduler settings.

    This is the analysis continuation of a completed relaxation/SCF/DFPT flow.
    It deliberately reuses the chosen force constants and spectrum.
    """
    from jobflow import Flow

    bands = electronic_bands_job(atoms, scf_dir, pw_kwargs, path, npoints)
    nscf = electronic_nscf_job(atoms, scf_dir, pw_kwargs, mesh)
    dos = electronic_dos_job(nscf.output["dir_name"], dos_input)
    pdos = element_dos_job(nscf.output["dir_name"], pdos_input)
    ph = phonon_bands_job(atoms, force_constants_dir, path, npoints)
    report = epc_report_job(
        bands.output,
        nscf.output,
        dos.output,
        pdos.output,
        ph.output,
        spectrum,
        metadata,
        output_dir,
    )
    for j, label in zip(
        [bands, nscf, dos, pdos, ph, report],
        ["bands", "DOS NSCF", "total DOS", "element DOS", "phonon path", "report"],
        strict=True,
    ):
        j.name = f"{metadata['material_id']}/{label}"
        j.metadata.update(
            material_id=metadata["material_id"], run_id=metadata["run_id"]
        )
    return Flow(
        [bands, nscf, dos, pdos, ph, report],
        name=metadata["run_id"],
        output=report.output,
    )


@job(data=True)
def ground_state_job(atoms, pw_kwargs, mesh, relax=False, dense_dir=None):
    """Native relaxation/SCF with a private, hash-verified dense EPC eigenvalue file."""
    from quacc.recipes.espresso.core import relax_job, static_job
    from quacc.schemas.epc import sha

    copies = (
        None
        if dense_dir is None
        else [{"source": dense_dir, "filenames": "pwscf.a2Fsave"}]
    )
    extra = {"relax_cell": True} if relax else {}
    result = strip_decorator(relax_job if relax else static_job)(
        atoms,
        preset=None,
        kpts=tuple(mesh),
        koffset=(0, 0, 0),
        copy_files=copies,
        **pw_kwargs,
        **extra,
    )
    _check(result, "pw")
    text = (Path(result["dir_name"]) / "pw.out").read_text()
    if "convergence has been achieved" not in text:
        raise RuntimeError("SCF convergence was not achieved")
    if relax and "bfgs converged" not in text.lower():
        raise RuntimeError("Cell optimization was not converged")
    if dense_dir and sha(Path(dense_dir) / "pwscf.a2Fsave") != sha(
        Path(result["dir_name"]) / "pwscf.a2Fsave"
    ):
        raise RuntimeError("Dense EPC eigenvalue transfer failed")
    return result


@job(data=True)
def epc_phonon_job(scf_dir, input_data):
    """DFPT/EPC on private SCF files, leaving upstream charge/eigenvalues intact."""
    from quacc.recipes.espresso.phonons import phonon_job

    result = strip_decorator(phonon_job)(
        copy_files=[{"source": scf_dir, "filenames": ["pwscf.save", "pwscf.a2Fsave"]}],
        input_data=input_data,
    )
    _check(result, "ph")
    d = Path(result["dir_name"])
    count = int((d / "matdyn0").read_text().splitlines()[1])
    n = input_data["inputph"]["el_ph_nsigma"]
    if len(list((d / "elph_dir").glob("a2Fq2r.*"))) != count * n:
        raise RuntimeError("Incomplete DFPT/EPC grid")
    if "No convergence has been achieved" in (d / "ph.out").read_text():
        raise RuntimeError("DFPT perturbation did not converge")
    return result


@job(data=True)
def epc_force_constants_job(ph_dir, asr="simple"):
    """Native q2r including the EPC directory required by QE's la2F path."""
    from quacc.recipes.espresso.phonons import q2r_job

    result = strip_decorator(q2r_job)(
        copy_files=[{"source": ph_dir, "filenames": ["matdyn*", "elph_dir"]}],
        input_data={"input": {"la2f": True, "zasr": asr}},
    )
    return _check(result, "q2r")


@job(data=True)
def epc_spectrum_job(fc_dir, input_data, nsigma):
    """Native matdyn including the EPC force constants, retaining every broadening."""
    from quacc.recipes.espresso.phonons import matdyn_job

    result = strip_decorator(matdyn_job)(
        copy_files=[{"source": fc_dir, "filenames": ["q2r.fc*", "elph_dir"]}],
        input_data=input_data,
    )
    _check(result, "matdyn")
    if not all(
        (Path(result["dir_name"]) / f"a2F.dos{i}").exists()
        for i in range(1, nsigma + 1)
    ):
        raise RuntimeError("Incomplete alpha2F broadening series")
    return result


def superconductivity_flow(
    atoms,
    metadata,
    output_dir,
    pw_kwargs,
    coarse_mesh,
    fine_mesh,
    ph_input,
    spectral_input,
    dos_input,
    pdos_input,
    spectrum_index=5,
    dos_mesh=(24, 24, 24),
    path=None,
    npoints=220,
):
    """Optimization-to-Tc/report Jobflow built from native QuAcc recipes.

    Numerical cutoffs, smearing, q grids, convergence settings and the displayed
    spectrum selection belong to the project. No cluster paths or Discord code
    are embedded here. Existing-result continuation uses epc_analysis_flow.
    """
    from copy import deepcopy

    from jobflow import Flow

    r1 = ground_state_job(atoms, pw_kwargs, coarse_mesh, relax=True)
    r2 = ground_state_job(r1.output["atoms"], pw_kwargs, coarse_mesh, relax=True)
    dense = deepcopy(pw_kwargs)
    dense["input_data"]["system"]["la2f"] = True
    fine = ground_state_job(r2.output["atoms"], dense, fine_mesh)
    coarse = ground_state_job(
        r2.output["atoms"], pw_kwargs, coarse_mesh, dense_dir=fine.output["dir_name"]
    )
    ph = epc_phonon_job(coarse.output["dir_name"], ph_input)
    fc = epc_force_constants_job(ph.output["dir_name"])
    nsigma = ph_input["inputph"]["el_ph_nsigma"]
    if not 1 <= spectrum_index <= nsigma:
        raise ValueError("Selected spectrum index outside broadening series")
    spec = epc_spectrum_job(fc.output["dir_name"], spectral_input, nsigma)
    inp = ph_input["inputph"]
    spectrum = {
        "directory": spec.output["dir_name"],
        "filename": f"a2F.dos{spectrum_index}",
        "qmesh": [inp[f"nq{i}"] for i in (1, 2, 3)],
        "broadening_Ry": inp["el_ph_sigma"] * spectrum_index,
        "broadening_index": spectrum_index,
    }
    metadata = dict(metadata, scf_fermi_eV=fine.output["results"]["fermi_level"])
    # Absolute energy bounds are best supplied by the project for a continuation;
    # a fresh run can omit Emin/Emax and use the program's eigenvalue limits.
    analysis = epc_analysis_flow(
        r2.output["atoms"],
        fine.output["dir_name"],
        fc.output["dir_name"],
        spectrum,
        metadata,
        output_dir,
        pw_kwargs,
        dos_input,
        pdos_input,
        dos_mesh,
        path,
        npoints,
    )
    upstream = [r1, r2, fine, coarse, ph, fc, spec]
    for j, label in zip(
        upstream,
        ["relax 1", "relax 2", "fine SCF", "coarse SCF", "DFPT EPC", "q2r", "alpha2F"],
        strict=True,
    ):
        j.name = f"{metadata['material_id']}/{label}"
        j.metadata.update(
            material_id=metadata["material_id"], run_id=metadata["run_id"]
        )
    return Flow([*upstream, analysis], name=metadata["run_id"], output=analysis.output)

"""Parsing and formula-based analysis of Quantum ESPRESSO EPC output.

The frequency moments and Tc expressions follow hyllios/utils
``autoph/allen_dynes.py`` (commit 4e86e82). These are modified McMillan and
Allen-Dynes estimates, not numerical solutions of the Eliashberg equations.
Signed spectra are retained. No plotting or workflow engine is imported here.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from monty.io import zopen
from monty.os.path import zpath
from scipy.constants import physical_constants
from scipy.integrate import simpson

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import NDArray

_NUMBER = r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?"
_EV_K = 1 / physical_constants["Boltzmann constant in eV/K"][0]
_RY_EV = physical_constants["Rydberg constant times hc in eV"][0]
_EV_CM = physical_constants["electron volt-inverse meter relationship"][0] / 100
_UNITS = {"Ry": _RY_EV, "eV": 1.0, "meV": 0.001, "cm^-1": 1 / _EV_CM}


def _text(path: str | Path) -> str:
    with zopen(zpath(Path(path)), "rt", encoding="utf-8") as stream:
        return stream.read()


def file_provenance(path: str | Path) -> dict[str, Any]:
    """Identify a source file without modifying or decompressing it in place.

    Parameters
    ----------
    path
        Source filename, with optional gzip compression inferred by zpath.

    Returns
    -------
    dict
        Resolved filename, SHA256 of stored bytes, size and hash convention.
    """
    actual = Path(zpath(Path(path))).resolve()
    return {
        "path": str(actual),
        "sha256": hashlib.sha256(actual.read_bytes()).hexdigest(),
        "bytes": actual.stat().st_size,
        "hash_scope": "stored bytes",
    }


def _table(path: str | Path) -> tuple[NDArray[np.float64], str]:
    text = _text(path)
    rows = []
    for line in text.splitlines():
        words = line.split()
        if not words or words[0].startswith(("#", "!")):
            continue
        try:
            float(words[0].replace("D", "E").replace("d", "e"))
        except ValueError:
            continue  # QE textual footer/header; numeric rows must be complete.
        try:
            rows.append([float(w.replace("D", "E").replace("d", "e")) for w in words])
        except ValueError as exc:
            raise ValueError(f"Malformed numeric row in {path}") from exc
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError(f"Missing or inconsistent numeric data in {path}")
    data = np.asarray(rows, dtype=float)
    if data.shape[1] < 2 or not np.isfinite(data).all():
        raise ValueError(f"Invalid/nonfinite numeric data in {path}")
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError(f"Non-increasing frequency grid in {path}")
    return data, text


def _unit(text: str, supplied: str | None) -> str:
    header = " ".join(
        line for line in text.splitlines() if line.lstrip().startswith("#")
    )
    detected = next(
        (
            unit
            for pattern, unit in [
                (r"(?i:Rydberg)|\bRy\b", "Ry"),
                (r"cm\^?-1", "cm^-1"),
                (r"\bmeV\b", "meV"),
                (r"\beV\b", "eV"),
            ]
            if re.search(pattern, header)
        ),
        None,
    )
    if supplied is not None and supplied not in _UNITS:
        raise ValueError(f"Unsupported frequency unit: {supplied}")
    if detected is not None and supplied is not None and detected != supplied:
        raise ValueError(
            f"Frequency unit conflicts with header: {detected} vs {supplied}"
        )
    if supplied is None and detected is None:
        raise ValueError("Frequency unit unavailable; supply frequency_unit explicitly")
    return supplied or str(detected)


def read_a2f(path: str | Path, frequency_unit: str | None = None) -> dict[str, Any]:
    """Read all signed total/mode columns of an a2F.dos file, including endpoints.

    Parameters
    ----------
    path
        File to read, optionally gzip compressed.
    frequency_unit
        Ry, eV, meV or cm^-1. Inferred from a QE comment header if omitted;
        explicit/header disagreements raise ValueError. Spectral amplitudes
        retain their native QE normalization without rescaling.

    Returns
    -------
    dict
        JSON-serializable frequencies, signed total and mode spectra, units,
        optional QE footer lambda, and source-file provenance.
    """
    data, text = _table(path)
    unit = _unit(text, frequency_unit)
    match = re.search(r"lambda\s*=\s*(" + _NUMBER + ")", text, re.IGNORECASE)
    return {
        "frequency": data[:, 0].tolist(),
        "frequency_unit": unit,
        "alpha2F": data[:, 1].tolist(),
        "alpha2F_modes": data[:, 2:].tolist(),
        "mode_indices": list(range(1, data.shape[1] - 1)),
        "spectral_normalization": "QE native; signed values and amplitudes unmodified",
        "qe_footer_lambda": float(match[1].replace("D", "E").replace("d", "e"))
        if match
        else None,
        "source": file_provenance(path),
    }


def read_matdyn_frequencies(
    path: str | Path, frequency_unit: str = "cm^-1"
) -> dict[str, Any]:
    """Read matdyn flfrq data, with three-coordinate or weighted q-point lines.

    Parameters
    ----------
    path
        matdyn frequency file, optionally compressed.
    frequency_unit
        Native QE flfrq frequencies are cm^-1. An explicit alternative unit
        declaration is retained; no frequency conversion is performed here.

    Returns
    -------
    dict
        Q coordinates, optional weights, all mode frequencies, units and source.
        An overflowed nks header is recorded, with the count inferred from data.
    """
    if frequency_unit not in _UNITS:
        raise ValueError(f"Unsupported frequency unit: {frequency_unit}")
    lines = [line.strip() for line in _text(path).splitlines() if line.strip()]
    if not lines:
        raise ValueError("Missing nbnd in empty matdyn frequency file")
    nbnd = re.search(r"nbnd\s*=\s*(\d+)", lines[0])
    nks = re.search(r"nks\s*=\s*(\d+)", lines[0])
    if not nbnd or int(nbnd[1]) < 1:
        raise ValueError("Missing/invalid nbnd in matdyn frequency header")
    modes = int(nbnd[1])
    points, weights, frequencies = [], [], []
    i = 1
    while i < len(lines):
        point = [float(x.replace("D", "E").replace("d", "e")) for x in lines[i].split()]
        if len(point) not in (3, 4):
            raise ValueError("Expected three q coordinates and an optional weight")
        points.append(point[:3])
        weights.append(point[3] if len(point) == 4 else None)
        i += 1
        values: list[float] = []
        while len(values) < modes and i < len(lines):
            values.extend(
                float(x.replace("D", "E").replace("d", "e")) for x in lines[i].split()
            )
            i += 1
        if len(values) != modes:
            raise ValueError("Incomplete/malformed matdyn frequency block")
        frequencies.append(values)
    if not frequencies or not np.isfinite(frequencies).all():
        raise ValueError("Empty/nonfinite matdyn frequencies")
    if not np.isfinite(points).all() or any(
        w is not None and not np.isfinite(w) for w in weights
    ):
        raise ValueError("Nonfinite q coordinates/weights")
    if nks and int(nks[1]) != len(points):
        raise ValueError("matdyn nks disagrees with frequency blocks")
    return {
        "qpoints": points,
        "qpoint_coordinate_system": "QE flfrq Cartesian coordinates",
        "weights": weights,
        "frequencies": frequencies,
        "frequency_unit": frequency_unit,
        "number_of_modes": modes,
        "number_of_qpoints": len(points),
        "declared_nks": int(nks[1]) if nks else None,
        "diagnostics": [] if nks else ["nks_header_unavailable_or_overflowed"],
        "source": file_provenance(path),
    }


def read_phonon_dos(
    path: str | Path, frequency_unit: str | None = None
) -> dict[str, Any]:
    """Read signed phonon total and atom-projected DOS, with explicit native units.

    Parameters
    ----------
    path
        QE fldos file, optionally compressed.
    frequency_unit
        Inferred from the header when omitted. DOS is per native frequency
        unit per cell; projected columns are atom ordered, not mode resolved.

    Returns
    -------
    dict
        Native frequency/DOS arrays, projected columns, units and provenance.
    """
    data, text = _table(path)
    unit = _unit(text, frequency_unit)
    return {
        "frequency": data[:, 0].tolist(),
        "frequency_unit": unit,
        "total_dos": data[:, 1].tolist(),
        "atom_projected_dos": data[:, 2:].tolist(),
        "dos_unit": f"states per ({unit}) per cell",
        "source": file_provenance(path),
    }


def analyze_a2f(
    spectrum: dict[str, Any], mu_stars: Sequence[float] = (0.10,)
) -> dict[str, Any]:
    """Integrate signed spectral moments and evaluate named Tc formula estimates.

    Parameters
    ----------
    spectrum
        Output of read_a2f. All signed rows remain in the returned result;
        integrations use strictly positive frequencies only.
    mu_stars
        Finite nonnegative Coulomb pseudopotentials. Invalid formula domains
        produce null Tc with a reason, never a physical zero sentinel.

    Returns
    -------
    dict
        Spectrum with moments in K, signed-weight diagnostics, mode lambdas,
        and method/mu-specific Tc records. No numerical Eliashberg solver is used.
    """
    mus = [float(mu) for mu in mu_stars]
    if not mus or any(not np.isfinite(mu) or mu < 0 for mu in mus):
        raise ValueError("mu_stars must be finite nonnegative values")
    w_all = np.asarray(spectrum["frequency"], dtype=float)
    a_all = np.asarray(spectrum["alpha2F"], dtype=float)
    positive = w_all > 0
    w, a = w_all[positive], a_all[positive]
    flags = []
    if np.any(a_all < 0):
        flags.append("negative_spectral_weight")
    if np.any((~positive) & (a_all != 0)):
        flags.append("nonpositive_frequency_weight_excluded_from_moments")
    result = dict(
        spectrum,
        **{"lambda": None},
        omega_log_K=None,
        omega_2_K=None,
        Tc=[],
        diagnostics=flags,
    )
    if len(w) < 3:
        reason = "insufficient_positive_frequency_points"
        result["diagnostics"].append(reason)
    else:
        lam = float(2 * simpson(a / w, x=w))
        lp = float(2 * simpson(np.maximum(a, 0) / w, x=w))
        ln = float(2 * simpson(np.minimum(a, 0) / w, x=w))
        result.update(
            {
                "lambda": lam if np.isfinite(lam) else None,
                "lambda_positive_part": lp if np.isfinite(lp) else None,
                "lambda_negative_part": ln if np.isfinite(ln) else None,
                "negative_lambda_fraction": abs(ln) / lp if lp > 0 else None,
            }
        )
        footer = spectrum["qe_footer_lambda"]
        result["qe_footer_lambda_difference"] = (
            lam - footer if footer is not None and np.isfinite(lam) else None
        )
        mode_values = np.asarray(spectrum["alpha2F_modes"], dtype=float)
        result["mode_lambda"] = (
            2 * simpson(mode_values[positive] / w[:, None], x=w, axis=0)
        ).tolist()
        result["mode_sum_max_abs_error"] = (
            float(np.max(np.abs(mode_values.sum(axis=1) - a_all)))
            if mode_values.shape[1]
            else None
        )
        reason = "nonpositive_or_nonfinite_lambda"
        if lam > 0 and np.isfinite(lam):
            logw = float(2 * simpson(a / w * np.log(w), x=w) / lam)
            w2sq = float(2 * simpson(a * w, x=w) / lam)
            with np.errstate(over="ignore", invalid="ignore"):
                wlog = float(np.exp(logw) * _UNITS[spectrum["frequency_unit"]] * _EV_K)
                w2 = float(np.sqrt(w2sq) * _UNITS[spectrum["frequency_unit"]] * _EV_K)
            reason = "invalid_frequency_moments"
            if wlog > 0 and w2 > 0 and np.isfinite([wlog, w2]).all():
                result.update(omega_log_K=wlog, omega_2_K=w2)
                reason = ""
        if reason:
            result["diagnostics"].append(reason)
    for mu in mus:
        for method in ["modified_mcmillan_omega_log", "allen_dynes_f1_f2"]:
            entry = {
                "method": method,
                "mu_star": mu,
                "Tc_K": None,
                "status": "unavailable",
                "reason": reason or None,
            }
            if not reason:
                lam = result["lambda"]
                denom = lam - mu * (1 + 0.62 * lam)
                if denom <= 0:
                    entry["reason"] = "nonpositive_formula_denominator"
                else:
                    logtc = (
                        np.log(result["omega_log_K"] / 1.2) - 1.04 * (1 + lam) / denom
                    )
                    if method == "allen_dynes_f1_f2":
                        f1 = (1 + (lam / (2.46 * (1 + 3.8 * mu))) ** 1.5) ** (1 / 3)
                        ratio = result["omega_2_K"] / result["omega_log_K"]
                        f2 = 1 + lam**2 * (ratio - 1) / (
                            lam**2 + (1.82 * (1 + 6.3 * mu) * ratio) ** 2
                        )
                        entry.update(f1=f1, f2=f2)
                        logtc += np.log(f1 * f2)
                    if not np.isfinite(logtc) or not np.log(
                        np.finfo(float).tiny
                    ) < logtc < np.log(np.finfo(float).max):
                        entry["reason"] = "Tc_numerically_unrepresentable"
                    else:
                        entry.update(
                            Tc_K=float(np.exp(logtc)), status="estimated", reason=None
                        )
            result["Tc"].append(entry)
    result["status"] = (
        "invalid" if reason else "analyzed_with_warnings" if flags else "analyzed"
    )
    return result


def epc_file_manifest(
    directory: str | Path, stage: str, dynamical_prefix: str = "matdyn"
) -> dict[str, Any]:
    """Validate and identify automatically staged q2r/matdyn inputs.

    Parameters
    ----------
    directory
        Completed source directory; gzip files are read without modifying it.
    stage
        q2r (DFPT matrices/EPC grid) or matdyn (force constants/EPC transforms).
    dynamical_prefix
        Basename prefix for DFPT dynamical matrices. Used only for q2r.

    Returns
    -------
    dict
        Filenames to copy, input hashes, grid dimensions, and the ordered Gaussian
        broadenings in Ry read from QE metadata (not guessed from filenames).
    """
    directory = Path(directory).expanduser().resolve()
    if stage not in ("q2r", "matdyn"):
        raise ValueError("stage must be q2r or matdyn")
    if Path(dynamical_prefix).name != dynamical_prefix:
        raise ValueError("Dynamical matrix prefix must be a basename")
    elph = directory / "elph_dir"
    if not elph.is_dir():
        raise FileNotFoundError(f"Missing EPC directory: {elph}")
    metadata = sorted(elph.glob("elph.inp_lambda.*"))
    if not metadata:
        raise FileNotFoundError("Missing elph.inp_lambda broadening metadata")
    broadening_sets = [
        re.findall(r"Gaussian Broadening:\s*(" + _NUMBER + r")\s*Ry", _text(p))
        for p in metadata
    ]
    if not broadening_sets[0] or any(
        values != broadening_sets[0] for values in broadening_sets
    ):
        raise ValueError("Inconsistent/missing EPC broadening metadata")
    broadenings = [float(value.replace("D", "E")) for value in broadening_sets[0]]
    if any(not np.isfinite(value) or value <= 0 for value in broadenings):
        raise ValueError("EPC broadenings must be finite positive values")
    files = list(metadata)
    if stage == "q2r":
        lines = _text(directory / f"{dynamical_prefix}0").splitlines()
        grid = [int(i) for i in lines[0].split()]
        nq = int(lines[1])
        if len(grid) != 3 or any(n < 1 for n in grid) or nq < 1:
            raise ValueError("Invalid DFPT q-grid header")
        metadata_indices = {
            int(p.name.removesuffix(".gz").split(".")[-1]) for p in metadata
        }
        if metadata_indices != set(range(1, nq + 1)) or len(metadata) != nq:
            raise ValueError("Incomplete/ambiguous EPC q-point metadata")
        files.extend(
            Path(zpath(directory / f"{dynamical_prefix}{i}")) for i in range(nq + 1)
        )
        pairs = {}
        for p in elph.glob("a2Fq2r.*"):
            match = re.fullmatch(r"a2Fq2r\.(\d+)\.(\d+)(?:\.gz)?", p.name)
            if match:
                key = int(match[1]), int(match[2])
                if key in pairs:
                    raise ValueError(f"Ambiguous duplicate EPC input: {p.name}")
                pairs[key] = p
        codes = sorted({code for code, _ in pairs})
        expected_pairs = {(code, q) for code in codes for q in range(1, nq + 1)}
        if len(codes) != len(broadenings) or pairs.keys() != expected_pairs:
            raise ValueError("Incomplete DFPT EPC q-point/broadening series")
        files.extend(pairs.values())
        names = [f"{dynamical_prefix}[0-9]*", "elph_dir"]
    else:
        grid = None
        files.append(Path(zpath(directory / "q2r.fc")))
        transformed = [
            p
            for p in elph.glob("a2Fmatdyn.*")
            if re.fullmatch(r"a2Fmatdyn\.\d+(?:\.gz)?", p.name)
        ]
        if len(transformed) != len(broadenings):
            raise ValueError("Incomplete q2r EPC force-constant broadening series")
        codes = [int(p.name.removesuffix(".gz").split(".")[-1]) for p in transformed]
        if len(set(codes)) != len(codes):
            raise ValueError("Ambiguous duplicate q2r EPC force constants")
        files.extend(transformed)
        names = ["q2r.fc*", "elph_dir"]
    return {
        "source_directory": str(directory),
        "filenames": names,
        "qmesh": grid,
        "broadenings_Ry": broadenings,
        "files": [file_provenance(p) for p in files],
    }


def analyze_epc_directory(
    directory: str | Path,
    mu_stars: Sequence[float] = (0.10,),
    broadenings_Ry: Sequence[float] | None = None,
    imaginary_tolerance_cm1: float = 10.0,
) -> dict[str, Any]:
    """Create a JSON scientific result from saved matdyn output, without QE.

    Parameters
    ----------
    directory
        Directory with a2F.dosN, matdyn.freq and matdyn.dos (gzip supported).
    mu_stars
        Coulomb pseudopotentials for the two named formula estimates.
    broadenings_Ry
        Expected ordered broadenings from the DFPT metadata. Omit only when
        unknown; then trailing missing spectra cannot be detected.
    imaginary_tolerance_cm1
        Positive tolerance for flagging sampled imaginary frequencies. This is
        a numerical diagnostic, not a proof of stability or convergence.

    Returns
    -------
    dict
        All spectra/units/moments/Tc records and source diagnostics. Missing or
        malformed components are distinguished from physical zero by statuses
        and null quantities. No display spectrum is selected automatically.
    """
    directory = Path(directory).expanduser().resolve()
    if not np.isfinite(imaginary_tolerance_cm1) or imaginary_tolerance_cm1 < 0:
        raise ValueError("Imaginary-frequency tolerance must be finite and nonnegative")
    if not mu_stars or any(not np.isfinite(mu) or mu < 0 for mu in mu_stars):
        raise ValueError("mu_stars must be finite nonnegative values")
    if broadenings_Ry is not None and (
        not broadenings_Ry or any(not np.isfinite(b) or b <= 0 for b in broadenings_Ry)
    ):
        raise ValueError("Broadenings must be finite positive values")
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "analyzed",
        "directory": str(directory),
        "method_scope": "isotropic formula-based Tc estimates, not numerical Eliashberg solutions",
        "mu_stars": list(mu_stars),
        "spectra": [],
        "diagnostics": [],
        "frequencies": None,
        "phonon_dos": None,
        "instability": None,
        "imaginary_tolerance_cm1": imaginary_tolerance_cm1,
    }
    for key, name, parser in [
        ("frequencies", "matdyn.freq", read_matdyn_frequencies),
        ("phonon_dos", "matdyn.dos", read_phonon_dos),
    ]:
        try:
            result[key] = parser(directory / name)
        except (OSError, ValueError, IndexError) as exc:
            result["diagnostics"].append(
                {
                    "component": key,
                    "status": "unavailable_or_invalid",
                    "reason": str(exc),
                }
            )
    available = {}
    for path in directory.glob("a2F.dos*"):
        match = re.fullmatch(r"a2F\.dos([1-9]\d*)(?:\.gz)?", path.name)
        if match:
            index = int(match[1])
            if index in available:
                raise ValueError(f"Ambiguous duplicate spectrum index {index}")
            available[index] = path
    qe_qsum = []
    try:
        text = _text(directory / "lambda")
    except FileNotFoundError:
        text = ""
    if text:
        for line in text.splitlines():
            match = re.search(
                r"Broadening\s+("
                + _NUMBER
                + r")\s+lambda\s+("
                + _NUMBER
                + r")\s+dos\(Ef\)\s+("
                + _NUMBER
                + r")\s+omega_ln\s*\[K\]\s+("
                + _NUMBER
                + ")",
                line,
            )
            if match:
                values = [float(x.replace("D", "E")) for x in match.groups()]
                qe_qsum.append(
                    dict(
                        zip(
                            [
                                "broadening_Ry",
                                "lambda",
                                "DOS_EF_QE_native",
                                "omega_log_K",
                            ],
                            values,
                            strict=True,
                        )
                    )
                )
        result["qe_qsum_source"] = file_provenance(directory / "lambda")
    result["qe_qsum"] = qe_qsum
    if broadenings_Ry is None:
        result["diagnostics"].append(
            {"component": "broadening_series", "status": "expected_count_unknown"}
        )
    expected = (
        set(range(1, len(broadenings_Ry) + 1))
        if broadenings_Ry is not None
        else set(range(1, max(available, default=0) + 1))
    )
    if available.keys() - expected:
        result["diagnostics"].append(
            {
                "component": "broadening_series",
                "status": "unexpected_spectrum_indices",
                "indices": sorted(available.keys() - expected),
            }
        )
    for index in sorted(expected | available.keys()):
        record: dict[str, Any] = {
            "index": index,
            "broadening_Ry": broadenings_Ry[index - 1]
            if broadenings_Ry is not None and index <= len(broadenings_Ry)
            else None,
        }
        if index not in available:
            record.update(
                status="unavailable",
                reason="missing_spectrum",
                **{"lambda": None},
                omega_log_K=None,
                Tc=[],
            )
        else:
            try:
                record.update(analyze_a2f(read_a2f(available[index]), mu_stars))
            except (OSError, ValueError) as exc:
                record.update(
                    status="invalid",
                    reason=str(exc),
                    **{"lambda": None},
                    omega_log_K=None,
                    Tc=[],
                )
        result["spectra"].append(record)
        if index <= len(qe_qsum):
            record["qe_qsum"] = qe_qsum[index - 1]
            if record["broadening_Ry"] is None:
                record["broadening_Ry"] = qe_qsum[index - 1]["broadening_Ry"]
            elif not np.isclose(
                record["broadening_Ry"],
                qe_qsum[index - 1]["broadening_Ry"],
                rtol=1e-5,
                atol=1e-8,
            ):
                record.update(status="invalid", reason="broadening_identity_conflict")
    frequencies = result["frequencies"]
    if frequencies is not None:
        mode_frequencies = np.asarray(frequencies["frequencies"])
        result["instability"] = {
            "minimum_frequency_cm1": float(mode_frequencies.min()),
            "negative_frequency_count": int(np.sum(mode_frequencies < 0)),
            "below_tolerance_count": int(
                np.sum(mode_frequencies < -imaginary_tolerance_cm1)
            ),
            "sampled_imaginary_modes": bool(
                np.any(mode_frequencies < -imaginary_tolerance_cm1)
            ),
        }
    if result["phonon_dos"] is not None:
        dos = result["phonon_dos"]
        integral = float(simpson(dos["total_dos"], x=dos["frequency"]))
        dos["diagnostics"] = {
            "integrated_states": integral,
            "negative_bins": int(np.sum(np.asarray(dos["total_dos"]) < 0)),
            "expected_modes": frequencies["number_of_modes"]
            if frequencies is not None
            else None,
        }
        if len(dos["frequency"]) < 2:
            dos["diagnostics"].update(
                integrated_states=None, status="insufficient_grid_points"
            )
    if (
        not result["spectra"]
        or any(r["status"] in ("invalid", "unavailable") for r in result["spectra"])
        or result["frequencies"] is None
        or result["phonon_dos"] is None
    ):
        result["status"] = "incomplete_or_invalid"
    elif result["instability"]["sampled_imaginary_modes"]:
        result["status"] = "sampled_instability"
    elif result["diagnostics"] or any(r["diagnostics"] for r in result["spectra"]):
        result["status"] = "analyzed_with_warnings"
    return result

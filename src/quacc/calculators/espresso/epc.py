"""Parsers and isotropic formula-based analysis of QE interpolated EPC data.

Spectral moments summarize the Eliashberg function alpha2F: lambda is the
dimensionless coupling strength, omega_log is the coupling-weighted logarithmic
average frequency, and omega_2 is the square root of the weighted second moment.
The two characteristic frequencies are reported in kelvin.

Tc estimates use the modified McMillan and Allen-Dynes expressions;
see Gao et al., Nat Commun 16, 8253 (2025), doi:10.1038/s41467-025-63702-w,
for an implementation of these methods. Simpson integration differs from QE's
bin-centred midpoint footer. These estimates are not numerical Eliashberg solutions.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from monty.io import zopen
from scipy.constants import physical_constants
from scipy.integrate import simpson

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import NDArray

    from quacc.types import (
        A2FSpectrumSchema,
        CopyFiles,
        EPCResultSchema,
        TcEstimateSchema,
    )

_NUMBER = r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?"
_TOKEN = r"(?:" + _NUMBER + r"|[-+]?(?:nan|inf(?:inity)?))"
_EV_K = 1 / physical_constants["Boltzmann constant in eV/K"][0]
_RY_EV = physical_constants["Rydberg constant times hc in eV"][0]
_EV_CM = physical_constants["electron volt-inverse meter relationship"][0] / 100
_UNITS = {"Ry": _RY_EV, "eV": 1.0, "meV": 0.001, "cm^-1": 1 / _EV_CM}
_NEGATIVE_FRACTION_TOLERANCE = 1e-6


def _float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _resolve_file(path: str | Path) -> Path:
    path = Path(path).expanduser()
    plain = path.with_name(path.name.removesuffix(".gz"))
    compressed = plain.with_name(plain.name + ".gz")
    if plain.exists() and compressed.exists():
        raise ValueError(f"Ambiguous plain/gzip files: {plain}")
    actual = plain if plain.exists() else compressed
    if actual.is_symlink():
        raise ValueError(f"Symbolic-link EPC input is unsupported: {actual}")
    if not actual.is_file():
        raise FileNotFoundError(f"Missing EPC file: {path}")
    return actual.resolve()


def _text(path: str | Path) -> str:
    with zopen(_resolve_file(path), "rt", encoding="utf-8") as stream:
        return stream.read()


def file_provenance(path: str | Path) -> dict[str, Any]:
    """Identify stored and uncompressed source bytes without modifying the file.

    Parameters
    ----------
    path
        Filename with optional gzip compression.

    Returns
    -------
    dict
        Absolute path, compression, byte count, stored SHA256 and reproducible
        uncompressed SHA256. Ambiguous plain/gzip pairs are rejected.
    """
    actual = _resolve_file(path)
    with zopen(actual, "rb") as stream:
        uncompressed = stream.read()
    return {
        "path": str(actual),
        "sha256": hashlib.sha256(actual.read_bytes()).hexdigest(),
        "bytes": actual.stat().st_size,
        "hash_scope": "stored bytes",
        "compression": "gzip" if actual.suffix == ".gz" else None,
        "uncompressed_sha256": hashlib.sha256(uncompressed).hexdigest(),
    }


def _table(path: str | Path) -> tuple[NDArray[np.float64], str, str]:
    text = _text(path)
    rows: list[list[float]] = []
    header: list[str] = []
    ended = False
    for line in text.splitlines():
        words = line.split()
        if not words:
            continue
        try:
            _float(words[0])
        except ValueError:
            if rows:
                ended = True
            else:
                header.append(line)
            continue
        if ended:
            raise ValueError(f"Non-numeric interruption inside data table: {path}")
        try:
            rows.append([_float(word) for word in words])
        except ValueError as exc:
            raise ValueError(f"Malformed numeric row in {path}") from exc
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError(f"Missing or inconsistent numeric data in {path}")
    data = np.asarray(rows, dtype=float)
    if data.shape[1] < 2 or not np.isfinite(data).all():
        raise ValueError(f"Invalid/nonfinite numeric data in {path}")
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError(f"Non-increasing frequency grid in {path}")
    return data, "\n".join(header), text


def _unit(header: str, supplied: str | None) -> str:
    detected = {
        unit
        for pattern, unit in [
            (r"(?i:Rydberg)|\b(?:Ry|RY)\b", "Ry"),
            (r"cm(?:\^|\*\*)?-1", "cm^-1"),
            (r"\b(?:meV|MEV)\b", "meV"),
            (r"\b(?:eV|EV)\b", "eV"),
        ]
        if re.search(pattern, header)
    }
    if len(detected) > 1:
        raise ValueError("Ambiguous frequency units in header")
    if supplied is not None and supplied not in _UNITS:
        raise ValueError(f"Unsupported frequency unit: {supplied}")
    if detected and supplied is not None and supplied not in detected:
        raise ValueError(
            f"Frequency unit conflicts with header: {detected} vs {supplied}"
        )
    if supplied is None and not detected:
        raise ValueError("Frequency unit unavailable; supply frequency_unit explicitly")
    return supplied or detected.pop()


def read_a2f(path: str | Path, frequency_unit: str | None = None) -> A2FSpectrumSchema:
    """Read signed total/mode a2F columns and the optional midpoint-sum footer.

    Parameters
    ----------
    path
        a2F.dosN file, optionally gzip compressed.
    frequency_unit
        Ry, eV, meV or cm^-1. Inferred from the pre-data header when omitted.
        Amplitudes retain their native QE normalization without rescaling.

    Returns
    -------
    A2FSpectrumSchema
        All endpoints and columns, units, footer and source provenance. Missing
        footers carry a possible-truncation diagnostic.
    """
    data, header, text = _table(path)
    match = re.search(r"lambda\s*=\s*(" + _TOKEN + ")", text, re.IGNORECASE)
    footer = _finite(_float(match[1])) if match else None
    delta = re.search(r"Delta\s*=\s*(" + _TOKEN + ")", text, re.IGNORECASE)
    return {
        "frequency": data[:, 0].tolist(),
        "frequency_unit": _unit(header, frequency_unit),
        "alpha2F": data[:, 1].tolist(),
        "alpha2F_modes": data[:, 2:].tolist(),
        "mode_indices": list(range(1, data.shape[1] - 1)),
        "spectral_normalization": "QE native; signed amplitudes unmodified",
        "qe_footer_lambda": footer,
        "qe_footer_delta_frequency": _finite(_float(delta[1])) if delta else None,
        "diagnostics": []
        if footer is not None
        else ["possible_truncation_missing_or_invalid_lambda_footer"],
        "source": file_provenance(path),
    }


def read_matdyn_frequencies(
    path: str | Path, frequency_unit: str = "cm^-1"
) -> dict[str, Any]:
    """Read matdyn q coordinates and wrapped 6f10.4 mode-frequency blocks.

    Parameters
    ----------
    path
        flfrq file, optionally compressed.
    frequency_unit
        Native QE flfrq unit is cm^-1. Alternative declarations are retained and
        honored by the downstream instability conversion.

    Returns
    -------
    dict
        Q coordinates, display-precision weights, mode frequencies and units.
        Asterisks in a frequency field raise a specific overflow error.
    """
    if frequency_unit not in _UNITS:
        raise ValueError(f"Unsupported frequency unit: {frequency_unit}")
    lines = [line.rstrip() for line in _text(path).splitlines() if line.strip()]
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
        point = [_float(x) for x in lines[i].split()]
        if len(point) not in (3, 4):
            raise ValueError("Expected three q coordinates and an optional weight")
        points.append(point[:3])
        weights.append(point[3] if len(point) == 4 else None)
        i += 1
        values: list[float] = []
        while len(values) < modes and i < len(lines):
            line = lines[i]
            if "*" in line:
                raise ValueError("matdyn frequency field overflow (****)")
            try:
                block = [_float(x) for x in line.split()]
            except ValueError:
                block = [_float(line[j : j + 10]) for j in range(0, len(line), 10)]
            values.extend(block)
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
        "weight_precision": "QE display precision; not silently normalized",
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
    """Read signed total and atom-ordered phonon DOS columns.

    Parameters
    ----------
    path
        fldos file, optionally compressed.
    frequency_unit
        Native frequency unit, inferred from the header when omitted.

    Returns
    -------
    dict
        Frequency/DOS arrays, atom projections, native density units and hashes.
    """
    data, header, _ = _table(path)
    unit = _unit(header, frequency_unit)
    return {
        "frequency": data[:, 0].tolist(),
        "frequency_unit": unit,
        "total_dos": data[:, 1].tolist(),
        "atom_projected_dos": data[:, 2:].tolist(),
        "dos_unit": f"states per ({unit}) per cell",
        "source": file_provenance(path),
    }


def validate_epc_settings(
    mu_stars: Sequence[float], imaginary_tolerance_cm1: float = 10.0
) -> None:
    """Validate inexpensive analysis settings before scheduling solver jobs.

    Parameters
    ----------
    mu_stars
        Nonempty sequence of finite, nonnegative Coulomb pseudopotentials.
    imaginary_tolerance_cm1
        Finite, nonnegative numerical instability threshold.
    """
    if not len(mu_stars) or any(not np.isfinite(mu) or mu < 0 for mu in mu_stars):
        raise ValueError("mu_stars must be finite nonnegative values")
    if not np.isfinite(imaginary_tolerance_cm1) or imaginary_tolerance_cm1 < 0:
        raise ValueError("Imaginary-frequency tolerance must be finite and nonnegative")


def analyze_a2f(
    spectrum: A2FSpectrumSchema, mu_stars: Sequence[float] = (0.10,)
) -> A2FSpectrumSchema:
    """Integrate signed moments and evaluate each Tc method in its own domain.

    Moments are weighted integrals of alpha2F. The dimensionless coupling is
    lambda = 2 * integral(alpha2F / omega). The logarithmic average omega_log
    sets the characteristic frequency scale in the Tc formulas. The square root
    of the coupling-weighted second frequency moment, omega_2, enters the
    Allen-Dynes corrections. Both characteristic frequencies are reported in
    kelvin. Modified McMillan uses lambda and omega_log; Allen-Dynes also requires
    a valid second moment. These estimates are not numerical Eliashberg solutions.

    Parameters
    ----------
    spectrum
        Parsed a2F data. Moments use positive frequencies without clipping signed
        amplitudes. Modified McMillan requires lambda/omega_log; Allen-Dynes also
        requires a valid second moment.
    mu_stars
        Finite, nonnegative Coulomb pseudopotentials.

    Returns
    -------
    A2FSpectrumSchema
        Signed spectrum, dimensionless lambda, frequency moments in K or null,
        diagnostics and explicit Tc
        records. Negative-weight status tolerance is a lambda fraction of 1e-6;
        smaller signed residuals remain recorded. Lambda above 5 is flagged only.
    """
    validate_epc_settings(mu_stars)
    w_all = np.asarray(spectrum["frequency"], dtype=float)
    a_all = np.asarray(spectrum["alpha2F"], dtype=float)
    if (
        w_all.ndim != 1
        or w_all.shape != a_all.shape
        or not np.isfinite([w_all, a_all]).all()
        or np.any(np.diff(w_all) <= 0)
    ):
        raise ValueError("Invalid/nonfinite spectrum arrays")
    factor = _UNITS[spectrum["frequency_unit"]] * _EV_K
    positive = w_all > 0
    w, a = w_all[positive], a_all[positive]
    diagnostics = list(spectrum.get("diagnostics", []))
    warnings = list(diagnostics)
    result: dict[str, Any] = dict(
        spectrum,
        **{"lambda": None},
        omega_log_K=None,
        omega_2_K=None,
        mode_lambda=[],
        mode_sum_max_abs_error=None,
        Tc=[],
        diagnostics=diagnostics,
        warnings=warnings,
        negative_lambda_fraction_threshold=_NEGATIVE_FRACTION_TOLERANCE,
    )
    reason = "insufficient_positive_frequency_points"
    if np.any(a_all < 0):
        diagnostics.append("negative_spectral_weight")
    if np.any((~positive) & (a_all != 0)):
        warnings.append("nonpositive_frequency_weight_excluded_from_moments")
    if len(w) >= 3:
        with np.errstate(all="ignore"):
            lam = float(2 * simpson(a / w, x=w))
            lp = float(2 * simpson(np.maximum(a, 0) / w, x=w))
            ln = float(2 * simpson(np.minimum(a, 0) / w, x=w))
            fraction = _finite(abs(ln) / lp) if lp > 0 else None
            result.update(
                {
                    "lambda": _finite(lam),
                    "lambda_positive_part": _finite(lp),
                    "lambda_negative_part": _finite(ln),
                    "negative_lambda_fraction": fraction,
                }
            )
            if fraction is None or fraction > _NEGATIVE_FRACTION_TOLERANCE:
                warnings.append("significant_negative_or_nonfinite_spectral_weight")
            footer = spectrum.get("qe_footer_lambda")
            result["qe_footer_lambda_difference"] = (
                _finite(lam - footer) if footer is not None else None
            )
            delta = spectrum.get("qe_footer_delta_frequency")
            if delta is not None and delta > 0 and np.all(w_all != 0):
                midpoint = _finite(float(2 * np.sum(a_all / w_all) * delta))
                result["qe_midpoint_lambda"] = midpoint
                result["qe_midpoint_footer_difference"] = (
                    _finite(midpoint - footer)
                    if midpoint is not None and footer is not None
                    else None
                )
            if (
                footer is not None
                and np.isfinite(lam)
                and abs(lam - footer) > max(1e-3, 0.01 * abs(footer))
            ):
                warnings.append(
                    "Simpson_positive_frequency_vs_QE_midpoint_disagreement"
                )
            modes = np.asarray(spectrum["alpha2F_modes"], dtype=float)
            if (
                modes.shape != (len(w_all), len(spectrum["mode_indices"]))
                or not np.isfinite(modes).all()
            ):
                raise ValueError("Inconsistent/nonfinite mode-column dimensions")
            if modes.shape[1]:
                ml = 2 * simpson(modes[positive] / w[:, None], x=w, axis=0)
                result["mode_lambda"] = [_finite(v) for v in ml]
                result["mode_sum_max_abs_error"] = _finite(
                    float(np.max(np.abs(modes.sum(axis=1) - a_all)))
                )
                if not np.isfinite(ml).all():
                    warnings.append("nonfinite_mode_integral")
            reason = "nonpositive_or_nonfinite_lambda"
            if lam > 0 and np.isfinite(lam):
                logw = float(2 * simpson(a / w * np.log(w), x=w) / lam)
                w2sq = float(2 * simpson(a * w, x=w) / lam)
                wlog = _finite(float(np.exp(logw) * factor))
                w2 = _finite(float(np.sqrt(w2sq) * factor)) if w2sq > 0 else None
                result["omega_log_K"] = wlog if wlog is not None and wlog > 0 else None
                result["omega_2_K"] = w2 if w2 is not None and w2 > 0 else None
                reason = (
                    ""
                    if result["omega_log_K"] is not None
                    else "invalid_logarithmic_moment"
                )
                if result["omega_2_K"] is None:
                    warnings.append("invalid_second_moment")
                if lam > 5:
                    warnings.append("large_lambda_formula_regime")
    if reason:
        diagnostics.append(reason)
    for mu in mu_stars:
        for method in ["modified_mcmillan_omega_log", "allen_dynes_f1_f2"]:
            entry: TcEstimateSchema = {
                "method": method,
                "mu_star": float(mu),
                "Tc_K": None,
                "status": "unavailable",
                "reason": reason or None,
                "warnings": [],
            }
            if (
                not reason
                and method == "allen_dynes_f1_f2"
                and result["omega_2_K"] is None
            ):
                entry["reason"] = "invalid_second_moment"
            if not entry["reason"]:
                lam = np.float64(result["lambda"])
                with np.errstate(all="ignore"):
                    denom = lam - mu * (1 + 0.62 * lam)
                    if not np.isfinite(denom) or denom <= 0:
                        entry["reason"] = "nonpositive_formula_denominator"
                    else:
                        if denom < 1e-6 * max(1, lam):
                            entry["warnings"].append("near_zero_formula_denominator")
                        logtc = (
                            np.log(result["omega_log_K"] / 1.2)
                            - 1.04 * (1 + lam) / denom
                        )
                        if method == "allen_dynes_f1_f2":
                            f1 = (1 + (lam / (2.46 * (1 + 3.8 * mu))) ** 1.5) ** (1 / 3)
                            ratio = (
                                np.float64(result["omega_2_K"]) / result["omega_log_K"]
                            )
                            f2 = 1 + lam**2 * (ratio - 1) / (
                                lam**2 + (1.82 * (1 + 6.3 * mu) * ratio) ** 2
                            )
                            entry["f1"] = _finite(f1)
                            entry["f2"] = _finite(f2)
                            logtc += np.log(f1 * f2)
                        if not np.isfinite(logtc) or not np.log(
                            np.finfo(float).tiny
                        ) < logtc < np.log(np.finfo(float).max):
                            entry["reason"] = "Tc_numerically_unrepresentable"
                        else:
                            entry["Tc_K"] = float(np.exp(logtc))
                            entry["status"] = "estimated"
            result["Tc"].append(entry)
    if any(tc["status"] != "estimated" for tc in result["Tc"]):
        diagnostics.append("Tc_unavailable")
        warnings.append("Tc_unavailable")
    result["status"] = (
        "invalid" if reason else "analyzed_with_warnings" if warnings else "analyzed"
    )
    return cast("A2FSpectrumSchema", result)


def epc_file_manifest(
    directory: str | Path, stage: str, dynamical_prefix: str = "matdyn"
) -> dict[str, Any]:
    """Validate exact QE 7.x EPC file identities and full-precision widths.

    Parameters
    ----------
    directory
        Completed DFPT source for q2r, or completed q2r source for matdyn.
    stage
        q2r expects a2Fq2r.51..50+n at every q; matdyn expects a2Fmatdyn.61..60+n.
    dynamical_prefix
        Plain dynamical-matrix basename prefix. Paths/globs are unsupported.

    Returns
    -------
    dict
        Staging names, hashes, q mesh and ordered Ry widths read from the first
        line of the EPC files. Rounded f7.3 logs are checked only to 5e-4 Ry.
    """
    directory = Path(directory).expanduser().resolve()
    if stage not in ("q2r", "matdyn"):
        raise ValueError("stage must be q2r or matdyn")
    if not re.fullmatch(r"[\w.-]+", dynamical_prefix) or dynamical_prefix in (
        ".",
        "..",
    ):
        raise ValueError(
            "Dynamical matrix prefix must be a basename without glob characters"
        )
    elph = directory / "elph_dir"
    if elph.is_symlink():
        raise ValueError("Symbolic-link EPC directory is unsupported")
    if not elph.is_dir():
        raise FileNotFoundError(f"Missing EPC directory: {elph}")
    metadata = sorted(elph.glob("elph.inp_lambda.*"))
    if not metadata:
        raise FileNotFoundError("Missing elph.inp_lambda broadening metadata")
    rounded = [
        [
            _float(x)
            for x in re.findall(
                r"Gaussian Broadening:\s*(" + _NUMBER + r")\s*Ry", _text(p)
            )
        ]
        for p in metadata
    ]
    if not rounded[0] or any(values != rounded[0] for values in rounded):
        raise ValueError("Inconsistent/missing EPC broadening metadata")
    n = len(rounded[0])
    files = list(metadata)
    if stage == "q2r":
        lines = _text(directory / f"{dynamical_prefix}0").splitlines()
        if len(lines) < 2:
            raise ValueError("Invalid DFPT q-grid header")
        grid = [int(i) for i in lines[0].split()]
        nq = int(lines[1])
        if len(grid) != 3 or any(v < 1 for v in grid) or nq < 1:
            raise ValueError("Invalid DFPT q-grid header")
        metadata_indices = {
            int(p.name.removesuffix(".gz").split(".")[-1]) for p in metadata
        }
        if metadata_indices != set(range(1, nq + 1)) or len(metadata) != nq:
            raise ValueError("Incomplete/ambiguous EPC q-point metadata")
        matrices = [
            _resolve_file(directory / f"{dynamical_prefix}{i}") for i in range(nq + 1)
        ]
        files.extend(matrices)
        matched = {}
        for path in elph.glob("a2Fq2r.*"):
            m = re.fullmatch(r"a2Fq2r\.(\d+)\.(\d+)(?:\.gz)?", path.name)
            if m:
                key = int(m[1]), int(m[2])
                if key in matched:
                    raise ValueError(f"Ambiguous duplicate EPC input: {path.name}")
                matched[key] = _resolve_file(path)
        expected = {(code, q) for code in range(51, 51 + n) for q in range(1, nq + 1)}
        if matched.keys() != expected:
            raise ValueError(
                "Incomplete or incorrect DFPT EPC q-point/broadening file numbers"
            )
        series = [
            [matched[code, q] for code in range(51, 51 + n)] for q in range(1, nq + 1)
        ]
    else:
        grid = None
        fc = _resolve_file(directory / "q2r.fc")
        files.append(fc)
        matched_transforms = {}
        for path in elph.glob("a2Fmatdyn.*"):
            m = re.fullmatch(r"a2Fmatdyn\.(\d+)(?:\.gz)?", path.name)
            if m:
                code = int(m[1])
                if code in matched_transforms:
                    raise ValueError("Ambiguous duplicate q2r EPC force constants")
                matched_transforms[code] = _resolve_file(path)
        if matched_transforms.keys() != set(range(61, 61 + n)):
            raise ValueError(
                "Incomplete or incorrect q2r EPC force-constant file numbers"
            )
        series = [[matched_transforms[code] for code in range(61, 61 + n)]]
    try:
        widths = [
            [_float(_text(p).splitlines()[0].split()[0]) for p in row] for row in series
        ]
    except (IndexError, ValueError) as exc:
        raise ValueError("Malformed full-precision EPC header") from exc
    full = widths[0]
    if any(not np.isfinite(v) or v <= 0 for v in full):
        raise ValueError("EPC broadenings must be finite positive values")
    if any(not np.allclose(row, full, rtol=1e-10, atol=1e-14) for row in widths):
        raise ValueError("Inconsistent full-precision EPC broadening headers")
    if not np.allclose(rounded, full, rtol=0, atol=5.00000001e-4):
        raise ValueError("Rounded EPC metadata disagrees with full-precision headers")
    files.extend(p for row in series for p in row)
    # Reject ambiguous entries before selecting the exact files. Individual file
    # copies create writable destination parents even for read-only gzip sources.
    for p in elph.rglob("*"):
        if p.is_symlink() or p.is_file():
            _resolve_file(p)
    return {
        "source_directory": str(directory),
        "filenames": [p.relative_to(directory).as_posix() for p in files],
        "qmesh": grid,
        "broadenings_Ry": full,
        "broadening_source": "full-precision EPC first-line deg",
        "rounded_broadening_tolerance_Ry": 5e-4,
        "files": [file_provenance(p) for p in files],
    }


def epc_copy_files(
    source_directory: str | Path,
    filenames: Sequence[str | Path],
    copy_files: CopyFiles | None,
    dynamical_prefix: str = "matdyn",
) -> CopyFiles:
    """Combine staging specifications while rejecting replacement of EPC inputs.

    Parameters
    ----------
    source_directory
        Validated authoritative source.
    filenames
        Files selected by its manifest.
    copy_files
        Optional additional QuAcc transfer specifications.
    dynamical_prefix
        Dynamical-matrix basename to protect from extra specifications.

    Returns
    -------
    list
        Additional specifications followed by the authoritative source. Extras
        targeting elph_dir, force constants or dynamical matrices are rejected.
    """
    for spec in copy_files or []:
        patterns = spec["filenames"]
        if not isinstance(patterns, list):
            patterns = [patterns]
        for pattern in patterns:
            for path in Path(spec["source"]).expanduser().glob(str(pattern)):
                relative = path.relative_to(Path(spec["source"]).expanduser())
                name = relative.name.removesuffix(".gz")
                if (
                    relative.parts[0] == "elph_dir"
                    or name == "q2r.fc"
                    or re.fullmatch(re.escape(dynamical_prefix) + r"\d+", name)
                ):
                    raise ValueError(
                        f"Extra copy_files overlaps validated EPC inputs: {relative}"
                    )
    return [
        *(copy_files or []),
        {"source": source_directory, "filenames": list(filenames)},
    ]


def _read_qsum(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, diagnostics = [], []
    text = _text(path)
    for number, line in enumerate(text.splitlines(), 1):
        if "Broadening" not in line:
            continue
        match = re.search(
            r"Broadening\s*("
            + _TOKEN
            + r")\s*lambda\s*("
            + _TOKEN
            + r")\s*dos\(Ef\)\s*("
            + _TOKEN
            + r")\s*omega_ln\s*\[K\]\s*("
            + _TOKEN
            + ")",
            line,
            re.IGNORECASE,
        )
        if not match:
            diagnostics.append(
                {
                    "component": "qe_qsum",
                    "status": "malformed_row",
                    "line_number": number,
                    "raw_line": line,
                }
            )
            continue
        values = [_finite(_float(v)) for v in match.groups()]
        row = dict(
            zip(
                ["broadening_Ry", "lambda", "DOS_EF_QE_native", "omega_log_K"],
                values,
                strict=True,
            )
        )
        row["line_number"] = number
        rows.append(row)
        if any(v is None for v in values):
            diagnostics.append(
                {
                    "component": "qe_qsum",
                    "status": "nonfinite_row",
                    "line_number": number,
                }
            )
    return rows, diagnostics


def _invalidate(record: dict[str, Any], reason: str) -> None:
    record.update(
        status="invalid",
        reason=reason,
        **{"lambda": None},
        omega_log_K=None,
        omega_2_K=None,
    )
    for tc in record.get("Tc", []):
        tc.update(Tc_K=None, status="unavailable", reason=reason)


def analyze_epc_directory(
    directory: str | Path,
    mu_stars: Sequence[float] = (0.10,),
    broadenings_Ry: Sequence[float] | None = None,
    imaginary_tolerance_cm1: float = 10.0,
) -> EPCResultSchema:
    """Analyze completed matdyn files with broadening-resolved diagnostics.

    Parameters
    ----------
    directory
        Directory containing a2F.dosN, matdyn.freq, matdyn.dos and optional lambda.
    mu_stars
        Coulomb pseudopotentials for both named formula estimates.
    broadenings_Ry
        Ordered full-precision widths from the EPC input manifest. When absent,
        identities remain unknown; the rounded lambda table cannot supply them.
    imaginary_tolerance_cm1
        Numerical sampled-instability threshold, not a physical stability proof.

    Returns
    -------
    EPCResultSchema
        Signed spectra, finite moments or null, per-method Tc, component and
        aggregate diagnostics, native units and source-file provenance.
    """
    directory = Path(directory).expanduser().resolve()
    validate_epc_settings(mu_stars, imaginary_tolerance_cm1)
    if broadenings_Ry is not None and (
        not len(broadenings_Ry)
        or any(not np.isfinite(v) or v <= 0 for v in broadenings_Ry)
    ):
        raise ValueError("Broadenings must be finite positive values")
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "analyzed",
        "directory": str(directory),
        "method_scope": "isotropic formula-based Tc estimates, not numerical Eliashberg solutions",
        "mu_stars": [float(mu) for mu in mu_stars],
        "spectra": [],
        "diagnostics": [],
        "frequencies": None,
        "phonon_dos": None,
        "instability": None,
        "imaginary_tolerance_cm1": float(imaginary_tolerance_cm1),
    }
    diagnostics = result["diagnostics"]
    for key, name, parser in [
        ("frequencies", "matdyn.freq", read_matdyn_frequencies),
        ("phonon_dos", "matdyn.dos", read_phonon_dos),
    ]:
        try:
            result[key] = parser(directory / name)
        except (OSError, ValueError, ArithmeticError, IndexError) as exc:
            diagnostics.append(
                {
                    "component": key,
                    "status": "unavailable_or_invalid",
                    "reason": str(exc),
                }
            )
    available: dict[int, list[Path]] = {}
    for path in directory.glob("a2F.dos*"):
        match = re.fullmatch(r"a2F\.dos([1-9]\d*)(?:\.gz)?", path.name)
        if match:
            available.setdefault(int(match[1]), []).append(path)
    qe_qsum: list[dict[str, Any]] = []
    try:
        qe_qsum, table_diagnostics = _read_qsum(directory / "lambda")
        diagnostics.extend(table_diagnostics)
        result["qe_qsum_source"] = file_provenance(directory / "lambda")
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:
        diagnostics.append(
            {
                "component": "qe_qsum",
                "status": "unavailable_or_invalid",
                "reason": str(exc),
            }
        )
    result["qe_qsum"] = qe_qsum
    if broadenings_Ry is None:
        diagnostics.append(
            {
                "component": "broadening_series",
                "status": "identities_and_expected_count_unknown",
            }
        )
    expected = (
        set(range(1, len(broadenings_Ry) + 1))
        if broadenings_Ry is not None
        else set(range(1, max(available, default=0) + 1))
    )
    if available.keys() - expected:
        diagnostics.append(
            {
                "component": "broadening_series",
                "status": "unexpected_spectrum_indices",
                "indices": sorted(available.keys() - expected),
            }
        )
    matched_rows = set()
    for index in sorted(expected | available.keys()):
        width = (
            float(broadenings_Ry[index - 1])
            if broadenings_Ry is not None and index <= len(broadenings_Ry)
            else None
        )
        record: dict[str, Any] = {"index": index, "broadening_Ry": width, "Tc": []}
        paths = available.get(index, [])
        if not paths:
            record.update(
                status="unavailable",
                reason="missing_spectrum",
                **{"lambda": None},
                omega_log_K=None,
                omega_2_K=None,
            )
        elif len(paths) > 1:
            _invalidate(record, "Ambiguous duplicate spectrum index")
            diagnostics.append(
                {"component": f"spectrum_{index}", "status": "ambiguous_plain_gzip"}
            )
        else:
            try:
                record.update(analyze_a2f(read_a2f(paths[0]), mu_stars))
            except (OSError, ValueError, ArithmeticError, IndexError) as exc:
                _invalidate(record, str(exc))
        if width is not None and qe_qsum and broadenings_Ry is not None:
            matches = [
                i
                for i, row in enumerate(qe_qsum)
                if row["broadening_Ry"] is not None
                and abs(row["broadening_Ry"] - width) <= 5.00000001e-5
            ]
            if len(matches) == 1:
                row_index = matches[0]
                # A rounded F8.4 value may match more than one true width.
                identities = [
                    v
                    for v in broadenings_Ry
                    if abs(qe_qsum[row_index]["broadening_Ry"] - v) <= 5.00000001e-5
                ]
                if len(identities) == 1:
                    record["qe_qsum"] = qe_qsum[row_index]
                    matched_rows.add(row_index)
                else:
                    diagnostics.append(
                        {
                            "component": f"spectrum_{index}",
                            "status": "ambiguous_rounded_qsum_width",
                        }
                    )
            elif len(matches) > 1:
                diagnostics.append(
                    {"component": f"spectrum_{index}", "status": "duplicate_qsum_width"}
                )
            elif len(qe_qsum) == len(broadenings_Ry) and not table_diagnostics:
                _invalidate(record, "broadening_identity_conflict")
            else:
                diagnostics.append(
                    {"component": f"spectrum_{index}", "status": "qsum_match_missing"}
                )
        result["spectra"].append(record)
    if qe_qsum and broadenings_Ry is not None and len(matched_rows) < len(qe_qsum):
        diagnostics.append(
            {
                "component": "qe_qsum",
                "status": "unmatched_or_ambiguous_rows",
                "row_indices": sorted(set(range(len(qe_qsum))) - matched_rows),
            }
        )
    frequencies = result["frequencies"]
    if frequencies is not None:
        with np.errstate(all="ignore"):
            values = (
                np.asarray(frequencies["frequencies"])
                * _UNITS[frequencies["frequency_unit"]]
                * _EV_CM
            )
        unstable = False
        if not np.isfinite(values).all():
            diagnostics.append(
                {
                    "component": "frequencies",
                    "status": "frequency_conversion_unrepresentable",
                }
            )
        else:
            unstable = bool(np.any(values < -imaginary_tolerance_cm1))
            result["instability"] = {
                "minimum_frequency_cm1": float(values.min()),
                "negative_frequency_count": int(np.sum(values < 0)),
                "below_tolerance_count": int(np.sum(values < -imaginary_tolerance_cm1)),
                "sampled_imaginary_modes": unstable,
            }
        for flag in frequencies["diagnostics"]:
            diagnostics.append({"component": "frequencies", "status": flag})
        if unstable:
            for record in result["spectra"]:
                for tc in record["Tc"]:
                    tc.setdefault("warnings", []).append(
                        "sampled_instability_QE_may_zero_imaginary_mode_gamma"
                    )
    if result["phonon_dos"] is not None:
        dos = result["phonon_dos"]
        with np.errstate(all="ignore"):
            integral = (
                _finite(float(simpson(dos["total_dos"], x=dos["frequency"])))
                if len(dos["frequency"]) > 1
                else None
            )
        expected_modes = frequencies["number_of_modes"] if frequencies else None
        dos["diagnostics"] = {
            "integrated_states": integral,
            "negative_bins": int(np.sum(np.asarray(dos["total_dos"]) < 0)),
            "expected_modes": expected_modes,
        }
        if integral is None:
            dos["diagnostics"]["status"] = "insufficient_or_nonfinite_grid_integral"
        elif expected_modes is not None and not np.isclose(
            integral, expected_modes, rtol=0.05
        ):
            dos["diagnostics"]["status"] = "DOS_normalization_outside_5_percent"
        if "status" in dos["diagnostics"]:
            diagnostics.append(
                {"component": "phonon_dos", "status": dos["diagnostics"]["status"]}
            )
    if (
        not result["spectra"]
        or any(r["status"] in ("invalid", "unavailable") for r in result["spectra"])
        or frequencies is None
        or result["instability"] is None
        or result["phonon_dos"] is None
    ):
        result["status"] = "incomplete_or_invalid"
    elif result["instability"]["sampled_imaginary_modes"]:
        result["status"] = "sampled_instability"
    elif diagnostics or any(r["warnings"] for r in result["spectra"]):
        result["status"] = "analyzed_with_warnings"
    return cast("EPCResultSchema", result)

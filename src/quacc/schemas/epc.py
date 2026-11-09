"""QE EPC spectrum integration, explicitly selected spectral provenance and reporting.

Frequency input is Ry; omega_log and the second moment are reported in kelvin.
Signed negative spectral weight is retained. No announcement threshold or
experimental-Tc fitting is applied. Plotting requires matplotlib and scipy.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC
from pathlib import Path

import numpy as np
from scipy.constants import physical_constants

RY_EV = physical_constants["Rydberg constant times hc in eV"][0]
RY_K = RY_EV / physical_constants["Boltzmann constant in eV/K"][0]
CM1_PER_MEV = physical_constants["electron volt-inverse meter relationship"][0] / 100000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_numeric_table(path):
    """Read numeric rows, retaining the last data row and skipping QE footers."""
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            r = [float(x.replace("D", "E").replace("d", "e")) for x in line.split()]
        except ValueError:
            continue
        if len(r) >= 2:
            rows.append(r)
    return np.array(rows, dtype=float)


def interpolated_frequency_check(path):
    """Read QE 7.5's four q/weight values and nbnd frequencies per point.

    The fixed-width nks header overflows on dense grids; infer the point count
    from the numeric body instead of trusting that header field.
    """
    lines = Path(path).read_text().splitlines()
    bands = int(re.search(r"nbnd\s*=\s*(\d+)", lines[0]).group(1))
    data = np.fromstring(" ".join(lines[1:]), sep=" ")
    if data.size % (4 + bands):
        raise ValueError("Malformed matdyn frequency grid")
    frequencies = data.reshape(-1, 4 + bands)[:, 4:]
    return {
        "points": len(frequencies),
        "bands": bands,
        "minimum_cm1": float(frequencies.min()),
        "below_minus10_cm1": int(np.sum(frequencies < -10)),
        "below_minus45_cm1": int(np.sum(frequencies < -45)),
    }


def integrate_a2f(path, mu_values=(0.10, 0.13, 0.15)):
    """QE a2F.dos: omega in Ry; total spectrum column 2. No frequency rescaling."""
    from scipy.integrate import simpson

    data = read_numeric_table(path)
    if data.ndim != 2 or len(data) < 5:
        raise ValueError("Not enough numeric a2F rows")
    if not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("Invalid frequency grid")
    mask = data[:, 0] > 0
    w, a = data[mask, 0], data[mask, 1]
    flags = []
    if np.any(data[:, 1] < -1e-12):
        flags.append("negative_spectral_weight")
    if np.any((data[:, 0] <= 0) & (np.abs(data[:, 1]) > 1e-12)):
        flags.append("nonpositive_frequency_weight")
    lam = float(2 * simpson(a / w, x=w))
    positive = float(2 * simpson(np.maximum(a, 0) / w, x=w))
    negative = float(2 * simpson(np.minimum(a, 0) / w, x=w))
    out = {
        "file": str(path),
        "frequency_unit": "Ry",
        "lambda": lam,
        "flags": flags,
        "numeric_rows": len(data),
        "nonpositive_frequency_rows": int(sum(~mask)),
        "negative_a2f_min": float(data[:, 1].min()),
        "negative_spectral_bin_count": int(np.sum(a < -1e-12)),
        "lambda_positive_part": positive,
        "lambda_negative_part": negative,
        "negative_lambda_fraction": abs(negative) / positive if positive > 0 else None,
        "Tc": [],
    }
    footer = re.search(r"lambda\s*=\s*([-+\d.EeDd]+)", Path(path).read_text())
    if footer:
        qe_lambda = float(footer.group(1).replace("D", "E").replace("d", "e"))
        out["qe_reported_lambda"] = qe_lambda
        out["lambda_difference_relative"] = abs(lam - qe_lambda) / max(
            abs(qe_lambda), 1e-12
        )
        if out["lambda_difference_relative"] > 0.05:
            out["flags"].append("integration_differs_from_QE_by_over_5_percent")
    if lam <= 0 or not np.isfinite(lam):
        out["flags"].append("nonpositive_or_invalid_lambda")
        return out
    wlog = float(np.exp(2 * simpson(a / w * np.log(w), x=w) / lam) * RY_K)
    w2sq = float(2 * simpson(a * w, x=w) / lam)
    if w2sq <= 0:
        out["flags"].append("invalid_second_moment")
        return out
    w2 = float(np.sqrt(w2sq) * RY_K)
    if not np.isfinite(wlog) or not np.isfinite(w2) or wlog <= 0:
        out["flags"].append("invalid_frequency_moments")
        return out
    out.update(omega_log_K=wlog, omega_2_K=w2)
    for mu in mu_values:
        denom = lam - mu * (1 + 0.62 * lam)
        if denom <= 0:
            out["Tc"].append(
                {
                    "mu_star": mu,
                    "McMillan_omega_log_K": None,
                    "Allen_Dynes_K": None,
                    "flag": "nonpositive_McMillan_denominator",
                }
            )
            continue
        tc = float(wlog / 1.2 * np.exp(-1.04 * (1 + lam) / denom))
        f1 = (1 + (lam / (2.46 * (1 + 3.8 * mu))) ** 1.5) ** (1 / 3)
        ratio = w2 / wlog
        f2 = 1 + lam**2 * (ratio - 1) / (lam**2 + (1.82 * (1 + 6.3 * mu) * ratio) ** 2)
        out["Tc"].append(
            {
                "mu_star": mu,
                "McMillan_omega_log_K": tc,
                "Allen_Dynes_K": float(f1 * f2 * tc),
                "f1": float(f1),
                "f2": float(f2),
            }
        )
    return out


def _json(path, data):
    """Replace a JSON file only after fsync; keep intermediate failures distinguishable."""
    import os
    import tempfile

    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
        name = f.name
    Path(name).replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def selected_spectrum(path, mu_values=(0.10, 0.13, 0.15)):
    """One identified spectrum supplies plotted alpha2F, cumulative lambda and Tc."""
    from scipy.integrate import cumulative_simpson

    result = integrate_a2f(path, mu_values)
    data = read_numeric_table(path)
    data = data[data[:, 0] > 0]
    cumulative = cumulative_simpson(
        2 * data[:, 1] / data[:, 0], x=data[:, 0], initial=0
    )
    if not np.isclose(cumulative[-1], result["lambda"], rtol=1e-10, atol=1e-12):
        raise ValueError("Cumulative lambda endpoint disagrees with scalar integration")
    result.update(
        source_sha256=sha(path), cumulative_lambda_endpoint=float(cumulative[-1])
    )
    return result, data[:, 0] * RY_EV * 1000, data[:, 1], cumulative


def _frequency_matrix(path):
    lines = Path(path).read_text().splitlines()
    nbnd = int(re.search(r"nbnd\s*=\s*(\d+)", lines[0]).group(1))
    data = np.fromstring(" ".join(lines[1:]), sep=" ")
    # QE 7.5 writes qx,qy,qz,weight for both DOS and arbitrary paths.
    if data.size % (nbnd + 4):
        raise ValueError("Malformed QE frequency table")
    return data.reshape(-1, nbnd + 4)[:, 4:]


def _plot_path(ax, info, values, **kwargs):
    x = np.array(info["axis_inverse_A"])
    q = np.array(info["kpoints_fractional"])
    breaks = (
        np.where(
            (np.abs(np.diff(x)) < 1e-10)
            & (np.linalg.norm(np.diff(q, axis=0), axis=1) > 1e-7)
        )[0]
        + 1
    )
    for inds in np.split(np.arange(len(x)), breaks):
        if len(inds):
            ax.plot(x[inds], values[inds], **kwargs)
    labels = [r"$\Gamma$" if t == "G" else t for t in info["labels"]]
    # Combine labels at disconnected segments at the same x coordinate.
    ticks, texts = [], []
    for t, label in zip(info["ticks"], labels, strict=True):
        if ticks and abs(t - ticks[-1]) < 1e-9:
            if label != texts[-1]:
                texts[-1] += "|" + label
        else:
            ticks.append(t)
            texts.append(label)
    ax.set_xticks(ticks, texts)
    ax.set_xlim(float(x[0]), float(x[-1]))
    for t in ticks:
        ax.axvline(t, color="0.85", lw=0.6)


def save_epc_report(bands, nscf, dos, pdos, phonons, spectrum, metadata, output_dir):
    """Save immutable report artifacts and a terminal event for an external outbox.

    Project metadata identifies pressure, spectrum, mu*, material/run and warnings.
    Projections are summed over each atom and orbital's LDOS column; the magnetic
    case is rejected rather than silently misparsed. All energy curves share one
    explicitly recorded SCF EF. This function never contacts Discord.
    """
    import shutil
    from datetime import datetime

    import matplotlib as mpl

    mpl.use("Agg")
    from ase.io import read
    from matplotlib import pyplot as plt

    d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)
    if (d / "COMPLETE.json").exists():
        raise FileExistsError(
            "Report already complete; reuse artifacts, do not overwrite"
        )
    mu = metadata["mu_star"]
    spectrum = dict(spectrum)
    if "file" not in spectrum:
        spectrum["file"] = str(Path(spectrum["directory"]) / spectrum["filename"])
    selected, energy, a2f, cumulative = selected_spectrum(spectrum["file"], (mu,))
    tc = selected["Tc"][0] if selected["Tc"] else {}
    selected["identity"] = spectrum
    # Read actual native pw eigenvalues (eV). A bands path has no independent EF.
    atom = read(Path(bands["dir_name"]) / "pw.out", format="espresso-out")
    if atom.calc.get_number_of_spins() != 1:
        raise NotImplementedError(
            "Spin-resolved electronic plotting needs an explicit parser"
        )
    eigen = np.array(
        [
            atom.calc.get_eigenvalues(kpt=k)
            for k in range(len(bands["bandpath"]["axis_inverse_A"]))
        ]
    )
    ef = float(metadata["scf_fermi_eV"])
    total = read_numeric_table(Path(dos["dir_name"]) / "pwscf.dos")
    projected = {}
    projected_energy = None
    files = sorted(Path(pdos["dir_name"]).glob("pwscf.pdos_atm#*"))
    if not files:
        raise ValueError("No atom/orbital projected DOS files")
    for p in files:
        element = re.search(r"atm#\d+\(([^)]+)\)", p.name).group(1)
        header = p.read_text().splitlines()[0].lower()
        if "up" in header or "down" in header:
            raise NotImplementedError("Spin-resolved PDOS not implemented")
        data = read_numeric_table(p)
        if projected_energy is None:
            projected_energy = data[:, 0]
        elif data.shape[0] != len(projected_energy) or not np.allclose(
            data[:, 0], projected_energy, atol=1e-6, rtol=0
        ):
            raise ValueError("Atomic/orbital PDOS energy grids differ")
        projected[element] = projected.get(element, np.zeros(len(data))) + data[:, 1]
    np.savez_compressed(
        d / "electronic_data.npz",
        eigenvalues_eV=eigen,
        EF_eV=ef,
        DOS_energy_eV=total[:, 0],
        projected_DOS_energy_eV=projected_energy,
        total_DOS=total[:, 1],
        **{f"PDOS_{k}": v for k, v in projected.items()},
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5),
        sharey=True,
        gridspec_kw={"width_ratios": [3, 1.3]},
        layout="constrained",
    )
    _plot_path(axes[0], bands["bandpath"], eigen - ef, color="#264f73", lw=0.65)
    axes[0].set_ylabel(r"Energy $E-E_F$ (eV)")
    axes[0].set_xlabel("Reciprocal-space path")
    axes[1].plot(total[:, 1], total[:, 0] - ef, color="black", label="Total")
    for element, values in projected.items():
        axes[1].plot(values, projected_energy - ef, label=element)
    axes[1].set_xlabel("DOS (states/eV/cell)")
    axes[1].legend()
    for ax in axes:
        ax.axhline(0, color="0.4", ls="--", lw=0.8)
        ax.set_ylim(*metadata.get("energy_window_eV", [-10, 10]))
    fig.suptitle(
        f"{metadata['material_id']} | {metadata['run_id']} | P target={metadata['pressure_GPa']:g} GPa\n"
        f"All energies referenced to SCF EF={ef:.4f} eV"
    )
    for ext in ["png", "pdf"]:
        fig.savefig(d / f"electronic_bands_dos.{ext}", dpi=180)
    plt.close(fig)

    frequencies = _frequency_matrix(Path(phonons["dir_name"]) / "matdyn.freq")
    phonon_dos = read_numeric_table(Path(spectrum["directory"]) / "matdyn.dos")
    grid_check = interpolated_frequency_check(
        Path(spectrum["directory"]) / "matdyn.freq"
    )
    minimum = float(frequencies.min())
    unstable = bool(np.any(frequencies < -10)) or grid_check["below_minus10_cm1"] > 0
    warnings = list(metadata.get("warnings", [])) + selected["flags"]
    if not tc or tc.get("Allen_Dynes_K") is None:
        warnings.append(
            "Tc undefined for this spectrum/mu*; inspect lambda and denominator"
        )
    if unstable:
        warnings.append(
            "imaginary phonon modes below -10 cm^-1; Tc not physically validated"
        )
    status = (
        "UNSTABLE"
        if unstable
        else "COMPLETED_WITH_WARNINGS"
        if warnings
        else "COMPLETED"
    )
    np.savez_compressed(
        d / "phonon_epc_data.npz",
        frequencies_cm1=frequencies,
        energy_meV=energy,
        alpha2F=a2f,
        cumulative_lambda=cumulative,
        phonon_DOS_frequency_cm1=phonon_dos[:, 0],
        phonon_DOS=phonon_dos[:, 1],
    )
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(14, 5),
        sharey=True,
        gridspec_kw={"width_ratios": [3, 1, 1.3, 1.3]},
        layout="constrained",
    )
    _plot_path(
        axes[0], phonons["bandpath"], frequencies / CM1_PER_MEV, color="#264f73", lw=0.7
    )
    axes[0].set_ylabel(r"Phonon energy $\hbar\omega$ (meV)")
    axes[0].set_xlabel("Reciprocal-space path")
    axes[1].plot(
        phonon_dos[:, 1] * CM1_PER_MEV, phonon_dos[:, 0] / CM1_PER_MEV, color="#264f73"
    )
    axes[1].set_xlabel("Phonon DOS\n(states/meV/cell)")
    axes[2].plot(a2f, energy, color="#bb5e17")
    axes[2].axvline(0, color="0.6", lw=0.7)
    axes[2].set_xlabel(r"$\alpha^2F(\omega)$")
    axes[3].plot(cumulative, energy, color="#477a3c")
    axes[3].axvline(selected["lambda"], color="0.6", lw=0.7, ls="--")
    axes[3].set_xlabel(r"Cumulative $\lambda(\omega)$")
    for ax in axes:
        ax.axhline(0, color="0.6", lw=0.7)
        ax.grid(alpha=0.15)

    def formatted(value, digits=3):
        return (
            f"{value:.{digits}f}"
            if value is not None and np.isfinite(value)
            else "undefined"
        )

    fig.suptitle(
        f"{metadata['material_id']} | {metadata['run_id']} | {status}\n"
        f"q={spectrum['qmesh']}, spectrum {Path(spectrum['file']).name}, sigma={spectrum['broadening_Ry']:.3f} Ry; "
        f"lambda={selected['lambda']:.5f}, omega_log={formatted(selected.get('omega_log_K'), 2)} K\n"
        f"mu*={mu:g}; McMillan={formatted(tc.get('McMillan_omega_log_K'))} K; "
        f"Allen-Dynes={formatted(tc.get('Allen_Dynes_K'))} K (isotropic estimates)"
    )
    for ext in ["png", "pdf"]:
        fig.savefig(d / f"phonon_dispersion_epc.{ext}", dpi=180)
    plt.close(fig)
    shutil.copy2(spectrum["file"], d / "selected_a2F.dos")
    _json(
        d / "bandpaths.json",
        {"electronic": bands["bandpath"], "phonon": phonons["bandpath"]},
    )
    result = {
        "schema_version": 1,
        "status": status,
        "metadata": metadata,
        "selected_spectrum": selected,
        "Tc": tc,
        "warnings": warnings,
        "frequency_unit_plots": "meV",
        "omega_log_unit": "K",
        "electronic_EF_eV": ef,
        "NSCF_EF_eV": nscf["results"].get("fermi_level"),
        "minimum_path_frequency_cm1": minimum,
        "output_directory": str(d),
        "interpolated_frequency_check": grid_check,
        "projection_note": "Element LDOS sums over atomic/orbital projections; an incomplete atomic basis need not sum to total DOS",
        "DOS_grid_note": "Native dos.x/projwfc.x grids retained independently; endpoints can differ by one energy step",
        "DOS_grid_points": len(total),
        "PDOS_grid_points": len(projected_energy),
        "created_utc": datetime.now(UTC).isoformat(),
    }
    result["report_source_sha256"] = sha(__file__)
    result["executions"] = {
        k: v.get("execution")
        for k, v in {
            "bands": bands,
            "nscf": nscf,
            "dos": dos,
            "pdos": pdos,
            "phonon_path": phonons,
        }.items()
    }
    _json(d / "scientific_result.json", result)
    # Completion marker is last. Delivery code checks every digest before posting.
    import os

    for p in d.iterdir():
        if p.is_file():
            with p.open("rb") as f:
                os.fsync(f.fileno())
    artifacts = {
        p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in d.iterdir()
        if p.is_file() and p.name != "COMPLETE.json"
    }
    _json(d / "COMPLETE.json", {"status": status, "artifacts": artifacts})
    return result

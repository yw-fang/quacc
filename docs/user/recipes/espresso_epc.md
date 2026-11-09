# EPC postprocessing from completed Quantum ESPRESSO calculations

[`epc_postprocess_flow`][quacc.recipes.espresso.epc.epc_postprocess_flow]
starts from completed DFPT/EPC data and runs **q2r → matdyn → scientific analysis**.
It does not run relaxation, SCF or DFPT, require electronic bands/PDOS, or import
plotting/notification code. Configure QE and QuAcc's execution environment in the
usual way before running a calculation.

```python
import json
from quacc.recipes.espresso.epc import epc_postprocess_flow

results = epc_postprocess_flow(
    source_directory="/path/to/completed/dfpt",
    mu_stars=(0.10, 0.13),
    job_params={
        "q2r_job": {"input_data": {"input": {"zasr": "simple"}}},
        "matdyn_job": {"input_data": {"input": {
            "asr": "simple", "nk1": 32, "nk2": 32, "nk3": 32, "deltae": 1.0,
        }}},
    },
)
# Plain-Python execution; with a workflow engine, consume this after execution.
with open("epc_results.json", "w") as stream:
    json.dump(results["epc_results"], stream, indent=2, allow_nan=False)
```

The flow supports the normal `job_params` and `job_decorators` customization for
`q2r_job`, `matdyn_job` and `epc_analysis_job`. Each solver uses a private staged
copy of its inputs; it does not decompress or modify the original directory.
The source directory must be accessible on the execution worker.

## Supported files and scope

This first implementation supports QE's **interpolated EPC** path (`la2f`), with
plain dynamical matrices `matdyn0`, `matdyn1`, … and the matching `elph_dir`
containing `a2Fq2r.*.*` and `elph.inp_lambda.*`. Gzip files are supported.
`q2r_job`'s `fildyn` setting can select another matrix basename, such as `qe.dyn`.
XML dynamical matrices and other EPC calculation formats are not implemented.

The flow validates the DFPT q-point/broadening series and q2r EPC transforms,
then stages all required matrices, force constants and `elph_dir` internally.
Users do not need to list each filename. The standalone q2r/matdyn recipes retain
their existing `copy_files` interface and additionally accept `source_directory`.

## Scientific result

`epc_results` is a JSON-serializable dictionary with:

- All signed total/mode `alpha2F` arrays, frequency units, broadening indices and
  widths in Ry, mode-integrated lambdas, lambda, omega_log_K and omega_2_K.
- `Tc` records with explicit `method`, `mu_star`, `Tc_K`, `status` and `reason`.
  Methods are `modified_mcmillan_omega_log` (omega_log/1.2 prefactor) and
  `allen_dynes_f1_f2` (the same expression with f1/f2 corrections).
- Native matdyn q coordinates, optional q weights, mode frequencies (cm^-1), and
  phonon DOS/atom projections with native density units. No imaginary frequencies
  or negative spectral values are clipped.
- Optional QE q-sum lambda/omega_log output, kept separate from spectral
  integrals and the a2F footer lambda; these quantities need not coincide.
- Diagnostics, source-file hashes (of stored bytes, including gzip), and the
  q2r/matdyn settings and input manifests.

Negative/nonpositive-frequency spectral rows remain in the arrays. Moments use
strictly positive frequencies and signed spectral amplitudes. Invalid moments or
formula denominators yield null Tc and an explicit reason, not a zero sentinel.
Large lambda is not automatically rejected. A missing or malformed output is
reported as incomplete/invalid; it is not interpreted as zero coupling.

Every available `a2F.dosN` is included. Expected broadening identities come from
DFPT metadata; missing spectra are represented explicitly. The standalone
analyzer can operate without those identities, but then cannot detect a missing
tail of the broadening series and records that limitation.

The default imaginary-mode tolerance is 10 cm^-1, configurable through
`imaginary_tolerance_cm1`. Passing this sampled-frequency check does not establish
thermodynamic stability, EPC convergence or experimental Tc accuracy. These are
**harmonic isotropic formula estimates, not numerical Eliashberg solutions**.

## Offline use and numerical reference

```python
from quacc.calculators.espresso.epc import analyze_epc_directory

scientific = analyze_epc_directory(
    "/path/to/saved/matdyn",
    mu_stars=(0.10,),
    broadenings_Ry=[0.005 * i for i in range(1, 11)],
)
```

`read_matdyn_frequencies`, `read_phonon_dos`, `read_a2f` and `analyze_a2f` are
independently testable without executables. A missing a2F unit must be supplied
explicitly; an explicit/header disagreement raises an error. Spectral amplitudes
retain the native QE normalization when declaring frequency units.

The equations and q2r/matdyn sequence follow
[hyllios/utils autoph](https://github.com/hyllios/utils/tree/4e86e82ae2eec3762f66e6228a70f67b69570789/autoph),
particularly `run_ph.py` and `allen_dynes.py`. The reference's low-lambda zero
fallback and clipping of an invalid second moment are deliberately not used;
unavailable formula results remain distinguishable from physical zero.

Tests use gzip-compressed saved MgB2 q4 spectra and check all ten broadenings and
three mu values against the saved method-matched estimates (relative tolerance
2e-7), along with units, signed data, missing files, staging, customization and
strict JSON serialization. These parser tests do not launch QE.

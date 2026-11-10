# Electron-phonon postprocessing

[`epc_postprocess_flow`][quacc.recipes.espresso.epc.epc_postprocess_flow] takes a
completed QE interpolated-EPC (`electron_phonon='interpolated'`) DFPT directory
and runs q2r, matdyn and scientific analysis using the existing solver jobs.

```python
from quacc.recipes.espresso.epc import epc_postprocess_flow

output = epc_postprocess_flow(
    "/path/to/completed/dfpt",
    mu_stars=(0.10, 0.13),
    job_params={
        "matdyn_job": {
            "input_data": {"input": {"nk1": 16, "nk2": 16, "nk3": 16, "ndos": 200}}
        }
    },
)
scientific = output["epc_results"]
```

Use the usual `job_params` and `job_decorators` for `q2r_job`, `matdyn_job` and
`epc_analysis_job`. The heterogeneous jobs do not support `job_params['all']`.
Nested `input_data['input']` is required for EPC controls. `la2f` must be a Python
boolean. Unique per-job result directories (`CREATE_UNIQUE_DIR=True`) are required.

Sources contain plain `matdyn0..N` matrices and `elph_dir`; gzip is supported.
Set q2r's `fildyn` to select another basename. The QE 7.5 file sets
`a2Fq2r.51..50+n.q` and `a2Fmatdyn.61..60+n` are validated internally. Full-precision
widths come from their headers; `el_ph_nsigma` is inferred, with conflicting
explicit counts rejected. XML matrices and other EPC formats are unsupported.
Read-only sources are supported by making only the private staged EPC tree writable.
Ambiguous plain/gzip pairs and extra transfers overlapping validated inputs are rejected.

The typed `EPCResultSchema` contains every broadening, signed total/mode spectra,
mode lambdas, native frequencies/DOS, moments in K, diagnostics and provenance.
`TcEstimateSchema` records explicit mu-star and either `modified_mcmillan_omega_log`
or `allen_dynes_f1_f2`. The latter additionally needs a valid second moment.
Unavailable estimates are null with reasons. Sampled instability and large lambda
are flagged, not silently clipped or rejected. A negative lambda fraction below
1e-6 is retained diagnostically without changing the status by itself.

## Imaginary phonons and diagnostic Tc

The `imaginary_mode_policy` option controls Tc evaluation when sampled imaginary
modes remain after the Gamma-point allowance, or stability cannot be assessed:

| Policy | Behaviour |
| --- | --- |
| `"warn"` (default) | Retain finite formula values with `status="diagnostic"`, an explicit reason and structured warnings. |
| `"skip_tc"` | Retain spectra and moments; do not evaluate either Tc formula. Tc entries are null with `status="unavailable"` and an explicit reason. |
| `"raise"` | Raise `ValueError` in the analysis job before Tc evaluation. Completed q2r/matdyn files remain available. |

For example, to suppress Tc for imaginary or unassessed phonons:

```python
output = epc_postprocess_flow(
    "/path/to/completed/dfpt",
    imaginary_mode_policy="skip_tc",
    gamma_imaginary_tolerance_cm1=45.0,
)
```

The same option is available on `epc_analysis_job` and `analyze_epc_directory`,
and can be customized through `job_params["epc_analysis_job"]`. Unsupported
policies are rejected before scheduling solver jobs.

The criterion is Gamma-aware, after conversion to cm⁻¹:

- At Gamma, up to three negative modes strictly between
  `-gamma_imaginary_tolerance_cm1` and zero are tolerated. The default allowance
  is 45 cm⁻¹. A mode at −45 cm⁻¹ or more negative is not exempted; endpoint
  roundoff from unit conversion is handled explicitly.
- If more than three Gamma modes meet that condition, the excess still triggers
  the policy. For reproducible bookkeeping, the three closest to zero are marked
  tolerated. This count-based heuristic does not identify acoustic eigenvectors.
- Every negative frequency away from Gamma triggers the policy, regardless of
  its magnitude. There is no uniform non-Gamma tolerance.

Gamma is identified as q=(0,0,0) in the parsed Cartesian coordinates, not by
reducing coordinates modulo integers. A nonzero point near Gamma is not exempted.
Every negative mode, including tolerated Gamma modes, remains recorded. The
Gamma allowance is configurable; it is not a universal physical stability rule.

`phonon_stability` distinguishes `"unstable"`, `"no_sampled_instability"` and
`"unknown"`. Missing, unreadable or unrepresentable frequency data give unknown
stability. The instability record retains the minimum frequency, negative,
tolerated and untolerated counts, and all negative modes with one-based
q-point/mode indices, coordinates and tolerance labels. `unstable_modes` contains
the untolerated subset. Passing this sampled heuristic does not establish
stability throughout the Brillouin zone.

Diagnostic values can be retained for selecting subsequent anharmonic studies.
They describe the supplied positive-frequency spectral data; QE may have zeroed
imaginary-mode linewidth contributions. They are not anharmonic Tc predictions
or bounds on an SSCHA-corrected Tc. Invalid spectra or formula domains always
retain null values and their original failure reasons, under every policy.

## Spectral moments and provenance

Moments use Simpson integration over positive frequencies. For an implementation
of the spectral-moment and Tc estimation methods, see Gao et al.,
[*Nat Commun* **16**, 8253 (2025)](https://doi.org/10.1038/s41467-025-63702-w).
QE's footer instead uses its bin-centred midpoint sum, including signed frequency
bins. Both comparisons and the distinct QE q-sum are retained; disagreement does
not establish convergence. Rounded q-sum widths are matched at 5e-5 Ry precision;
missing/malformed auxiliary rows cannot supply unknown spectrum identities.

The `settings` field retains the written QE input texts. Provenance includes their
hashes and stored/uncompressed hashes of scientific inputs and outputs. Parsers
and `analyze_epc_directory` also operate directly on saved files. These harmonic,
isotropic formula estimates are not numerical Eliashberg solutions, physical
stability checks or experimental Tc validation.

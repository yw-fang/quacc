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

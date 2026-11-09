# Electron–phonon superconductivity reports

This branch adds `quacc.recipes.espresso.epc` and `quacc.schemas.epc`, reusing
the existing Espresso relaxation, static, NSCF, bands, DOS, projected DOS,
phonon, q2r and matdyn recipes. Install the `epc` and `jobflow` extras.

The project supplies pseudopotentials, cutoffs, convergence criteria, pressure,
k/q meshes, broadening, mu*, paths and the selected spectrum. The library contains
no Hyperion paths, material-specific thresholds or Discord credentials.

`superconductivity_flow(...)` constructs a Jobflow graph for two variable-cell
relaxations, dense `la2f` SCF, coarse SCF with a verified dense eigenvalue copy,
DFPT/EPC, q2r, matdyn, and electronic/phonon reporting. `epc_analysis_flow(...)`
continues an already completed calculation without repeating SCF/DFPT. Cluster
resources and execution belong to a deployment client, not these recipes.

Reports save electronic bands with total/element DOS relative to one recorded
SCF EF; phonon dispersion with DOS, selected alpha2F and cumulative lambda;
PNG/PDF files; raw numeric NPZ arrays; selected-spectrum copy/hash; scalar
McMillan/Allen–Dynes values and units. The final completion marker records
SHA256 checksums after files are flushed. Existing completed reports are reused,
not overwritten. An external service can verify the marker and notify users.

`selected_spectrum` accepts QE `a2F.dosN` with frequencies in Ry. It preserves
signed spectral weight, reports negative contributions, and checks that the
cumulative Simpson lambda endpoint agrees with scalar Simpson integration.
Omega_log and the root second moment are in **K**. Modified McMillan uses the
omega_log/1.2 prefactor; Allen–Dynes includes f1/f2. Undefined denominators remain
undefined. There is no Tc announcement threshold and no rejection based solely
on large lambda. Imaginary modes below −10 cm−1 flag `UNSTABLE`; this tolerance
distinguishes small numerical acoustic residuals, not physical metastability.

The plotting parser currently supports nonmagnetic, scalar-relativistic QE
outputs. Spin-resolved data raise an explicit error. Atomic projections need not
sum exactly to total DOS because the atomic basis can be incomplete. No
anisotropic, multiband or anharmonic Eliashberg treatment is implied.

Validation: analytical Einstein-mode/moments, large-lambda preservation, signed
spectra, independent unit conversion; real Hyperion continuation of MgB2 with
band output recovery, NSCF, total/element DOS, phonon path, and plotting. The fresh
13-stage factory was graph/serialization checked, but has not been rerun as a
second expensive benchmark. Its constituent relaxation/DFPT sequence follows the
previous completed benchmark. Further generalized-material validation and
upstream API review remain necessary.

Numerical reference:
[hyllios/utils autoph](https://github.com/hyllios/utils/tree/4e86e82ae2eec3762f66e6228a70f67b69570789/autoph).
This implementation does not execute its bot, import its credentials, apply its
20 K notification threshold, or require its external Eliashberg solver.

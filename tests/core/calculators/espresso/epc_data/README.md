# QE 7.5 electron-phonon test data

These are small software-regression fixtures, **not converged superconductivity
benchmarks**. Reference moments and Tc values are written inline in `test_epc.py`.
They were computed using an independent implementation of the spectral-moment
and Tc estimation methods implemented in Gao et al.,
[*Nat Commun* **16**, 8253 (2025)](https://doi.org/10.1038/s41467-025-63702-w).
These fixture-specific values are not results reported in that paper. The 2e-7
relative tolerance checks agreement between implementations; it is not an accuracy
bound on quadrature, convergence or Tc. All gzip mtimes are zero.

## `pb/`: one-atom fcc Pb, strong coupling

Saved matdyn outputs from the completed DFPT source in
`tests/core/recipes/espresso_recipes/data/epc_test`. That source is used by the
native q2r/matdyn test and by the staging tests.

- `a2F.dos1.gz` is the complete 120-bin spectrum at 0.0025 Ry, interpolated on an
  8³ mesh. All three mode columns and the signed rows are kept. λ ≈ 7.9, so the
  Allen–Dynes f1·f2 corrections are significant.
- `lambda.gz` contains both native q-sum rows.
- `matdyn.freq` contains the first two q-point blocks, with `nks` adjusted to 2.
- `matdyn.dos` contains the header and the first three rows. It is not a
  normalized DOS.

The DFPT source has 17 files and is about 5 KB compressed. It covers four
irreducible q-points of a 3×3×3 grid, and two broadenings: 0.0025 and 0.005 Ry.
QE prints 0.0025 as `0.003` in `elph.inp_lambda`, which exercises the
full-precision broadening path.

The data was generated with the following settings:

- Potential: scalar-relativistic PZ `pb_s.UPF` (sha256 `24fe8cc9…d27af`). SOC is
  omitted.
- Structure and cutoffs: a = 9.2225583816 bohr; 30/120 Ry cutoffs.
- Electronic grids: unshifted 18³ (fine) and 9³ (coarse) grids, with cold smearing
  of 0.05 Ry.
- Convergence thresholds: 1e-10 for SCF and 1e-12 for DFPT.

## `al/`: one-atom fcc Al, weak coupling

`a2F.dos10.gz` is the complete 50-bin, three-mode spectrum at 0.05 Ry broadening.
It comes from QE's official `PHonon/examples/example03`, run with
`Al.pz-vbc.UPF` (md5 `614279c8…d420`). The q-mesh was 4×4×4 and the matdyn mesh
10³.

λ ≈ 0.37, so Allen–Dynes stays close to McMillan and the spectrum checks the
weak-coupling regime. The total α²F is nonnegative, but some mode columns have
negative values.

Native QE runs and QuAcc runs with the same executable produced identical spectra.

## Not bundled

The following are not included: pseudopotentials, wavefunctions, full DOS and
frequency files, and the SCF/DFPT inputs. Edge cases (malformed, ambiguous or
missing data, units, formula domains and numerical limits) are covered by small
synthetic tables written inside the tests.

# Minimal EPC fixtures

`a2F.dos5.gz` is one complete MgB2 QE 7.5 PBEsol q4 spectrum at a Gaussian
broadening of 0.025 Ry. All 812 data rows and nine mode columns are preserved
for numerical quadrature; thinning this spectrum would change the regression.
`expected.json` contains independently saved estimates at three mu-star values.
The relative numerical tolerance is 2e-7 for lambda, frequency moments and both
Tc methods. The gzip file is deterministic (mtime=0).

`matdyn.freq` contains two native q-point blocks, with the declared `nks`
changed to 2. `matdyn.dos` contains the header and three native rows. These
excerpts test parsing only; the DOS excerpt is not a normalized full spectrum.
`provenance.json` records uncompressed fixture hashes and full-source hashes.

Small hand-written tables in the tests cover signed data, format failures,
broadening discovery, missing series and file staging. They are synthetic, not
additional calculated materials. The full ten-broadening MgB2 regression is
validation evidence rather than a bundled unit-test data set.

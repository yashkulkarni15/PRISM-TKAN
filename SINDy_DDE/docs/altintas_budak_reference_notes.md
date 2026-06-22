# Altintas-Budak Verifier Notes

This slice implements a zero-order milling stability-lobe scaffold for PRI2M.
The exact textbook benchmark constants were not present locally, so the
`textbook-smoke` verifier mode uses a placeholder reference case and marks
strict textbook validation as failed until those constants are supplied.

Core assumptions:

- Structural dynamics use one SDOF receptance per axis:
  `G(jw) = (1/k) / (1 - (w/wn)^2 + j 2 zeta w/wn)`.
- Directional factors use the unnormalized Altintas-Budak engaged-arc
  integrals for `alpha_xx`, `alpha_xy`, `alpha_yx`, and `alpha_yy`.
- Numeric `cut_dir` values are interpreted as `1 = up/conventional`,
  `2 = down/climb`, and `0 = slot/full`.
- Stability candidates use eigenvalues of `inv(G A)` with
  `blim = -2 pi Lambda_R (1 + kappa^2) / (N Kt)`, where
  `kappa = Lambda_I / Lambda_R`.
- Candidate spindle speeds use
  `rpm = 60 fc / (N (lobe_order + epsilon / (2 pi)))`.

Dataset validation status:

- `scripts/verify_altintas_budak.py --mode dataset-blim` reads only the
  lightweight CSV feature table and does not open raw H5 files or pickles.
- The current scaffold reuses `ktx` as y-axis stiffness when a separate y-axis
  stiffness is absent and assumes full slot engagement when radial immersion is
  not available.
- Because the local dataset generator constants and engagement convention are
  not fully documented in this repo, dataset comparisons are calibration
  scaffolds rather than authoritative acceptance checks unless run with
  `--strict` and an agreed tolerance.

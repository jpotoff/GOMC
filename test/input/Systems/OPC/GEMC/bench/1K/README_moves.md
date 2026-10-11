# OPC GEMC move-coverage configs

Short (1000-step, ConsoleFreq 1) variants of `GEMC_prod.conf`, each isolating
one move type so that a mismatch names the code path that broke rather than
just saying "the trajectory diverged".

They exist because `GEMC_test.conf` is `VolFreq 1.0` -- 100% volume moves --
so for a long time it was the only OPC test being run and it exercises only
`BoxInterTemplate` and `VirialCalcTemplate`. The entire CBMC path was
untested.

## Which move reaches which kernel

Measured by mutation: perturbing a kernel by 1e-6 and seeing which configs'
energies move.

| config | moves | kernel exercised | ENER_ lines moved |
|---|---|---|---|
| `GEMC_test.conf` | VolFreq 1.0 | `BoxInterTemplate`, `VirialCalcTemplate` | - |
| `GEMC_dis.conf` | DisFreq 1.0 | `MoleculeInterTemplate` | 0 |
| `GEMC_rot.conf` | RotFreq 1.0 | `MoleculeInterTemplate` | 0 |
| `GEMC_swap.conf` | SwapFreq 1.0 | **`ParticleInterTemplate`** (CBMC) | 1923 / 2002 |
| `GEMC_regrowth.conf` | RegrowthFreq 1.0 | **`ParticleInterTemplate`** (CBMC) | 1477 / 2002 |
| `GEMC_allmoves.conf` | production mix | all of the above | 993 / 2002 |
| `GEMC_allmoves_notab.conf` | production mix | as above, exact-erfc fallback | 136 / 2002 |

**Displacement and rotation do not reach `ParticleInterTemplate`.** They move a
molecule rigidly and go through `MoleculeInterTemplate`; `ParticleInter` is the
per-trial-position CBMC kernel, reached only by swap and regrowth. If you are
changing `ParticleInter`, `GEMC_swap.conf` and `GEMC_regrowth.conf` are the
tests that matter.

## The tabulated real-space kernel

All of these except `GEMC_allmoves_notab.conf` set
`EwaldTabulateRealSpace True`, matching `GEMC_prod.conf`. This is load bearing:
the branch-free Coulomb path in `CalculateEnergy` is gated on
`FFAdapter::TableRange()`, which returns false when the table is off, so
without the flag that path is never reached and a test of it passes
vacuously. `GEMC_allmoves_notab.conf` keeps coverage of the exact-erfc
fallback.

## Use

Copy the whole `bench/` directory into a scratch dir (copy, do not symlink),
run, and compare `ENER_`/`STAT_` lines against a reference build. Confirm a
new branch is actually reached by mutating it and checking that the energies
move -- a passing diff against an unreached path means nothing.

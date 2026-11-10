# Population (SMC) CBMC Growth for GOMC

## Status (2026-11-10): implemented, awaiting end-to-end exactness test

**Code:**
- [src/cbmc/SMCGrowth.h](src/cbmc/SMCGrowth.h): the generic engine
  (sequential Monte Carlo with conditional SMC for the old configuration).
- `DCGraph::BuildSMC`: the GOMC particle (`GraphParticle`) and the `Build`
  path. It reuses the `DCLinkedHedron` trial interface written for recoil
  growth (`GenerateTrials`, `ActualTrial`, `Commit`, `CacheOldBonds`).
- [test/src/SMCGrowthTest.cpp](test/src/SMCGrowthTest.cpp): exactness and
  unbiasedness tests on a toy chain, plus a disabled equal-cost benchmark
  (`--gtest_also_run_disabled_tests`).
- Instrumented builds only (`GOMC_INSTRUMENT`): `CBMC::ProbeSMC`, driven from
  `MoleculeTransfer` by the `GOMC_SMC_PROBE*` environment variables (below).

**Behaviour:**
- **Off by default.** With `CBMC_SMC false` the trajectory is bit-identical
  to before.
- **Where it applies:** every move that calls `DCGraph::Build`, which is
  swap (GEMC and GCMC, `MoleculeTransfer`) and intra-swap (`IntraSwap`), for
  acyclic molecules with at least 3 atoms.
- **Where it does not apply** (standard CBMC is used, with no error):
  - ring molecules (`DCCyclic`); a kind whose graph has an edge that is not a
    `DCLinkedHedron` gets a startup warning;
  - moves with a cavity, a fixed COM or a rotated backbone (MEMC, targeted
    swap);
  - regrowth and crankshaft;
  - molecules with 1 or 2 atoms, which do not use `DCGraph`.
- **Settings it overrides** for the growth it handles. Both are reported at
  startup.
  - `CBMC_First` and `CBMC_Nth`: every growth step, including the start
    node, uses `CBMC_SMCTrials` LJ trials per particle.
  - `CBMC_LookAhead*`: the look-ahead is not used.
- **Settings it keeps:** `CBMC_Ang` and `CBMC_Dih` still set the bonded
  trials inside each step.
- **Cannot be combined** with `CBMC_Recoil`; that is an input error.

## Keywords

| Keyword | Meaning | Default |
|---|---|---|
| `CBMC_SMC` | enable SMC (population) growth for `DCGraph::Build` (swap, intra-swap) | `false` |
| `CBMC_SMCParticles` | number of particles N | 300 |
| `CBMC_SMCTrials` | LJ trials per particle per growth step k | 1 |

Example:

```
CBMC_SMC           true
CBMC_SMCParticles  300
CBMC_SMCTrials     1
```

**Choosing N and k.** At a fixed cost N·k per step, put the budget into N:
k = 1 was best in every test.
- On all-atom C16 the median new/old log-weight deficit fell from −6.9 to
  −2.9 to −1.0 at N = 100, 300 and 1000.
- At the same total number of trials, CBMC stayed at about −100.

The deficit shrinks roughly as 1/N. N must grow about linearly with chain
length, so the cost per growth is about n².

## Method

**Growth.** N partial chains (particles) grow side by side.
- At step i each particle draws k trials and picks one with probability
  proportional to its weight, exactly as single-chain CBMC does.
- Its step weight is W = mean trial weight, and its running log weight L
  accumulates log W.
- **Resampling:** when the effective sample size falls below τN, the
  population is resampled multinomially in proportion to exp(L).
  - GOMC uses τ = 2, which resamples every step.
  - The mean of exp(L) is folded into the estimate and L is reset.
- At the end one particle is chosen in proportion to exp(L).
- The estimate Ẑ = Π over epochs of mean exp(L) is an unbiased estimator of
  the Rosenbluth normalisation (Del Moral 2004).
- Special cases: N = 1 is standard CBMC with k trials; τ = 0 is N
  independent CBMC chains with averaged weights.

**Old configuration (conditional SMC).** This follows Andrieu, Doucet &
Holenstein, J. R. Stat. Soc. B 72, 269 (2010).
- Particle 0 is the old molecule.
- Its trial set at each step is its actual placement plus k − 1 fresh
  trials.
- It always keeps its own lineage through resampling.
- The other N − 1 particles are grown and resampled from all N.

**Acceptance.** The molecule's weight is Ẑ, so the move's usual
W_new / W_old acceptance is unchanged.
- The move is a Gibbs refresh of the auxiliary particles followed by a
  particle independent Metropolis–Hastings step, so it is exact.
- Weighing the old molecule with an unconditional run breaks detailed
  balance. The toy test detects this as a negative control.

**Why it helps.** The gain comes from memory, meaning dead ends that depend on
the earlier part of the chain.
- Resampling discards lineages that are heading into crowded regions before
  they pay for it.
- A memoryless chain gains nothing: SMC and CBMC are then equivalent.

**Known limits:**
- Information that arrives late in the chain needs N ~ 1/p.
- N must grow linearly with chain length.
- Path degeneracy.
- Glassy or very dense states have not been tested.

**Implementation notes:**
- **Edge order:** each `Build` draws a start node and edge order, as standard
  CBMC does, shared by the new and old growths and by all particles.
- **Trial counts:** `DCData::nLJTrialsFirst` and `nLJTrialsNth` are set to k
  for the duration of `BuildSMC` and then restored.
- **Particles are `TrialMol` copies.** Overlap flags are carried across
  copies explicitly, because the `TrialMol` copy constructor resets them.
- **Bond lengths.** The extra trials of the reference particle use the old
  molecule's bond lengths, as GOMC's CBMC old weighing does. New particles
  use equilibrium lengths. The two are identical when bonds are rigid.
- **Threads.** With `+pN`, each step's particles are extended on N OpenMP
  threads.
  - Each extra thread has a worker copy of the molecule's `DCGraph`, with
    its own components, `DCData` scratch and generator. A worker generator
    is reseeded from the main one at every growth.
  - The schedule is static, so a run is reproducible for a fixed thread
    count. With one thread, no extra random numbers are drawn and
    trajectories are identical to the serial code.
  - The energy code keeps one local candidate list per thread. The packed
    box is built before the parallel loop.
  - Resampling and the copying of particles stay serial.
  - Instrumented builds stay serial.
- **Neighbour lists.** For k = 1 a step has 1-2 trial positions, below the
  default threshold for building a local candidate list. SMC lowers the
  threshold to 1, except when the cell stencil covers the box. Copies of
  one parent are grown next to each other and reuse one list.
- **Weights:** they are handed to the move as exp(ln Ẑ), as GOMC's other
  growth methods do. On AA C16 in the liquid ln Ẑ ≈ −12, far from underflow.

## Validation

- **Toy chain** (`SMCGrowthTest`):
  - Ẑ is unbiased against a transfer-matrix ln Z.
  - A Markov chain of SMC moves reproduces full enumeration for 8 (N, k, τ)
    cases, with max |z| < 2.8.
  - The unconditional-old variant fails with max |z| = 194.
- **AA C16 probe** (OpenFF, 907 probes, same environment for new and old):
  - With CBMC the median deficit is −216, −126, −105 and −102 at 30, 100,
    300 and 1000 trials per step.
  - With SMC (k = 1) the median deficit is −40, −6.9, −2.9 and −1.0 at the
    same trial counts.
  - Same-box acceptance proxy: SMC 0.48 at N = 1000 and 0.23 at N = 300;
    CBMC ≤ 1e-3.
- **AA C16 GEMC smoke test** (3000 steps):
  - Swap acceptance: SMC 300×1 gave 45% (box 0) and 33% (box 1), against
    0.33% and 0.34% for CBMC.
  - No drift in box counts or density.
- **Not yet done:**
  - End-to-end exactness of the real move: SMC on vs off on a small UA alkane
    GEMC, comparing densities, counts, energies and vapour pressure.
  - GCMC and intra-swap runs; CUDA build.

## Instrumentation (GOMC_INSTRUMENT builds)

| Variable | Meaning | Default |
|---|---|---|
| `GOMC_SMC_PROBE` | settings to probe, `N:k,N:k,...` (N = 1 is CBMC) | unset (off) |
| `GOMC_SMC_PROBE_EVERY` | probe every n-th transfer move | 1 |
| `GOMC_SMC_PROBE_BOX` | box to probe | 0 |
| `GOMC_SMC_PROBE_OUT` | output file | `smc_probe.dat` |

Each probe removes a random molecule of the box from the cell list. For each
setting it records ln Ẑ_new, ln Ẑ_old and the wall time. Nothing is committed
to the system.

`c16a_cbmc_notes/scripts/smc_probe_report.py` summarises the output:
- median, mean and sd of the deficit;
- failure rate;
- acceptance proxy;
- time per pair.

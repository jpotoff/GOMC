# Speculative Parallel Swap Moves: Design

Status: proposal only. No code has been written. The point of this document
is to decide whether the move is worth building.

## Summary

A molecule-transfer (swap) move is rejected about 99% of the time for
all-atom alkanes, and a rejected move leaves the state unchanged. N swap
attempts drawn from the current state can therefore be computed in parallel,
then accepted or rejected **in order**. Every attempt up to and including the
first acceptance is exactly what a serial run would have done. Attempts after
an acceptance were computed against a state that no longer exists; they are
discarded and recomputed.

The move is exact. Its value depends on two things:

1. **The fraction of run time spent in swaps.** All-atom hexane GCMC on a
   9950X, single thread:

   | Move | Time | Share |
   |---|---|---|
   | Molecule Transfer | 87.6 s | 71.2% |
   | Regrowth | 15.6 s | 12.7% |
   | Displacement + Rotation | 19.7 s | 16.0% |

2. **Whether single-trajectory wall time is the constraint.** For throughput
   across state points, N independent single-thread runs scale almost
   perfectly and no in-run parallelism can beat them.

Expected speedup per trajectory, from Amdahl's law with S = 0.712 and the
measured acceptance of about 1.2%:

| Threads | Wasted work | Speedup | Swaps + regrowth speculative (S = 0.84) |
|---|---|---|---|
| 4 | ~2% | 2.1× | 2.7× |
| 8 | ~4% | 2.6× | 3.8× |
| 16 | ~9% | 2.9× | 4.7× |
| ∞ | — | 3.5× | 6.2× |

The regrowth column assumes regrowth acceptance is also low; see
[Extensions](#extensions).

## Why the current code does not parallelise

Single-thread and multi-thread profiles of the same hexane GCMC run (9950X):

- **1 thread:** branch sampling ≈ 40–45%, intermolecular CBMC energy ≈ 14%,
  intramolecular nonbonded ≈ 8–11%, `MoleculeInter` (displacement/rotation)
  ≈ 11%, reciprocal Ewald ≈ 4%.
- **2 threads:** the worker spends 40% of all samples in
  `kmp_flag_64::wait`, i.e. it is idle about 80% of the time.
- **4 threads:** 66% of all samples are idle workers.

The only parallel region in CBMC growth is the loop over trials in
`ParticleInter`, about 10 trials wide. Since the per-growth-step candidate
list (`CalculateEnergy::BeginLocalCandidates`) that region has little work
left. Everything else in a swap — branch and torsion sampling, intramolecular
energy, the look-ahead — runs on the main thread. More parallelism has to
come from coarser units of work, and a whole swap attempt is the natural one.

## The move

### Semantics

A new move, or a mode of `MoleculeTransfer`, with batch size N
(`SwapBatch N`, default 1):

```
on selection at step s:
  i = 0
  while i < N:
    launch attempts i, i+1, ..., N-1 in parallel, all from the current state
    for j = i, i+1, ..., N-1:          # decide in order
      decide attempt j
      if accepted:
        apply attempt j to the state
        break
    i = j + 1                          # attempts after j are discarded
```

**Exactly N attempts are decided per selection.** Attempts discarded after
an acceptance are recomputed from the new state in the next round. The move
is the composition of N swap kernels, each of which satisfies detailed
balance, so it preserves the stationary distribution, like any fixed
sequence of valid moves.

### A tempting variant that is wrong

"Attempt swaps until one is accepted, up to N" stops on an outcome. Without
counting the rejected attempts as time spent in the current state, that is
the jump chain of the Markov chain. In the large-N limit it samples
π(x)(1 − r(x)) rather than π(x), where r(x) is the rejection probability.
States where insertion is hard would be under-weighted. **The number of
decided attempts per selection must not depend on their outcomes.** It is
fixed at N here.

Pre-drawing the move types of the next few steps and running consecutive
swap steps speculatively is exact as well, but it finds little work: with
`SwapFreq` 0.2, runs of consecutive swap steps average 1.25. The fixed-N
composite move is the only version that produces enough parallel work.

### Random numbers

Each attempt draws from its own stream, keyed by (seed, step, attempt index
j). A recomputed attempt j uses the same key as the discarded speculative
one. That is legitimate: no decision ever used the discarded results. It
gives two useful properties:

- **The output is independent of the thread count.** At fixed N, T = 1 and
  T = 16 produce identical trajectories, which is the main correctness test
  for the parallel machinery.
- **The parallel execution matches the serial semantics exactly.** It equals
  "decide N attempts in order, attempt j using stream (seed, s, j)".

CBMC code calls `PRNG` (Mersenne Twister) through `DCData::prng`. Each
worker needs its own `PRNG`, reseeded per attempt from a hash of (seed, s,
j). Reseeding costs microseconds, against milliseconds per attempt.
Alternatively, back `PRNG` by the existing `Random123Wrapper`, which is
counter-based and keyed by step already.

With N = 1 and a "legacy stream" switch that uses the main `PRNG`, the
refactored move should reproduce the current trajectory bit for bit. That is
the first validation gate.

### Molecule choice and acceptance

The attempt chooses its molecule and boxes (`PickMolAndBoxPair2`) at proposal
time, from the state its round started from. For every decided attempt that
state is the one in effect at its decision: any attempt computed against an
older state was discarded. The same holds for `GetCoeff()` (molecule counts
and volumes) and for `MoleculeTailChange`. No correction is needed.

## What has to change in GOMC

The current swap mutates shared state in several places. The work is mostly
making an attempt **side-effect free until it is accepted**.

### 1. Removing the molecule from its box

`MoleculeTransfer::Transform` calls `cellList.RemoveMol(molIndex, ...)`
before growth, so that the molecule's old configuration does not interact
with itself. `Accept` adds it back. That mutates the cell list and bumps its
version, which also invalidates the box-pack cache. Two attempts on
different molecules cannot both do it.

**Change:** exclude the molecule by index in the intermolecular kernels.

- `ParticleInterTemplate`, packed path: the packed arrays already carry
  `pMol`; add `mol != excluded` to the pass-2 compaction mask.
- `ParticleInterTemplate`, cell path: skip `particleMol[j] == excluded` in
  the gathered loop.
- `BuildLocalCandidates`: skip the excluded molecule when building the
  list, so the kernel needs no test at all.

The excluded molecule is a per-call argument (or part of the attempt
context), never a member. Skipping it visits the same pairs, in the same
order, as removing it from the cell list. The energies are therefore
bit-identical to the current move, which the N = 1 gate checks.

### 2. CBMC scratch and state

Each `MoleculeKind` owns one `cbmc::CBMC *builder` (`MakeCBMC`). The builder
and its `DCData` hold all growth scratch:

- `DCData`: `multiPositions`, `inter`, `real`, `ljWeights`, `bonded`,
  `nonbonded*`, `overlap`, angle and torsion arrays, `bendTables`;
- the DC objects themselves: `DCLinkedHedron` has `laPos`, `trialTheta`,
  bond lengths, `laV`; `DCHedron` has the branch sampler;
- references to `sys.prng` and `sys.calcEnergy`.

**Change:** one builder per (worker, molecule kind), each with its own
`DCData`, its own `PRNG`, and a pointer to its worker's energy context (see
below). `MoleculeKind::Build` gains an overload that takes the builder to
use. Builders are created once at start-up. Memory is small: the scratch is
sized by trial counts and molecule size.

`cbmc::TrialMol` objects are already per move; they become per attempt.

### 3. Mutable caches in `CalculateEnergy`

- `boxPack*` (the whole-box pack) is rebuilt lazily when the cell list
  version changes. During a batch the cell list no longer changes (see 1),
  so build both boxes' packs **once before launching a round**. The attempts
  then only read them.
- `local*` (the per-growth-step candidate list) is written by every growth
  step. It moves into a per-worker context passed down through
  `ParticleInter`.

Concretely, an `EnergyContext` struct (candidate-list buffers, excluded
molecule index) is owned by each worker, and `ParticleInter` takes a
`const EnergyContext *` (nullptr = current behaviour).

### 4. Ewald

For each attempt, `SwapDestRecip` and `SwapSourceRecip` write the trial
structure factors into the shared `sumRnew[box]` / `sumInew[box]`, computed
from the read-only `sumRref` / `sumIref`. On acceptance, `UpdateRecip(box)`
swaps the `sumRnew` and `sumRref` pointers.

**Change:** per-worker `sumRnew` / `sumInew` buffers for both boxes. On
acceptance, swap the accepted attempt's buffer into `sumRref`, which is the
same pointer swap as today. `SwapCorrection` and `SwapSelf` are already pure.

The inner `omp parallel for` over k-vectors in `SwapDestRecip` must not
spawn threads inside an attempt. With OpenMP's default of one active level,
nested regions run on one thread, so this needs no code change, only a check.

**Out of scope for the first version:** `EwaldCached` (`CachedFourier true`,
per-molecule cos/sin caches mutated by `RestoreMol`) and the CUDA build. The
speculative mode refuses to start with either.

### 5. Statistics, output and instrumentation

- `moveSetRef.Update(MOL_TRANSFER, ...)` is called once per decided attempt,
  in order. Discarded attempts are not counted.
- Energy bookkeeping (`sysPotRef`), coordinates, COM, `molLookRef`, cell
  list and velocities are updated only in the in-order decision stage, as
  in `Accept` today.
- `instr::Collector` (`GOMC_INSTR`) is a global singleton that CBMC writes
  to during growth. First version: refuse `Instrument true` together with
  `SwapBatch > 1`. Later: buffer instrumentation records per attempt and
  commit only the decided ones, so diagnostics describe the chain and not
  the discarded work.
- Move timing (`moveTime`) records wall time for the composite move. Per
  attempt it is not comparable to N = 1, so report attempts per second as
  well.

### 6. Scheduling

Attempts differ a lot in cost. An attempt whose first bead overlaps
everywhere is rejected in microseconds; a full growth takes milliseconds. A
static split across threads would leave threads idle. Use dynamic
scheduling (`omp parallel for schedule(dynamic,1)` over attempts, or OpenMP
tasks). Keep one persistent parallel region per round, not one per attempt.

### Not touched

Displacement, rotation, volume and multiparticle moves, all other CBMC moves
(regrowth, crankshaft, MEMC, NeMTMC, targeted swap), and the GPU code.

## Validation plan

Each gate must pass before the next.

1. **Refactor, N = 1, legacy streams.** Exclusion by index, per-worker
   builders and contexts, private Ewald buffers, all exercised with one
   attempt per move and the main `PRNG`. Hexane GCMC and ethanol GEMC must
   reproduce the current trajectories bit for bit (final energies and
   molecule counts to printed precision), as in the candidate-list change.
2. **Determinism, N = 8, per-attempt streams.** Identical output for
   T = 1, 2, 4, 8 threads.
3. **Statistics.** Hexane GCMC P(N) and mean energy, and ethanol GEMC
   coexistence densities: N = 1 against N = 8, from independent seeds,
   agreeing within error bars. A debug-build check re-evaluates each
   accepted attempt serially from the pre-move state and asserts identical
   weights.
4. **Performance.** Attempts per second and decided swaps per wall-second,
   against threads, on the 9950X. Compare with the Amdahl estimate above.

## Cost and risk

- **Size of the change:** moderate to large. It touches `MoleculeTransfer`,
  `MoleculeKind` (builder per worker), `DCData` and `MakeCBMC`,
  `CalculateEnergy` (context parameter, exclusion mask), `Ewald` (private
  buffers), `PRNG` (per-attempt streams), config parsing (`SwapBatch`), and
  instrumentation gating. Most of it is plumbing; the move logic itself is
  short.
- **Main risk:** hidden shared state in CBMC. Any member written during
  growth that is not replicated per worker is a data race that produces
  wrong weights, not a crash. Gates 1 and 2 are designed to catch exactly
  this, and running under ThreadSanitizer once is cheap insurance.
- **Maintenance:** every future CBMC change must keep growth free of shared
  writes. The per-worker context makes that explicit, which arguably also
  helps.
- **The plumbing is reusable.** Side-effect-free attempts with private
  contexts are also what speculative regrowth, or batched Widom insertions,
  would need.

## Extensions

- **Regrowth** (12.7% of the hexane run) uses the same CBMC machinery. It
  benefits only if its acceptance is low: with acceptance a and batch N the
  wasted work grows as about aN²/2, so at a ≳ 20% batches are cut short too
  often to help. Check `% Accepted Regrowth` before deciding.
- **GEMC** works unchanged: the attempt chooses the box pair, and both boxes'
  structure factors are private per attempt.
- **Widom insertion** (test-particle μ) has no accept/reject chain at all:
  every insertion is independent and nothing is discarded. The same
  per-worker plumbing gives it perfect parallelism.

## Alternatives

| Option | Gain per trajectory | Throughput | Effort |
|---|---|---|---|
| Independent single-thread runs | 1× | ~T× | none |
| Speculative swaps (this document) | 2–3× at 4–16 threads | below independent runs | moderate–large |
| Speed up branch sampling (`BranchSampler`: fewer rejections, no repeated trig, no `erf` per try) | maybe 1.2–1.5×, serial | multiplies with everything else | small–moderate |
| Dual-cutoff CBMC | ≤ ~1.07× (acts on the 14% intermolecular share only) | multiplies | moderate |

The `BranchSampler` work is independent of this proposal and speeds up every
mode, including independent runs. It is the cheaper first step whichever way
this decision goes.

## Decision criteria

Build it if all of the following hold:

1. Wall time of a single trajectory is a real constraint: long
   equilibrations, single expensive state points, or cluster allocations
   with a fixed number of cores per job.
2. Swaps stay at 60% or more of run time after the `BranchSampler`
   improvements. Those shrink every CBMC move about equally, so the share
   should hold, but it should be re-measured.
3. Ideally, regrowth acceptance is low enough to include regrowth, which
   raises the ceiling from 3.5× to 6.2×.

If the work is mostly many independent state points, run them as
independent single-thread jobs and spend the effort on `BranchSampler`
instead.

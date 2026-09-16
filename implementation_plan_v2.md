# GOMC GEMC Performance Plan (profile-driven)

Supersedes `implementation_plan.md`, which was written against pre-`templates` code and
whose line references and code sketches no longer match the source.

## The profile

Merging the OpenMP-worker and main-thread rows for the same symbol:

| Cost centre | worker | main | **total** |
|---|---:|---:|---:|
| `BoxInterTemplate<BoxDimensions>` | 22.26 | 8.23 | **30.5%** |
| `__libm_erfc_l9` | 14.58 | 4.12 | **18.7%** |
| `kmp_flag_64::wait` (OpenMP idle/barrier) | 10.42 | — | **10.4%** |
| `ParticleInterTemplate<BoxDimensions>` | 7.72 | 3.04 | **10.8%** |
| `Ewald::BoxReciprocalSetup` | 7.01 | 2.34 | **9.4%** |
| `FFParticle::CalcCoulomb` | 4.51 | 2.81 | **7.3%** |
| `MoleculeInterTemplate` | 1.96 | — | 2.0% |
| `FFParticle::CalcEn` (4-arg + 2-arg) | 2.27 | — | 2.3% |
| `VirialCalcTemplate` | 1.34 | — | 1.3% |
| | | | **92.7%** |

Three readings drive everything below.

**1. Real-space electrostatics is 26% of runtime.** `erfc` (18.7%) plus `CalcCoulomb`
(7.3%) — and `erfc` alone is the second-largest single symbol in the run.

**2. The forcefield kernels are not being inlined.** `FFParticle::CalcCoulomb` and
`FFParticle::CalcEn` appear as *standalone symbols* rather than folded into
`BoxInterTemplate` / `ParticleInterTemplate`. That is direct evidence the virtual
dispatch through `forcefield.particles->` is blocking inlining, which in turn blocks
vectorisation of the pair loop. The merged templates parameterise on `BoxType` only
(devirtualising `InRcut`); the forcefield type is still resolved through the vtable.

**3. Threads spend 10.4% waiting.** Not computing — idling at barriers.

---

## Phase 1 — Tabulate the real-space Ewald kernel

**Target: the 18.7% in `erfc`, plus part of the 7.3% in `CalcCoulomb`.**

### Why the previous attempt was reverted

`b77961b2` added `num::erfc_cody`: a degree-25 Chebyshev polynomial, then

```cpp
return p * std::exp(-x * x);
```

`279b59ec` restructured the polynomial into Estrin form to shorten the dependency chain,
and `da7e8498` reverted the whole thing.

Per the author, the revert had two causes: the approach was simply a poor one, and the
code was being cleared out while chasing an unrelated electrostatics bug — which turned
out to be an error in updating k-vectors after an accepted volume move, not anything to
do with the approximation.

That matters for what follows: **there is no outstanding accuracy finding against
approximating the real-space kernel.** The earlier attempt failed because it still
called a libm transcendental — `erfc` is itself roughly `exp(-x²) × rational`, so
swapping one libm call for another plus 25 FMAs trades like for like.

### What to do instead

Do not approximate `erfc` in isolation. Tabulate the entire quantity the callers need.
Every call site has the same shape:

```cpp
double dist = sqrt(distSq);
double val  = forcefield.alpha[b] * dist;
return qi_qj_Fact * std::erfc(val) / dist;
```

so tabulate `f(r²) = erfc(α·r)/r` directly, indexed on `r²`. One lookup then replaces
the `erfc`, the `exp` inside it, the `sqrt`, **and** the division.

- Uniform grid in `r²` over `[rCutLowSq, rCutCoulombSq]`, one table per box (`alpha` is
  per-box). Built once at startup; rebuild only if `alpha[b]` or `rCutCoulombSq[b]`
  changes.
- Linear interpolation to start; move to cubic only if the accuracy test demands it.
  Keep the table small enough to stay resident in L2 (~8K points ≈ 64 KB per box).
- Do the same for the virial form used by `CalcCoulombVir`.
- Guard `r → 0`: `erfc(αr)/r` diverges. Table starts at `rCutLowSq`; anything below that
  is already an overlap rejection, but the guard must be explicit rather than implied.

### Acceptance

Detailed balance is *not* at risk — a deterministic approximation used consistently
samples a slightly different Hamiltonian exactly. The real question is whether
coexistence properties shift. So:

1. Unit test in the style of `MieExponentTest.cpp`: table vs `std::erfc` across the full
   `r²` range, asserting a stated relative tolerance. Pick table resolution *from* a
   target error rather than guessing.
2. A GEMC coexistence run against the analytic kernel, comparing densities with error
   bars — not a bit-comparison. Interpolation error ~1e-9 should sit far below
   statistical noise; demonstrate that rather than assume it.

**Expected: 15–20% of wall clock.**

---

## Phase 2 — Devirtualise the forcefield type

**Target: the ~9.6% sitting in un-inlined `CalcEn`/`CalcCoulomb`, plus vectorisation.**

This is the surviving good idea from the old plan's Phase 1, and the profile now
justifies it directly rather than by assertion.

- Template the hot functions on the concrete forcefield type and dispatch once, the way
  `BoxType` already is. `forcefield.vdwKind` and `isMartini` are fixed after startup.
- **Sequence matters: do this after Phase 1.** While `erfc` remains a libm call the pair
  loop is latency-bound and cannot vectorise no matter how well it inlines. With the
  table in place the whole kernel becomes straight-line arithmetic, and inlining is what
  lets the compiler vectorise it.
- Watch the instantiation count. The old plan budgeted 5 FF types × 3 electrostatic
  combinations; layered on the existing `BoxType` templating across 6 functions that is
  180 copies of a large loop body. Collapse the electrostatic/Ewald pair to one bool and
  check compile time and I-cache before committing to the full cross product.

**Expected: 5–10%, and more once it composes with Phase 1.**

---

## Phase 3 — Recover the 10.4% spent waiting

Two structural causes are visible in the source:

- **`ParticleInterTemplate` parallelises over CBMC trials**, and trials are only 8–12
  (`CBMC First atom trials 12`, `Secondary 10`). On a node with more than ~12 cores the
  surplus threads have nothing to do and spin at the barrier. This is the most likely
  single contributor.
- **Many short parallel regions per MC move.** Region entry/exit is microseconds; the
  enclosed work is often comparable.

Cheap diagnostics before any code change:

1. Sweep `OMP_NUM_THREADS` and plot speedup. If it flattens near the trial count, the
   `ParticleInter` granularity is confirmed as the ceiling.
2. Set `KMP_BLOCKTIME=0` and re-measure. The default 200 ms spin burns CPU between short
   regions; this is a one-line environment experiment with no code risk.
3. Re-test `schedule(guided)` vs the current default static in `BoxInterTemplate`.
   `c0285206` moved dynamic→static on the older code; worth re-measuring now.

Then, if confirmed: give `ParticleInter` a thread team sized to the work
(`num_threads(min(trials, ...))`) so surplus threads sleep instead of spinning, or
restructure to parallelise over trials × neighbours together. Also hoist the per-trial
`std::vector<uint> nIndex` allocation — it currently mallocs inside the parallel region
on every trial, which is allocator contention across threads.

**Expected: perhaps half of the 10.4%.**

---

## Phase 4 — Halve the cell-list scan

**Target: part of `BoxInterTemplate`'s 30.5%.**

Two independent inefficiencies:

- **Every pair is visited twice.** The loop scans all 27 neighbour cells and discards
  half the pairs with `currParticle < nParticle`. A half stencil (13 cells plus half the
  home cell) visits each pair once and removes the test, roughly halving inner-loop trip
  count.
- **At GEMC liquid-box sizes the cell list prunes almost nothing.** Cells are `rCut`
  wide and clamped at `max(floor(side/rCut), 3)`. A 300-molecule n-octane box at ~43 Å
  with `rCut` 14 Å gives a 3×3×3 grid, so the 27-cell stencil *is* the whole box and
  `BoxInter` degenerates to O(N²). Halving cell width (5³ stencil) raises the fraction
  of scanned volume that is actually within cutoff from ~16% to ~27%.

These interact, so measure them together rather than separately.

---

## Phase 5 — PME: fix the per-trial-move FFT

PME on `origin/PME` is reported correct but slow, "very slow for single molecule moves."
That is structural and localisable.

### Where it goes

The trial-move path is well designed at the top level: `MolReciprocal` calls
`DeltaERecip`, which gets the linear term from `InterpolatePotential` against a cached
potential mesh — cheap, O(nAtoms × pmeOrder³) — and only does an FFT on *acceptance*
(`updateSRef`). So far so good.

The cost is in the quadratic self-term. `ComputeDeltaSsq` runs on **every trial move**
and does three things whose cost is independent of how little moved:

```cpp
fill(scratchMesh[box], scratchMesh[box] + Kx * Ky * Kz, 0.0);   // zero the whole grid
... spread 2*nAtoms atoms ...                                    // the only local part
fftw_execute(scratchPlan[box]);                                  // full 3D FFT
return SumMeshEnergy(box, S_delta[box], nullptr, false);         // full reciprocal pass
```

For a 48³ mesh that is an ~884 KB memset, a ~9.4 Mflop transform, and a 55k-point
reduction (with an integer `/` and `%` per point) — to account for a perturbation that
is nonzero on roughly `2·nAtoms·pmeOrder³` ≈ 650 of 110,592 grid points. Standard Ewald's
incremental update for the same move is ~90 Kflop. That is the ~100× being felt.

### The fix

The quantity `ComputeDeltaSsq` returns is the reciprocal-space self-energy of a *localised*
charge perturbation: `nAtoms` charges at their new positions and the same charges negated
at their old ones. For that few point charges it has a closed form in real space, via the
standard reciprocal/real-space identity — pair terms in `erf(α·r)/r` plus the usual self
term. That is O((2·nAtoms)²): 36 pair terms for a three-site model, against a 9.4 Mflop
FFT. GOMC already has this machinery — the intramolecular corrections in `Ewald` use
exactly `erf(α·r)/r`.

Note this composes with Phase 1: if the real-space kernel is tabulated, the `erf` form
needed here is the same table's complement.

**Consistency caveat.** The closed form and the gridded form differ by PME's own
interpolation error, so trial ΔE and the post-acceptance recomputed energy would use
slightly different estimators. The perturbation term is small and local, so evaluating it
exactly is arguably *more* accurate than gridding it — but it must be demonstrated, not
assumed. The acceptance path's full FFT resets the reference each time, so there is no
unbounded drift; the risk is a small bias in acceptance. Extend the existing PME unit
tests (`4f6824e5`, `ff5ad5b8`, `3d0900b2`) to cover it.

### Independent cheap wins in the same path

- Zero only the touched grid points (scatter, then undo) rather than the whole mesh.
- `MolReciprocal` heap-allocates `atomIndices` and `charges` per move, and does
  `cachedOldCoords[box].Uninit(); .Init(length)` — a free/malloc of an `XYZArray` on
  every trial move. Hoist all of these.
- `SumMeshEnergy` recovers `ix, iy, iz` with an integer divide and modulo per grid point;
  nested loops remove that.

### But first: is PME the right algorithm here at all?

In the profiled GEMC run `MolReciprocal` does not appear in the top fifteen — incremental
Ewald already makes single-molecule reciprocal updates cheap. `BoxReciprocalSetup` (the
full recalculation, i.e. volume moves) is 9.4%.

So PME's upside is bounded by that 9.4%, while its cost lands on the most frequent
operation in the simulation. At current system sizes standard Ewald with running structure
factors is the better algorithm for GEMC. PME's case is asymptotic: `BoxReciprocalSetup` is
O(N × imageSize) ≈ O(N²) whereas a PME full recalculation is O(N log N).

Recommendation: fix the trial-move FFT so PME is not pathological, then **measure the
crossover N** against standard Ewald. Do not sequence PME ahead of Phases 1–2 unless the
target systems are far larger than what was profiled.

---

## Dropped from the old plan

| Item | Why |
|---|---|
| `GetNeighborList` return-by-copy | Still correct and trivial, but it does not appear in the profile. Do it opportunistically, not as a phase. |
| Filter-then-compute restructure | Motivation was that `pow` blocked SIMD; `pow` is gone. Re-measure after Phases 1–2 before adding scratch buffers. Its sketch also had a 256-element buffer where neighbourhoods hold thousands. |
| Linearise `EnumerateLocal` for `MoleculeInter` | `MoleculeInter` is 2.0%. Retarget the idea at `ParticleInter` (10.8%), which uses the same linked-list walk. |

---

## Expected overall

Phases 1–4 plausibly remove 35–40% of current runtime, i.e. roughly **1.5–1.7×**. The
old plan's "3–8×" was not attainable — after the transcendentals and the dispatch
overhead are gone, what remains in `BoxInter` is real pair work bounded by the O(N²)
behaviour described in Phase 4.

Re-profile after each phase. The ordering above assumes the current distribution; Phase 1
alone will change it enough to make the remaining estimates stale.

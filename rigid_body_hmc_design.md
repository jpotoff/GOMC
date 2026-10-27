# Rigid-Body Hybrid Monte Carlo: Design

Status: proposal only. No code has been written.

## Summary

A collective move that displaces and rotates every molecule in a box at
once, by integrating a short rigid-body molecular-dynamics trajectory and
accepting or rejecting the endpoint with the Metropolis test on the total
energy change. Hybrid Monte Carlo (HMC; Duane, Kennedy, Pendleton and Roweth
1987; Mehlig, Heermann and Forrest 1992 for molecular systems) is exact for
any time step and trajectory length, because the integrator is
volume-preserving and time-reversible. Integration error lowers the
acceptance rate but never biases the averages.

It is meant to replace the multiparticle move (`MultiParticleFreq`), and in
dense liquids possibly most single-molecule displacement and rotation.
Molecules are rigid during the trajectory; their internal degrees of freedom
are still sampled by CBMC regrowth and crankshaft.

**What it buys.**

- **Against multiparticle:** many force evaluations per accept/reject
  instead of one, with momentum carried between steps, so molecules travel
  ballistically over a trajectory rather than by a random walk of tiny
  steps. Translation and rotation are coupled instead of alternating.
- **Against single-molecule displacement/rotation:** it depends on the
  system (see [Expected efficiency](#expected-efficiency)). For OPC water at
  260 K, where relaxation needs collective rearrangement of the
  hydrogen-bond network, an MD reference projects HMC at about 5–20× the
  sampling per CPU-second of single-molecule moves, with GOMC's current
  force kernels. For a simple liquid like hexane the advantage is expected
  to be much smaller.
- **On top of that:** all the work is one large, regular force evaluation
  per step, which parallelises across all cores and on a GPU, and can use a
  Verlet neighbour list and eventually a GROMACS-style tiled kernel (see
  [Neighbour lists](#neighbour-lists-and-the-tiled-kernel)). Single-molecule
  moves share none of those gains.

## Why multiparticle is slow

From `src/moves/MultiParticle.h`:

- Every move costs a full energy-and-force evaluation of the box:
  `cellList.GridAll`, `Ewald::BoxReciprocalSums`, `CalculateEnergy::BoxForce`
  (short-range energy and forces), `Ewald::BoxReciprocal`,
  `Ewald::BoxForceReciprocal` and `CalculateTorque`. A rejected move pays
  another `GridAll` to restore the cell list.
- For that, it takes **one** force-biased step (the smart-MC proposal with
  λ = 0.5), and either translates every molecule or rotates every molecule,
  never both in the same move.
- The step size (`mp_t_max`, `mp_r_max`) is adapted to the target
  acceptance. For a move of every molecule, the step has to shrink as the
  system grows, and successive accepted moves add up as a random walk.

One full-box evaluation therefore moves each molecule a very short distance.

## The move

### State and Hamiltonian

For each mobile molecule i in the chosen box:

- centre of mass **R**ᵢ, total mass Mᵢ, momentum **P**ᵢ;
- orientation Qᵢ (rotation matrix, body → space), principal moments
  I₁, I₂, I₃, body-frame angular momentum **π**ᵢ;
- body-frame atom coordinates **b**ᵢₐ, fixed for the duration of the move,
  so that atom a sits at **x**ᵢₐ = **R**ᵢ + Qᵢ **b**ᵢₐ.

H = U(**x**) + Σᵢ [ |**P**ᵢ|² / (2Mᵢ) + Σₖ πᵢₖ² / (2Iᵢₖ) ]

U is the full GOMC potential. Over a rigid-body move the intramolecular
energy, the Ewald self and intramolecular-correction terms, and the tail
correction are all constant. Only the intermolecular LJ, real-space Ewald
and reciprocal Ewald energies change.

### One HMC move

1. **Choose the box** as multiparticle does: box 0 in GCMC; in GEMC
   optionally the liquid or gas box (`MultiParticleLiquid`/`Gas`).
2. **Build body frames.** For each mobile molecule: unwrap around its COM
   (`UnwrapPBC`), compute the inertia tensor, diagonalise it (3 × 3 Jacobi),
   and store **b**ᵢₐ in the principal frame and Qᵢ.
3. **Draw momenta** from the Maxwell–Boltzmann distribution: each component
   of **P**ᵢ ~ N(0, Mᵢ kT), each πᵢₖ ~ N(0, Iᵢₖ kT). Use `Random123Wrapper`
   keyed by (step, molecule), as multiparticle already does, so the draw is
   reproducible and parallel.
4. **Initial energy and forces.** U₀, the forces and the torques at the
   start; reuse them from the previous HMC move when no other move has been
   accepted since (see [Energy consistency](#energy-consistency)). If the
   neighbour list is stale, rebuild it in the same pass.
5. **Integrate L steps** with the splitting below.
6. **Final energy** U₁, computed exactly, as for every other move.
7. **Accept** with probability min(1, exp(−β ΔH)). On acceptance, write the
   new coordinates and COMs and update the reciprocal-space sums. On
   rejection, restore the old coordinates and cell list. Momenta are
   discarded either way.

### Integrator

One step of length Δt, symmetric (Strang) splitting:

```
kick   (Δt/2):  P  += (Δt/2) F            L += (Δt/2) τ   (space frame)
                π  = Qᵀ L
free rotor (Δt):  R1(Δt/2) R2(Δt/2) R3(Δt) R2(Δt/2) R1(Δt/2)
drift  (Δt):    R  += Δt P / M ;  wrap COM into the box
positions:      x  = R + Q b      (from body coordinates, every step)
forces:         F, τ at the new positions
kick   (Δt/2):  as above
```

The free-rotor step is the splitting of Dullweber, Leimkuhler and
McLachlan (1997): each Rₖ(h) is the exact flow of the sub-Hamiltonian
πₖ²/(2Iₖ), a rotation of both Q and **π** about principal axis k by the
angle h πₖ / Iₖ. NO_SQUISH (Miller et al. 2002) is the equivalent quaternion
formulation; the rotation-matrix form is closer to GOMC's existing
`RotationMatrix` utilities. Each piece is the exact flow of part of H, so
the composition is symplectic (volume-preserving), and the symmetric order
makes it time-reversible.

Atom positions are rebuilt from the body coordinates every step, so the
molecular geometry does not drift. The rotation matrices lose
orthogonality only at round-off level over L steps; do not reorthonormalise
during a trajectory, since that would break exact reversibility for
nothing.

### Units

GOMC uses Å, amu and kelvin (energy/k_B). The natural time unit is
τ = Å √(amu / (k_B K)) = 1.0967 ps, so **1 fs = 9.118 × 10⁻⁴ τ**. In these
units k_B = 1: velocity components have variance T/M, and forces (K/Å) and
torques (K) come out of `BoxForce` and `CalculateTorque` already consistent.
Expose the time step to the user in fs.

### Why it is exact, and what would break it

The proposal map (momentum draw, then L integrator steps, then momentum
flip) is deterministic, volume-preserving and an involution. Metropolis on
ΔH then satisfies detailed balance for **any** Δt and L. That requires:

- **Forces that are a deterministic function of the current positions.**
  They do not have to be the exact gradient of U. Volume preservation and
  reversibility hold for any such force function, so cheaper forces only
  lower acceptance (see [Stage 2](#stage-2-optimisations)), provided ΔH uses
  the exact U.
- **No changes within a trajectory** that depend on the path: no adaptive
  Δt, no ad hoc renormalisation.
- **A neighbour list, if used, that always contains every pair within the
  force cutoff.** A Verlet list rebuilt whenever an atom has moved more
  than half the skin is fine: the forces are still the exact function of
  the positions, whichever list produced them.
- **Δt and L independent of the current state.** Fixing them, or drawing
  them from a fixed distribution each move, is valid. Adapting Δt from the
  acceptance history makes the chain slightly non-Markovian: adapt during
  equilibration only, then freeze. This matches how the existing step-size
  adaptation should be read.

**Early termination is allowed.** A trajectory that hits a state-defined
failure (non-finite force, a pair inside `RcutLow`, a single-step energy
jump above a threshold) can be stopped and rejected immediately. The
reverse trajectory visits the same states, so the abort indicator is
symmetric and detailed balance still holds. This also saves the rest of
the trajectory.

Floating-point round-off makes reversibility approximate, and chaotic
dynamics amplify it with L. For trajectories up to about 1 ps (a few hundred
steps) the amplified error stays many orders of magnitude below anything
measurable.

### Special molecules

- **Fixed molecules** (`IsFix`): excluded, as in multiparticle.
- **Single atoms:** translation only.
- **Linear molecules:** one principal moment is zero. Mark axes with
  Iₖ < ε · max(I) as inactive (πₖ ≡ 0, no rotor sub-step); the torque about
  that axis is zero for point masses on a line.
- **Molecules across the box boundary:** body frames are built from
  unwrapped coordinates; positions are rewrapped after each drift, as
  multiparticle already does.

## Energy consistency

`MultiParticle::CalcEn` computes the new energy from scratch (`BoxForce`
plus `BoxReciprocal`) and compares it with `sysPotRef`, the running total
maintained by every move since the last full recalculation. Any drift in
that running total goes straight into ΔU.

HMC needs the initial forces anyway, so it should compute U₀ fresh, with
the same functions and summation order that give U₁. Then ΔU is consistent
to round-off. On acceptance, `sysPotRef` takes the freshly computed values
for the box, which also resynchronises the running total.

Reuse: if no other move has been accepted in the box since the previous HMC
move (`moveSetRef.GetSingleMoveAccepted`, which multiparticle uses
already), the previous final energy and forces are the current ones. With
HMC replacing most displacement and rotation, only swaps and regrowths
invalidate them, and those are accepted rarely.

## Implementation in GOMC

### Reused

| Piece | Existing code |
|---|---|
| Short-range energy and forces | `CalculateEnergy::BoxForce` (cell list, vectorised pass 1) |
| Reciprocal sums, energy, forces | `Ewald::BoxReciprocalSums`, `BoxReciprocal`, `BoxForceReciprocal` |
| Torques | `CalculateEnergy::CalculateTorque` |
| Cell list rebuild | `CellList::GridAll` |
| Per-molecule random numbers | `Random123Wrapper` |
| Accept bookkeeping | the `MultiParticle::Accept` pattern: swap coordinate, COM and force arrays, `UpdateRecip`, `velocity.UpdateBoxVelocity` |
| Force arrays | `atomForce*`, `molForce*`, `molTorque*` members already in `System`/`MoveBase` |

### New

- `src/moves/HybridMC.h`: the move (Prep / Transform / CalcEn / Accept, as
  for other moves). Transform runs the whole trajectory; CalcEn produces U₁.
- A rigid-body helper (body frames, inertia diagonalisation, the
  free-rotor sub-steps), testable on its own.
- Saved state for rejection: coordinates and COMs (already double-buffered
  by multiparticle), reciprocal sums (`sumRnew`/`sumRref` swap, as now).
- Config keywords:

  | Keyword | Meaning |
  |---|---|
  | `HMCFreq` | move frequency, like the other `*Freq` keywords |
  | `HMCTimeStep` | Δt in fs; adapted during equilibration if `HMCAdjust true` |
  | `HMCSteps` | L, or `Lmin Lmax` to draw L uniformly per move; a few hundred steps (0.5–1 ps), see [Expected efficiency](#expected-efficiency) |
  | `HMCSkin` | Verlet-list skin in Å, see [Neighbour lists](#neighbour-lists-and-the-tiled-kernel) |
  | `HMCLiquid` / `HMCGas` | GEMC box choice, as for multiparticle |
- Output: acceptance per box, mean ΔH, ⟨exp(−β ΔH)⟩ (see
  [Validation](#validation)), Δt in use, and the move time.

### Ensembles

The move works at fixed volume and molecule count within one box, so it
combines with volume moves (NPT, GEMC) and swaps (GCMC, GEMC) like any other
move. The set of mobile molecules is rebuilt each move, as
`SetMolInBox` does now.

### Parallelism

Each step is one full force evaluation, which already uses OpenMP in
`BoxForce` and the Ewald routines; the integrator updates are a parallel
loop over molecules. The CUDA build has GPU kernels for the same force
terms. Version 1 runs the integrator on the CPU; a GPU version would move
the integrator into a kernel to avoid a host–device transfer each step.

## Expected efficiency

Measured on a 4-core VM (no AVX), one core per run, October 2026. The
metric is the CPU time needed to reach the same centre-of-mass
mean-squared displacement (MSD) or the same orientation autocorrelation.
Comparing values at a fixed CPU time instead understates the differences,
because the curves are sublinear.

### Single-molecule moves against multiparticle (measured, GOMC)

| System | Multiparticle | Single-molecule moves | Single moves vs multiparticle |
|---|---|---|---|
| All-atom hexane, 273 molecules, 400 K | 144 ms/move, 39% accepted | 2.4 ms/move, 50% accepted | 3.4–3.6× faster |
| OPC water, 500 molecules, 1.000 g/cm³, 260 K | 18.9 ms/move, 40% accepted | 0.145 ms/move, 50% accepted | 2.0–2.2× faster |

Multiparticle loses to plain single-molecule moves in both, but by less in
cold water, where relaxation is collective.

### MD reference for OPC water at 260 K (GROMACS, projected to GOMC cost)

HMC with a good time step moves molecules like MD of the same length,
scaled by its acceptance. The same OPC system was run in GROMACS 2024.5
(topology matching GOMC to 0.4% in potential energy, same starting state),
and each MD step was charged at GOMC's measured cost of one full force
evaluation (18.9 ms, the multiparticle move). GROMACS's own speed does not
enter.

| Method | CPU time to MSD 1.44 Å² | CPU time to dipole autocorrelation 0.77 |
|---|---|---|
| Multiparticle (measured) | 880 s | ≈ 880 s |
| Single-molecule moves (measured) | 393 s | ≈ 413 s |
| MD, all momenta redrawn every 0.1 ps (projected) | ≈ 56 s | ≈ 48 s |
| MD, weak thermostat (projected) | ≈ 17 s | ≈ 20 s |

HMC acceptance, estimated from the energy error of constant-energy GROMACS
runs over 0.1–1 ps windows (Gaussian estimate, 500 molecules):

| Δt | Energy-error spread (kT) | Acceptance |
|---|---|---|
| 1 fs | 0.30–0.46 | ≈ 0.85 |
| 2 fs | 0.55–0.67 | ≈ 0.75 |
| 3 fs | 1.2–1.45 | ≈ 0.5 |
| 4 fs | ≈ 2.5 | ≈ 0.2 |

Conclusions:

- **Δt = 2 fs at about 75% acceptance** maximises acceptance × Δt. The
  energy error does not grow with trajectory length for a symplectic
  integrator, so long trajectories cost no acceptance.
- **Trajectories should be long, 0.5–1 ps (250–500 steps).** Redrawing all
  momenta every 0.1 ps cut diffusion by 2.7×: it destroys the collective
  momentum that carries molecules out of their cages.
- **Projected gain for cold water:** about 5–20× the sampling per
  CPU-second of single-molecule moves, and 12–40× that of multiparticle,
  with GOMC's current force kernels.
- **Hexane has not been measured this way.** For a simple liquid that
  relaxes by local diffusion the advantage over single-molecule moves is
  expected to be much smaller; an MD reference would settle it.
- **Acceptance falls with system size.** The energy-error spread grows
  about as √N, so ten times more molecules needs Δt of roughly 1.2–1.5 fs
  for the same acceptance. The optimal HMC acceptance for large systems is
  about 65% (Beskos et al. 2013); a target of 65–80% is reasonable.
- **Per-step cost is assumed, not measured:** one HMC step is taken to cost
  one multiparticle move, since both run the same kernels.

### Headroom

GROMACS ran the same system at about 2.3 ms per step on one SSE2 thread,
about 8× less than GOMC's full force evaluation. The difference is Verlet
neighbour lists, PME and fully vectorised tiled pair kernels. Every factor
recovered there goes straight into HMC efficiency, and single-molecule
moves do not share it (see
[Neighbour lists](#neighbour-lists-and-the-tiled-kernel)).

### Relation to volume moves

Converging the density of cold OPC water needed about 5% volume moves. Each
volume move is a full-box energy evaluation, so in such runs volume moves
take most of the CPU time. They can share the neighbour list (below), and
HMC relaxes the hydrogen-bond network between volume moves far faster than
single-molecule moves. The direct test is NPT on
`test/input/Systems/CBMC/OPC_260K`, measuring the density autocorrelation
per CPU-second.

## Neighbour lists and the tiled kernel

### The list over its lifetime

**Within one HMC move.** Build a Verlet list with radius r_list = r_c + s,
where r_c is the largest cutoff (Coulomb real-space or LJ) and s the skin.
Keep the positions it was built from as reference positions. After each
step, rebuild when the two largest atom displacements from the references
add up to more than s. A complete list gives forces that are an exact
function of the positions, so reversibility is unaffected by when the
rebuilds happen. On rejection, restore a saved copy of the starting list or
mark the list stale.

**Between moves: a validity check, not updates.** The list records its
reference positions, the box dimensions, the cell-list version
(`CellList::Version()`, bumped by every molecule mutation) and how much of
the skin has been used. It stays valid while the cell-list version and box
are unchanged; HMC moves track their own displacements. Otherwise it is
marked stale and rebuilt lazily, at the next move that uses it.

A rebuild costs about one cell-list pass, a fraction of one force
evaluation. Against a 250–500-step HMC move that is under 1%, so
incremental updates are not worth their bookkeeping in version 1.

### Effect of other moves

| Move | Uses the list | After acceptance |
|---|---|---|
| HMC | yes, every step | list stays valid; displacement tracking continues |
| Volume (rigid molecules) | yes, for the short-range energy | valid while the skin allowance holds, see below |
| Swap, deletion | no | still complete: removed atoms are skipped by box membership; no rebuild |
| Swap, insertion | no | stale: the new atoms are missing. Rebuild lazily, or add incrementally (below) |
| Regrowth, crankshaft, other CBMC | no | stale |
| Single-molecule displacement/rotation | optional, see below | stale unless displacement tracking is added |

**Swaps and CBMC cannot use the list.** Their trial positions are new
(insertion trials, the generated trials of the old configuration), and the
growing molecule's focus atom is not in the list. They keep the cell list
and the per-growth-step candidate list
(`CalculateEnergy::BeginLocalCandidates`). The cell list must therefore
stay current at all times, as it does now.

**Incremental insertion**, if profiling ever shows rebuilds after accepted
insertions matter (high swap acceptance, as for gases or united-atom
molecules near the critical point, together with short HMC moves or
frequent volume moves): find the new atoms' neighbours with the cell list
and record their inserted positions as references. With a half list (each
pair stored once) it is enough to put every pair involving the new atoms in
the new atoms' own lists, so no existing list changes. This needs per-atom
storage that can grow, rather than one packed array.

An accepted swap also invalidates the cached HMC forces and energy (see
[Energy consistency](#energy-consistency)), so the next HMC move does one
fresh evaluation anyway; a needed list rebuild shares that pass.

### Volume moves

With rigid molecules a volume move scales the centres of mass by
λ = (V′/V)^(1/3). An intermolecular atom-pair distance changes by at most
|λ − 1| (r_list + 2 r_mol), where r_mol is the largest molecular radius.
For a ±0.5% scaling with a 13 Å list radius and water-sized molecules that
is under 0.1 Å. Charge it against the skin like a displacement: while the
total used stays below s/2, the list is still complete in the new box (the
minimum image is evaluated with the new box at each pair).

The volume move then sums its short-range energy over the list, without
rebuilding and walking the cell list, which speeds up its real-space part
as much as an HMC step. Reciprocal space is still recomputed for the new
box and becomes the main cost of a volume move, which is one more argument
for PME.

### Single-molecule moves (optional)

Track each atom's displacement from its reference position. A displaced or
rotated molecule within its allowance can sum its energy over the list (a
sphere of neighbours instead of 27 cells); a molecule that moves farther
marks the list stale. This is the standard Verlet-list Monte Carlo scheme.
It is independent of HMC and would speed up displacement and rotation too.

### Box-size constraint

The list radius must stay below half the shortest box edge, or the minimum
image becomes ambiguous. `test/input/Systems/CBMC/OPC_260K` has
RcutCoulomb 12 Å in a 24.64 Å box, which leaves a skin of at most 0.32 Å.
Options: lower RcutCoulomb to about 10 Å (raising the Ewald reciprocal
accuracy to match), use a larger box (1,000 or more waters), or fall back
to the cell list when r_c + s ≥ L/2. Small GEMC boxes have the same limit.

### Path to a GROMACS-style tiled kernel

GROMACS's non-bonded scheme (nbnxm) sorts atoms spatially, groups them into
clusters of 4, builds a cluster-pair list from bounding-box distances with
an automatic buffer, and computes every pair of an i-cluster against a
j-cluster (4×4 or 4×8) as dense SIMD arithmetic, masking pairs beyond the
cutoff and excluded pairs instead of branching. It computes extra distances
to avoid gathers and branches entirely.

What the steps above already provide:

| Ingredient | GOMC status |
|---|---|
| A list that lives across many steps, with rebuild rules | the lifetime and validity rules above, which do not depend on whether the list holds atoms or clusters |
| Packed, spatially ordered coordinates and parameters | partly: `BuildBoxPacked` and `BuildCellOrdered`; clusters need columns sorted in z, groups of 4, and forces scattered back |
| Branch-free masked compaction and SIMD distance passes | `BoxInterTemplate`, `ParticleInterTemplate` (`DistSqRange`, AVX-512 compress); a tiled kernel drops compaction altogether |
| Enough full-box evaluations to pay for the setup | HMC steps and volume moves |

What is new:

- cluster grid, bounding boxes and cluster-pair list construction;
- 4×N kernels in double precision (AVX2 holds 4 doubles, AVX-512 holds 8:
  half GROMACS's mixed-precision lane count);
- exclusion masks. Every same-molecule pair is excluded from the
  intermolecular energy, so the mask is a molecule-index comparison;
- per-pair-type parameter lookup for several kinds and Mie exponents, plus
  GOMC's tabulated real-space Ewald;
- energy-only, energy-plus-force and virial variants, with per-molecule
  force and torque sums for HMC.

The tiled kernel serves full-box evaluations only: HMC steps, volume moves,
`BoxInter`/`BoxForce` and pressure. Single-molecule moves and CBMC need the
energy of a few atoms at new positions, which clusters do not fit; they
keep the cell list, the candidate list or the optional per-atom Verlet
list. The cluster structure is also what GROMACS's GPU kernels use, so it
is the natural route to a GPU HMC.

### Staged plan

1. **Neighbour-list infrastructure behind a small interface** ("full-box
   energy and forces using the list"), holding the validity rules
   (cell-list version, box, displacement and volume allowance) and lazy
   rebuild, first with a simple per-atom kernel. HMC and volume moves use
   it from the start.
2. **Replace the implementation behind the interface** with cluster lists
   and 4×N SIMD kernels: orthogonal boxes, no fractional molecules (those
   keep the current path), AVX2 and AVX-512. The move code does not change.
   Debug builds cross-check against the existing kernel, as the
   per-growth-step candidate list does.
3. **Later:** dynamic pruning, non-orthogonal boxes, a GPU version.

Expected gain of stage 2 over stage 1: plausibly 2–4× on the short-range
part, from GROMACS's experience and the cost of gathers on AVX-512.
Reciprocal-space Ewald is unaffected and then dominates, so PME (in
development on another branch) is the next step. Together they account for
most of the roughly 8× gap between GOMC's and GROMACS's cost per force
evaluation.

### What the per-atom vectorisation attempt showed

Before starting on clusters, `BoxForceTemplate` was restructured the way
`BoxInterTemplate` is (commit dceacbb1, reverted afterwards).
- **Pass 2:** AVX-512 compress stores copied the pairs inside the cutoff
  into contiguous buffers (distance², the three minimum-image components,
  and the charge or kind of j).
- **Pass 3:** became dense, scatter-free Coulomb-table and fixed-exponent
  LJ loops.
- **Scatter:** a separate loop applied the j-forces.

Energies matched the old kernel to rounding on OPC HMC and UA hexane GEMC.

Measured on the 9950X (icpx, znver5) with `OPC_260K` and 4 threads:

| | ff69a26a (scalar pass 3) | dceacbb1 (vectorised pass 3) |
|---|---|---|
| Instructions | 1.810 × 10¹² | 1.179 × 10¹² (−35%) |
| IPC | 2.87 | 1.76 |
| Cycles | 631.6 × 10⁹ | 671.6 × 10⁹ (+6%) |
| Elapsed | 30.24 s | 31.84 s (+5%) |

The 1-thread annotate put about half of the kernel's samples on two
instruction groups:
- **Table gathers, ≈ 23.5%:** eight `vgatherdpd` per eight pairs, one per
  coefficient of the 64-byte energy/virial line. The LJ parameter gathers
  added ≈ 3%.
- **Compress stores to memory, ≈ 23%:** twelve `vcompresspd`/`vpcompressd`
  per eight pairs.

Lessons for the tiled kernel:
1. **No compaction.** Mask the pairs outside the cutoff instead. The
   wasted lanes cost less than compress stores.
2. **No element-wise gathers from the Coulomb table.** Either load each
   pair's whole line and transpose in registers, or do not use a table.
3. **Judge on the target hardware.** Fewer instructions did not mean fewer
   cycles. Every variant is judged by elapsed time on the AVX-512 machine.

### Tiled kernel: design decisions

**Scope.** The tiled kernel computes full-box short-range energy, forces
and (later) the virial, for orthogonal boxes with no fractional molecule.
Everything else keeps the current kernels, which also remain the
reference:
- non-orthogonal boxes;
- CFCMC/MEMC λ-scaled molecules;
- vdW forms without a fixed-exponent evaluator (Martini, Exp-6);
- Coulomb pairs outside the real-space table's range.

The dispatch happens once per call in `CalculateEnergy::BoxForce`, not per
pair. Nothing in the kernel assumes a particular molecule or force field:
- sites with zero charge or zero ε are ordinary lanes;
- exclusions come from comparing molecule indices.

**Clusters.**
- **Formation:** spatial, as in GROMACS. Bin atoms into columns on an x–y
  grid, sort each column by z, and cut it into clusters of 4 atoms.
- **Padding:** a short column is padded with dummy atoms, which have
  charge 0, ε 0 and an excluded molecule index.
- **Why not molecular clusters:** molecule-based clusters (one cluster per
  molecule) would suit 4-site water but nothing else, so they are not
  used.
- **Rebuild:** clusters are rebuilt from the cell-ordered arrays
  (`BuildCellOrdered`) at each list build. Coordinates, charges, kinds and
  molecule indices are stored in cluster order, so the kernel never
  indexes through a permutation.

**Tile shape.** i-clusters of 4 against j-clusters matched to the SIMD
width:
- **AVX-512:** 4 × 8, so one j-cluster fills a register of 8 doubles.
- **AVX2:** 4 × 4.

Each i-atom is broadcast against the j lanes. For each tile:
- **Distance and cutoff:** 4 distance computations of one register each,
  one cutoff mask per i-atom.
- **Exclusions:** same molecule, padding, and j ≤ i on the diagonal tile.
- **Pair list:** a half list over cluster pairs (ci ≤ cj), so each atom
  pair is computed once.

**Forces without scatter.**
- **i-forces:** accumulate in registers across a j-cluster list.
- **j-forces:** within a tile, the j-force register accumulates
  lane-for-lane over the 4 i-atoms and is then added to the j-cluster's
  forces with one contiguous vector load-add-store.
- **Storage:** forces live in cluster order and are permuted back to atom
  order once per evaluation, an O(N) pass.

Per-thread force buffers with the current OpenMP array reduction come
first. Smarter buffer schemes come later only if profiling asks for them.

**Coulomb: transposed table, not analytical erfc.** GROMACS's CPU kernels
evaluate the real-space Ewald term analytically. For GOMC the deciding
issue is consistency: the energy the HMC move stores in `sysPotRef` must be
the same function of the coordinates that `BoxInter` and the
single-molecule moves use. Otherwise, mixing moves samples a slightly
inconsistent Hamiltonian. The kernel therefore uses the existing
`EwaldRealTable` coefficients:
- each pair's line is loaded whole (one 64-byte load for the energy and
  virial cubics);
- an 8 × 8 in-register transpose turns 8 such lines into 8 coefficient
  vectors.

That replaces 64 gathered elements per 8 pairs with 8 full-width loads and
about 24 shuffles. Pairs outside the table range drop the whole call to
the reference kernel, as `tabBad` does now. An analytical erfc only makes
sense if every kernel in GOMC switches to it together.

**LJ parameters.** Lookups go per kind pair through the flattened table,
so NBFIX keeps working. With i fixed per broadcast, each lane needs
`kind_i + kind_j × stride`. That is a gather of 3 parameters (σ², ε·c_n,
n/6 or ε·c_n·6), which measured ≈ 3% of the kernel in the per-atom attempt.
That cost is accepted in version 1. A later option is a per-tile parameter
block when a j-cluster holds a single kind.

**List lifetime under HMC: rebuild at every force evaluation (version
1).**
- **Why rebuild every time:** a cluster-pair list built with r_list = r_c
  and no skin is complete for the positions it was built from, so forces
  stay an exact function of the coordinates and reversibility is not in
  question.
- **Cost:** about 500 clusters (OPC, 2,000 sites) give about 125,000
  cluster pairs to test by bounding-box distance, against about 10⁶ pair
  interactions per evaluation. That should be a few percent, to be
  confirmed by profiling.
- **Small boxes:** it also avoids the box-size constraint above. A
  24.64 Å box with r_c 12 Å has no room for a useful skin, and a list
  without a skin needs only r_c < L/2, which GOMC already requires.

Skinned lists kept across steps (the validity rules in
[The list over its lifetime](#the-list-over-its-lifetime)) become a stage
2 optimisation for large boxes.

**Testing without the target ISA.** The development VM has no AVX at all.
So the kernel is written against a thin SIMD layer of a few dozen inline
functions: load, broadcast, FMA, masked add, compare-to-mask, the table
transpose, horizontal sum. The layer has three backends:
- a scalar backend (arrays of 4 or 8 doubles), which runs anywhere;
- AVX2;
- AVX-512.

The tile logic (cluster formation, list build, masks, force accumulation,
the permutation back) is validated with the scalar backend against the
current kernel on the VM. Each primitive has a unit test that compares the
SIMD backends with the scalar one, run on the AVX-512 machine. Performance
is measured only there.

**Validation.**
- **Debug builds:** cross-check energies and forces against the current
  kernel on every call, as the per-growth-step candidate list does.
- **Systems:** OPC HMC (charged, mixed zero/non-zero sites) and UA hexane
  GEMC (uncharged).
- **Acceptance criteria:**
  - energies agree to rounding;
  - the HMC trajectory agrees over 20 moves;
  - ⟨exp(−βΔH)⟩ is unchanged.

**Benchmark and decision.**
- **Benchmark:** `OPC_260K` with `HMCSteps 50` and 20 moves, at 1 and 4
  threads, against ff69a26a, by `perf stat` elapsed time and cycles.
- **Decision:** the kernel stays only if it beats the current kernel at
  both thread counts on the 9950X. A UA hexane HMC case is checked as well,
  so the decision does not rest on water alone.

**Revised staged plan.** This replaces stage 1 of the plan above. A
per-atom Verlet list is no longer a step on the way, because the cluster
list does its job.
1. **Cluster list rebuilt at every evaluation** + 4 × N kernel for
   `BoxForce`: energy and forces, scalar/AVX2/AVX-512 backends. HMC uses it.
2. **Energy-only and virial variants** for `BoxInter` (volume moves) and
   pressure.
3. **Skinned lists kept across steps and moves** for large boxes;
   reciprocal-space MTS or PME.
4. **Later:** non-orthogonal boxes, fractional molecules in the tiled path,
   GPU.

## Stage 2 optimisations

All of these keep the move exact, because the acceptance test uses the
exact energy.

- **Verlet neighbour list for the trajectory.** Build at the start of each
  trajectory with a skin; rebuild when any atom has moved more than half
  the skin. For hexane (cutoff 9 Å, 10.5 Å cells) a 1 Å skin examines about
  7× fewer pairs than the 27-cell stencil, before the j > i halving.
- **Multiple time stepping for reciprocal space** (r-RESPA; Tuckerman,
  Berne and Martyna 1992). Apply the reciprocal-space force as an outer
  kick every m steps: symmetric impulse splitting is still symplectic and
  reversible. It removes most per-step `BoxReciprocalSums` and
  `BoxForceReciprocal` calls.
- **Cheaper forces inside the trajectory**, such as a shorter cutoff, with
  the exact energy at the endpoints: the dual-cutoff idea, which fits HMC
  far better than CBMC.
- **Partial momentum refresh** (generalised HMC; Horowitz 1991): keep part
  of the momentum between moves and flip it on rejection, which gives
  Langevin-like persistence. It needs momenta stored across other moves, so
  not in version 1.
- **Flexible-molecule HMC:** constrained bonds (RATTLE) with flexible
  angles and torsions. It would also sample conformations in dense liquids,
  but needs intramolecular forces that GOMC's rigid-body code does not
  have.

## Validation

1. **Integrator unit tests** (no GOMC system needed):
   - free asymmetric rotor: energy and angular momentum conserved to
     O(Δt²), with no secular drift over 10⁶ steps;
   - reversibility: integrate L steps, flip momenta, integrate L steps, and
     recover the start to round-off;
   - linear molecule and single atom: no motion about inactive axes.
2. **Energy-error scaling on hexane:** mean ΔH ∝ Δt² over Δt = 0.5–4 fs.
3. **⟨exp(−β ΔH)⟩ = 1.** For a correct volume-preserving, reversible
   integrator started from equilibrium this identity holds exactly. It is a
   cheap, sensitive check (report it in the output). A value off by more
   than its error bar means a bug.
4. **Distributions:** NVT hexane and ethanol with HMC only (plus regrowth)
   against displacement/rotation only: mean energy, its variance, and the
   COM radial distribution function, within error bars. NPT density and
   GEMC coexistence densities likewise.
5. **Efficiency:** COM mean-squared displacement and the orientation
   autocorrelation per CPU-second and per wall-second, at 1, 4 and 16
   threads, against single-molecule moves and multiparticle.

## Cost and risk

- **Size:** moderate. The integrator and rigid-body helper are new but
  self-contained; force, energy, Ewald, cell-list and accept code are
  reused. Config parsing and output follow the multiparticle pattern.
- **Main risks:**
  - an integrator bug that breaks volume preservation or reversibility
    silently biases results. Validation steps 1–3 catch it: ⟨exp(−β ΔH)⟩
    in particular;
  - per-step cost without a neighbour list. Version 1 inherits the full
    `BoxForce` and reciprocal cost every step, so Stage 2 decides how
    competitive the move becomes;
  - small Δt in systems with light, fast rotors, keeping trajectories short
    in physical time.

## Decision criteria

Build version 1 if:

1. displacement, rotation and multiparticle together are a large part of
   run time, or the system needs collective moves that single-molecule
   moves handle poorly (dense, slowly relaxing liquids); and
2. parallel throughput matters: one run spread over many cores or a GPU,
   rather than many independent single-thread runs.

Measure first, with no new code: COM mean-squared displacement and the
orientation autocorrelation per CPU-second for single-molecule moves
against multiparticle on liquid hexane. That shows how far multiparticle
is behind, and the version 1 HMC result is then compared with the same
numbers.

## References

- S. Duane, A. D. Kennedy, B. J. Pendleton, D. Roweth, Phys. Lett. B 195,
  216 (1987). Hybrid Monte Carlo.
- B. Mehlig, D. W. Heermann, B. M. Forrest, Phys. Rev. B 45, 679 (1992).
  HMC for molecular systems.
- A. Dullweber, B. Leimkuhler, R. McLachlan, J. Chem. Phys. 107, 5840
  (1997). Symplectic splitting for rigid bodies.
- T. F. Miller III et al., J. Chem. Phys. 116, 8649 (2002). NO_SQUISH.
- A. M. Horowitz, Phys. Lett. B 268, 247 (1991). Generalised HMC.
- M. Tuckerman, B. J. Berne, G. J. Martyna, J. Chem. Phys. 97, 1990
  (1992). r-RESPA multiple time stepping.
- A. Beskos, N. Pillai, G. Roberts, J.-M. Sanz-Serna, A. Stuart, Bernoulli
  19, 1501 (2013). Optimal HMC acceptance.

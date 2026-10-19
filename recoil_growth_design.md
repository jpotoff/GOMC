# Recoil Growth for GOMC CBMC — Design

## Goal

Remove the "look-ahead" loss in CBMC growth. On the all-atom hexane GEMC test
(2 molecules, near-ideal gas, ideal swap acceptance 50%), swap acceptance is
now 13.6% after the Jacobian–Gaussian branch sampler and fresh-hedron-per-trial
changes. Diagnostics showed that the remaining ~3.5× loss is not local:

- At the same growth step with the same set of placed partner atoms, the best
  of 9 new trials has a 1-5+ energy above 500 K in 46–48% of steps, versus
  6–7% for the old configuration's random trials.
- More trials per step do not help (`CBMC_Nth` 10 → 200 is flat), and moving
  1-5+ energies into torsion selection recovers only 1.1–1.4×.
- The clashes are H···H contacts with atoms placed by *earlier* growth steps.

Sequential CBMC commits each step without knowing whether later atoms will
fit. Recoil growth (RG) [1, 2] keeps a small search tree: a trial is only
selected if growth can continue a fixed number of steps beyond it, and a
blocked step backs up and retries the previous one. Its weights are exact, so
detailed balance holds.

Target: ideal-gas hexane swap acceptance > 40% (80% of the 50% ceiling)
at comparable cost per swap, with no change in sampled ensembles.

---

## Background: how growth works today

`DCGraph::Build` (swap, intra-swap, GCMC insertion/deletion) and
`DCGraph::Regrowth` grow a molecule as a sequence of components:

1. a starting node (`DCFreeHedron`: first atom + its branch, random orientation);
2. edges in a random fringe order (`DCLinkedHedron`: branch on the next atom).

For each component the new and old configurations are built back to back
(`PrepareNew/BuildNew/PrepareOld/BuildOld`), so the growth order is the same
for both. Each component multiplies its Rosenbluth factor into
`TrialMol::totalWeight` and commits its atoms with `AddAtom`/`AddBonds`.
Nothing can be un-committed: `TrialMol` has no `RemoveAtom`, and
`Bonds::AddBond` has no inverse.

Inside `DCLinkedHedron::BuildNew` each of `CBMC_Nth` LJ trials is: a fresh
branch geometry (weight from `CBMC_Ang` JG trials), a torsion selected from
`CBMC_Dih` trials (weight = torsion sum), then intermolecular + 1-5+ energy.

---

## The algorithm

### Standard recoil growth (continuous potentials) [1, 2]

Growth steps i = 1…n in a fixed order. Parameters: k trials per node, recoil
length l.

- A trial t at step i is **open** with probability p(t) = min(1, e^{-βu(t)}),
  where u(t) is its nonbonded energy with the atoms placed so far (one coin
  flip per trial).
- A node at step i is **alive** if it can be extended l steps (to step i+l, or
  to the end of the molecule) through open trials. The search is depth-first
  and stops as soon as one path of length l is found.
- The trial chosen at step i is chosen uniformly at random among the feasible
  trials (equivalently, the first feasible trial in a random order). Growth
  then moves to step i+1 from that trial.
- After the chain is complete, for each step i the number of **feasible**
  trials m_i (open and alive, out of k) is counted; this requires finishing
  the alive-test for the trials not examined during the search.

The configuration is generated with probability ∝ Π_i p_i / m_i, so

    W = Π_i  m_i · e^{-βu_i} / p_i ,        acc(o→n) = min(1, W(n) / W(o))

where u_i, p_i refer to the trial chosen at step i. For the old
configuration, the actual placement is trial 1 at every step (it is feasible
by construction), k−1 trials are generated, and their feasibility is tested
the same way.

### Generalization to GOMC's importance-weighted trials

GOMC trials are not drawn exactly from the bonded distribution: each carries a
bonded importance weight a(t) (JG branch weight × torsion Rosenbluth sum).
The natural generalization replaces the Boltzmann factor by the trial's full
weight,

    w(t) = a(t) e^{-βu(t)},   p(t) = min(1, w(t) / C),   W = Π_i m_i · w_i / p_i ,

with C a positive scale constant per step type (e.g. the running mean of a).
C affects efficiency only, not correctness. **This generalization must be
derived carefully and verified (Phase 0) before it is built into GOMC.**

### Mapping onto GOMC

- One RG **trial** at a step = what one LJ trial does today in
  `DCLinkedHedron` (fresh branch + torsion selection + nonbonded energy).
  `k` replaces `CBMC_Nth` for RG steps.
- The k trials at a node are generated and their energies evaluated **as a
  batch** (one `ParticleInter`/`ParticleNonbonded` call over k positions, as
  today), then the coin flips are drawn. This keeps the vectorized energy
  path and is equivalent to the sequential version.
- Phase 1–2 scope: the **starting node stays standard CBMC** (its weight
  multiplies in as now); RG runs over the edge steps. If every trial at the
  first RG step is dead, the move fails, as a CBMC dead end does today.
- The same random fringe order is used for the new and old configurations,
  as today.

### Cost

Per step, roughly k evaluations on the chosen path, plus alive-tests of
depth ≤ l for the other k−1 trials (worst case k^l, typically far fewer since
most subtrees succeed on the first open path). For hexane with k = 4–6 and
l = 2 the expected number of trial evaluations per growth is similar to
today's `CBMC_Nth` = 10. The old-side weight requires the same alive-tests.
Intramolecular evaluations are cheap; in dense liquids intermolecular
evaluations dominate, as they do now.

---

## Phase 0: Derivation and standalone prototype

### Problem

The generalized weights (importance-weighted trials, nested JG and torsion
weights) are not in the literature in this exact form, and a mistake in RG
weights silently biases the ensemble.

### Solution

- Write out P_gen and the acceptance rule for the generalized scheme,
  including the old-side feasibility test.
- Standalone C++ prototype on a toy chain (bonded terms + LJ between atoms
  ≥ 4 bonds apart, no box), reusing `BranchSampler`.
- Validate with a Markov-chain test like `BranchSamplerTest`: the stationary
  distribution of end-to-end distance, torsions and 1-5 distances must match
  an independent Metropolis reference for several (k, l), including
  importance-weighted trials.
- Measure acceptance versus (k, l) on an all-atom hexane-like toy to choose
  defaults and confirm the expected gain before touching GOMC.

#### Files

| File | Change |
|---|---|
| `test/src/RecoilGrowthTest.cpp` | [NEW] toy-chain validation |

---

## Phase 1: Trial interface for growth components

### Problem

Components build and commit in one call; RG needs to generate trials without
committing, evaluate them against the current partial molecule, place them,
and remove them again when it backs up.

### Solution

Add an optional trial interface to `DCComponent`, implemented first by
`DCLinkedHedron`:

```cpp
struct StepTrial {           // one placement of a step's atoms
  XYZ pos[MAX_BONDS];        // positions of the atoms placed by the step
  double bondedWeight;       // a(t): JG branch weight x torsion sum
  Energy energy;             // bonded, 1-3/1-4, 1-5+, inter, real
  // branch angles etc. needed to restore hedron state
};

virtual bool SupportsTrials() const { return false; }
// basis, prevPhi, rotation axis from the atoms already placed
virtual void BeginStep(TrialMol &mol, uint molIndex, bool old);
// generate k trials and evaluate their energies as a batch
virtual void GenerateTrials(TrialMol &mol, uint molIndex, uint k,
                            StepTrial *out);
// the old configuration's actual placement for this step
virtual void ActualTrial(TrialMol &oldMol, uint molIndex, StepTrial &out);
virtual void Place(TrialMol &mol, const StepTrial &t);
virtual void Unplace(TrialMol &mol);
```

`TrialMol` gains `RemoveAtom(i)` (clears `atomBuilt`) and a way to undo
`AddBonds` (e.g. `Bonds::RemoveBond`, or snapshot/restore of the bond list).
The existing `BuildNew/BuildOld` paths are re-expressed on top of the trial
interface, so standard CBMC and RG share one implementation.

#### Files

| File | Change |
|---|---|
| `src/cbmc/DCComponent.h` | trial interface (default: unsupported) |
| `src/cbmc/DCLinkedHedron.{h,cpp}` | implement interface; BuildNew/Old use it |
| `src/cbmc/TrialMol.{h,cpp}` | `RemoveAtom`, bond undo |
| `src/cbmc/Bonds` (in `TrialMol.h`) | `RemoveBond` or snapshot |

Verification: standard CBMC results are statistically unchanged (acceptance
and energies on hexane, ethane, pentane/hexane GCMC).

---

## Phase 2: Recoil growth engine for `DCGraph::Build`

### Solution

#### [NEW] `src/cbmc/RecoilGrowth.{h,cpp}`

- Takes the ordered list of steps (the fringe order, drawn once per move),
  k, l and C.
- `GrowNew(newMol)`: depth-first search with batch trial generation, coin
  flips, placement/removal; returns W(n) and leaves the chosen configuration
  placed in `newMol` with its energies added.
- `WeighOld(oldMol)`: walks the actual configuration, generating k−1 trials
  per step and testing their feasibility; returns W(o).
- Each tree node stores its k trials and open/alive flags, so a node is never
  re-evaluated during one growth.

#### [MODIFY] `src/cbmc/DCGraph.{h,cpp}`

- `Build`: starting node as now; then, if RG is enabled, collect the edge
  sequence first (the current fringe walk without building), call
  `GrowNew`/`WeighOld`, and multiply the weights into the trial molecules.
  Otherwise unchanged.

#### [MODIFY] `src/ConfigSetup.{h,cpp}`

New keywords (defaults keep current behavior):

| Keyword | Meaning | Default |
|---|---|---|
| `CBMC_Recoil` | enable recoil growth for DCGraph molecules | `false` |
| `CBMC_RecoilLength` | recoil length l | 2 |
| `CBMC_RecoilTrials` | trials per node k | 4 |

---

## Phase 3: Other entry points

`DCGraph::Regrowth` (partial regrowth from a fixed node) uses the same engine
on its edge sequence. Then `BuildGrowNew/Old` (MEMC) and `BuildGrowInCav`
(TargetedSwap, flexible). `DCFreeHedron` can implement the trial interface so
the starting node also joins the search.

---

## What This Plan Does NOT Do

- No change to `DCCyclic` (rings) or `DCLinear` (≤ 2 atoms).
- No change to the bonded samplers (`BranchSampler`, torsion selection).
- No change to rigid-insertion paths (`BuildIDNew/Old`, NeMTMC, MEMC rigid).
- No change for users who do not enable `CBMC_Recoil`.

---

## Risks and open questions

1. **Weight derivation** (Phase 0). The generalized open probability with
   importance-weighted trials must be proven and tested before integration.
2. **Search caps.** Dense phases can make the search tree large. A hard cap
   on nodes explored changes the generation probability; it is only exact if
   the same rule is applied identically on the old side, which needs care.
   Preferred: choose k and l so caps are never reached, and treat a cap hit as
   a fatal diagnostic during testing.
3. **Scale C.** Too small → most trials open (RG degenerates toward plain
   search); too large → few open trials. Efficiency only.
4. **Intermolecular batching.** Generating k trials per node as a batch keeps
   the current vectorized `ParticleInter`; per-trial evaluation would be
   slower.
5. **Interaction with cavity/seed options** (`SetSeed`, fixed COM) in MEMC
   and TargetedSwap — Phase 3.

---

## Expected Impact

| Test | Now | Target |
|---|---|---|
| Ideal-gas hexane swap (ceiling 50%) | 13.6% | > 40% |
| Regrowth acceptance, all-atom alkanes | baseline | improved (same mechanism) |
| Dense-liquid swaps, long chains | low | the main motivation of RG in the literature [1] |

---

## Verification Plan

1. Phase 0 toy-chain distribution test against an independent Metropolis
   reference, several (k, l), including importance-weighted trials.
2. Unit test of the GOMC engine on a small molecule with an exactly known
   distribution (Markov-chain test as in `BranchSamplerTest`).
3. Bonded-only ceiling test (`GOMC_NO15` scratch switch): RG must reach ~50%.
4. Full hexane test: acceptance and cost versus (k, l); mean intramolecular
   energies equal to standard CBMC within error.
5. Regression suite: ethane, TraPPE branched alkanes, pentane/hexane GCMC,
   OPC water, BPTI; standard CBMC unchanged when `CBMC_Recoil false`.
6. Dense test: hexane liquid–vapor GEMC; coexistence densities equal to
   standard CBMC within error; transfers accepted per CPU-hour.

---

## References

[1] S. Consta, N. B. Wilding, D. Frenkel, Z. Alexandrowicz, "Recoil growth: An
efficient simulation method for multi-polymer systems," J. Chem. Phys. 110,
3220 (1999).
[2] D. Frenkel, B. Smit, *Understanding Molecular Simulation*, 2nd ed.,
Chapter 13 (configurational-bias Monte Carlo; recoil growth section).

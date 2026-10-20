# CBMC-Seeded NeMTMC — Design

## Goal

Cut the dissipated work of NeMTMC insertions by placing and growing the
inserted molecule with CBMC at a starting coupling λ_b, and then running the
non-equilibrium schedule only from λ_b to 1.

**Test system.** UA TraPPE hexane, GCMC, 300 K, 460 molecules, local
relaxation, 50 windows × 50 steps, `CBMC_Ang 4`, `CBMC_Dih 20`. A pilot run
logging the work of each window gave:

| | Insertion | Deletion |
|---|---|---|
| βW, 5th–95th percentile | −6 to 40 | 8 to 15 |
| βW, median | 8–10 | 12 |

**The problem is insertion.**
- Total dissipation is about 12 kT on a uniform schedule and 10.7 kT on the
  optimized one. Nearly all of it is in insertions.
- The insertion work has a heavy high tail: molecules that start in a bad
  place and never recover.
- Acceptance is 4.3% (uniform) and 4.7% (optimized). Window spacing has gone
  as far as it can.
- Dissipation per window peaks at λ ≈ 0.1–0.35. That range is about half of
  the total.

Today the molecule enters the destination box as a rigid copy of its
source-box conformation, at a uniformly random position, at λ = 0. Replacing
the λ = 0 → λ_b part of the schedule with one CBMC growth:
- removes the dissipation in that range, because the CBMC step is
  equilibrium sampling with Rosenbluth weights;
- picks the starting position by Boltzmann weight;
- gives the molecule a conformation suited to its new surroundings.

**Limiting cases.** λ_b = 1 is a plain CBMC swap. λ_b = 0 is today's
NeMTMC, with ideal-chain regrowth in place of the rigid transfer. λ_b
becomes a tuning parameter between the two.

**Target.** Acceptance per CPU-hour at the best λ_b at least 2× today's
optimized-schedule NeMTMC on the hexane GCMC test, with no change in the
sampled ensemble.

---

## Pre-existing issue: the last relaxation step breaks time reversal

This was found while working out the acceptance rule below. It is now fixed
in `NEMTMC::Transform`, independently of this design.

`NEMTMC::Transform` ([NeMTMC.h](src/moves/NeMTMC.h)) runs M windows, each
"jump, then relax":

```
J1 R1 J2 R2 ... JM RM
```

The last relaxation, R_M, runs at the physical end state (λ_D = 1,
λ_S = 0), after all the work has been counted.

**What exact acceptance requires.** For coeff·e^(−βW) to be exact
(Crooks; Nilmeier et al., PNAS 2011), the reverse move must be able to
generate the time-reversed path. The time reverse of the path above starts
with a relaxation at the end state.

**What the reverse move actually does.** The reverse move (boxes swapped)
also starts with a jump. In terms of the forward states i = 0..M:
- the forward move relaxes at i = 1..M;
- the reverse move relaxes at i = 0..M−1.

The two sets are shifted by one.

**Why this biases sampling.** R_M does not enter the acceptance test, but it
is applied only when the move is accepted. Accepted end states arrive with
weight π(y)·a(y), where a(y) is the acceptance probability of the reverse
move from y. R_M preserves π, but not π·a. a(y) depends on the local
environment of the transferred molecule, which is exactly what R_M relaxes.
So balance is violated. The size of the bias is unknown, probably small, and
not zero.

**Fix.** Skip `Propagation()` after the last window:

```
J1 R1 ... J(M−1) R(M−1) JM
```

- **Why it works:** forward and reverse moves now both relax at i = 1..M−1,
  and the protocol is its own time reverse.
- **Within a window:** forward relaxes the source box, then the destination
  box. The reverse move relaxes its source (our destination) first, which is
  the correct reversed order.
- **Cost:** saves 1/M of the relaxation.
- **Change:** one line in the loop, for both the rigid path and the new
  CBMC path.

---

## Background: how NeMTMC works today

`NEMTMC::Transform`:
1. **Set λ to 0 in both boxes.** `BuildIDNew(newMolNEMT)` places a rigid
   copy at a random position and orientation in the destination box. Its
   weight is ignored.
2. **Move the molecule to the destination box** (`ShiftMolToDestBox`).
   Then call `BuildIDOld(oldMolNEMT)`, which also has no effect on
   acceptance.
3. **Run the windows,** for idx from M−1 down to 0:
   - λ_S = λ[idx] and λ_D = λ[M−idx];
   - `CalcEnNEMT` computes the inter-energy, tail-correction and Ewald
     changes of the jump;
   - `AddWork`, then `UpdateEnergy`;
   - `Propagation()` relaxes the source box, then the destination box.
4. **Move the intramolecular energy:** subtract the current source
   conformation's intra energy from the source box, and add the original
   conformation's intra energy to the destination box.

`Accept` uses coeff·e^(−βW). Intra energy is not in W, because the transfer
is rigid.

**Two facts the design relies on (checked in the code):**
- **CBMC already handles a fractional molecule.**
  `CalculateEnergy::ParticleInterTemplate` scales pairs that involve
  `lambdaRef.GetMolIndex(box)` by its λ_VDW and λ_Coulomb. This is how the
  intra-swap and regrowth steps inside the relaxation work at partial λ
  today.
- **The builders needed already exist.** `DCGraph::BuildNew` and
  `DCGraph::BuildOld` grow only the new or only the old configuration, each
  with its own random starting node and order. MEMC uses them. Independent
  orders keep super-detailed balance, because each order is chosen uniformly
  and independently.

---

## The algorithm

Let μ_0 < μ_1 < … < μ_M be the `LambdaVDW` list, with μ_0 = λ_b and
μ_M = 1. `LambdaCoulomb` is handled in the same way.

**Forward move** (molecule goes from S to D):

1. **CBMC insertion at λ_b.**
   - Set λ_S = 1 and λ_D = λ_b, with the molecule still registered in S.
   - Call `BuildNew(newMol)` in D. Its inter energies are scaled by λ_b, and
     the result is the Rosenbluth weight W_n.
   - If W_n < `SMALL_WEIGHT`, reject now. The acceptance is exactly 0, so
     this is exact, and it skips the whole schedule.
   - Add the chosen configuration's inter energy at λ_b to D's energy.
   - Add the tail-correction and Ewald (self, correction, reciprocal)
     changes for D going from 0 to λ_b to the work. CBMC trials see only
     real-space terms, the same as in `MoleculeTransfer`.
   - Move the molecule to D.
2. **Windows.** Run the existing loop unchanged:
   - λ_S goes μ_M → μ_0 and λ_D goes μ_0 → μ_M;
   - work is accumulated as today;
   - there is no relaxation after the last window.
3. **CBMC removal at λ_b.**
   - S is now at λ_S = μ_0 = λ_b. Set `oldMol` to the current (relaxed)
     source coordinates.
   - Call `BuildOld(oldMol)` in S at λ_b, with the molecule registered in D
     so it doesn't see itself. The result is W_o.
   - Add the tail-correction and Ewald changes for S going from λ_b to 0 to
     the work, and remove S's inter energy at λ_b.
4. **Intramolecular energy.** Bookkeeping as today:
   - subtract the current source conformation's intra energy from S;
   - add the intra energy of the CBMC-grown conformation, recorded at
     step 1, to D.

**Acceptance:**

```
P_acc = min(1, coeff · (W_n / W_o) · exp(−β W))
```

- coeff is the existing `GetCoeff()`.
- W is all the work: the end terms from steps 1 and 3 plus the window jumps.
- W_n and W_o include intra energy, as in a CBMC swap.

**Reverse move.** The reverse move is the same algorithm with the boxes
swapped:
- its CBMC insertion is the reverse of our step 3;
- its windows retrace ours;
- its CBMC removal is the reverse of our step 1.

**Why it is exact.** Write out the path probability: CBMC selection of the
new configuration, then the relaxation kernels (each in detailed balance at
its fixed λ pair), then generation of the old configuration's trial set.
Dividing by the probability of the reverse path gives the ratio above. This
is the standard super-detailed-balance argument for CBMC, combined with the
NCMC path argument. The symmetric protocol from the section above is
required.

**Limiting cases** (to be used as tests):
- `LambdaVDW 1.0` (one entry, no windows): W = tail-correction and Ewald
  terms only. This is exactly `MoleculeTransfer`.
- λ_b = 0: W_n/W_o is a ratio of ideal-chain Rosenbluth weights, and the
  windows are today's.

**GCMC reservoir.** The reservoir box is at or above `BOXES_WITH_U_NB`, so
its W is the ideal-chain weight, exactly as in a CBMC swap. As today, the
reservoir needs regrowth moves to keep its conformations ideal-gas
distributed.

---

## Changes

### [MODIFY] `src/ConfigSetup.{h,cpp}`
- **New keyword:** `NeMTMCCBMCStart  true|false`, default false. False keeps
  today's rigid path bit-for-bit, apart from the end-relaxation fix.
- **λ_b source:** taken from `LambdaVDW[0]` and `LambdaCoulomb[0]`.
- **Validation when true:** the lists must be strictly increasing, with
  last entry 1 and first entry ≥ 0.
- **Validation when false:** require the first entry to be 0. Today's code
  doesn't check this, but the rigid path assumes it.

### [MODIFY] `src/moves/NeMTMC.h`
- **`Transform`:**
  - add a CBMC-start branch for steps 1 and 3;
  - keep the rigid branch as is;
  - share the window loop between the two branches, which already skips the
    last `Propagation()`.
- **New state:** W_n, W_o, the end-step tail-correction and Ewald terms, and
  an early-reject flag.
- **New helper `CalcEndTerms(box, λ_from, λ_to, mol)`:** factors the
  tail-correction and Ewald pieces out of `CalcEnNEMT`, so the end steps
  reuse the same code.
- **`Accept`:** multiply by W_n/W_o when CBMC start is on. The rejection path
  is unchanged, since it already restores all coordinates and energies.
- **Early rejection:** returns a reject state from `Transform` that `Accept`
  already handles, as in `MoleculeTransfer`.

### No change
CBMC code (`src/cbmc/`), relaxation moves, local relaxation.

---

## What this does NOT do
- It does not tune λ_b automatically. λ_b is chosen by a scan (see
  Verification).
- It does not add delayed-acceptance pre-screening on W_n. A two-stage test
  needs a first-stage ratio that is reversible on its own, and W_n alone is
  not.
- It does not change the λ spacing. The schedule above λ_b can be
  re-optimized with the pilot-run script afterwards.

---

## Risks and open questions
1. **Soft-core at small λ_b.** CBMC discriminates only if overlaps are
   penalized. With `ScaleAlpha 0.5`, small λ_b may let beads sit inside
   neighbours, leaving the windows to push them out. The scan will show
   where CBMC stops helping.
2. **Self-exclusion in `BuildOld`.** The old configuration's trials must not
   interact with the molecule itself. It is registered in D at step 3. This
   must be verified in the CBMC energy path: the tail-correction counts
   (`kCount`) and the cell list both need checking.
3. **Ewald with charged molecules.** The end-step reciprocal and self terms
   follow `CalcEnNEMT`'s linear λ_Coulomb scaling. This needs a test on a
   charged system, as hexane is uncharged.
4. **Cost.** One CBMC growth costs about 1 ms, against an attempt of about
   0.4 s, so it is negligible. Early rejection saves time.
5. **Gain is uncertain.** It depends on how much of the insertion dissipation
   comes from bad starting points rather than inherent lag. Only the scan can
   tell.

---

## Verification
1. **Regression.** With `NeMTMCCBMCStart false`, the existing GCMC and GEMC
   GTests pass. No GTest covers NeMTMC, so add one for the rigid path and one
   for the CBMC path.
2. **Swap limit.** `NeMTMCCBMCStart true` with `LambdaVDW 1.0` must match a
   `SwapFreq`-only run (same CBMC trials) on acceptance and ⟨N⟩, within error,
   on the GEMC hexane test.
3. **λ_b scan** (local, efficiency only):
   - λ_b ∈ {0, 0.1, 0.2, 0.3, 0.5}, same total relaxation budget, about
     3,000 attempts each, 2 seeds;
   - windows above λ_b taken from the optimized schedule;
   - report acceptance, ⟨βW⟩ by direction, and accepted transfers per
     CPU-hour against today's rigid start.
4. **Correctness** (your cluster): GCMC P(N,E) for CBMC swap only, rigid
   NeMTMC with the fix, and CBMC-seeded NeMTMC at the best λ_b. Each
   replicate gets its own seed and no shared checkpoint.

---

## References
- J. P. Nilmeier, G. E. Crooks, D. D. L. Minh, J. D. Chodera, "Nonequilibrium
  candidate Monte Carlo is an efficient tool for equilibrium simulation,"
  PNAS 108, E1009 (2011).
- G. E. Crooks, "Entropy production fluctuation theorem and the
  nonequilibrium work relation for free energy differences," Phys. Rev. E 60,
  2721 (1999).
- D. Frenkel and B. Smit, *Understanding Molecular Simulation*, 2nd ed.,
  ch. 13 (CBMC and super-detailed balance).

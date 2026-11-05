# Sampling Instrumentation

Diagnostics for CBMC and molecule-transfer moves. They show why moves are
rejected and where the problem lies: in finding a cavity, in growing a
conformation, in the trial counts, or in the method itself.

## Turning it on

Instrumentation is compiled in only on request, so production binaries carry
no cost.

```
./metamake.sh -A native -i      # builds into bin_instrument/; bin/ is untouched
```

or, with CMake directly, `-DGOMC_INSTRUMENT=on`. In the config file:

```
Instrument  true  [frequency]  [categories...]
```

| Setting | Meaning |
|---|---|
| `frequency` | Steps between rewrites of the summary file, with cumulative totals. `0` or omitted means write only at the end of the run. |
| `categories` | Any of `rejection`, `lnacc`, `weights`, `nemtmc`, `nemtwork`, `all`. The default is `all`, which is every category except `nemtwork`. |

A binary built without `-i` accepts the keyword, prints a warning and ignores
it.

**Output files:**
- `<OutputName>_instrument.dat`: the summary, in sections described below.
- `<OutputName>_nemtwork.dat`: only with `nemtwork`. One line per NeMTMC
  attempt, giving the work of every λ window.

**Report:**

```
python3 scripts/instrument_report.py run_instrument.dat [--all] [--schedule N]
```

The report prints a diagnosis for every move, box and molecule kind.
- `--all` adds displacement and rotation moves.
- `--schedule N` prints an N-window NeMTMC λ schedule at equal thermodynamic
  length. It needs GCMC data.

## What is recorded

Every row is keyed by move, box, molecule kind and `nested`. `nested` = 1
marks moves run inside another move, such as the relaxation moves inside
NeMTMC, so they don't mix with the top-level statistics.

### `[rejection]`: why each attempt ended

| Reason | Meaning |
|---|---|
| `accepted` | Accepted. |
| `metropolis` | The acceptance ratio was finite and the attempt lost the random draw. |
| `overlap_first` | Every first-stage trial was inside `RcutLow`. |
| `overlap_later` | Every trial of a later growth stage was inside `RcutLow`. |
| `zerow_first` | Every first-stage trial weight was zero (energy above about 745 kT), but not all trials overlapped. |
| `zerow_later` | The same, at a later growth stage. |
| `overlap` | A move-level overlap with no stage that failed outright: rigid moves, MEMC, or an overlapping trial picked from nonzero weights. |
| `lowweight_first` | No stage failed outright, but the total weight fell below `SMALL_WEIGHT` (1e-38). It is attributed to the stage with the smallest mean trial weight, and here that is the first stage. |
| `lowweight_later` | The same, with a later growth stage as the bottleneck. |
| `early` | Rejected before trials were generated. The code is broken down in `[early]`, for example `outside_relax_radius` for local NeMTMC relaxation. |

The *first* stage is the placement of the first atom (`DCSingle`) or of a
rigid molecule's position and orientation (`DCRotateCOM`).

**How to read it:**
- **First-stage failures dominate:** this is a cavity problem. More trials
  help little. Consider CFCMC, MEMC, NeMTMC, or cavity-biased insertion.
- **Later-stage failures dominate:** this is a conformational problem. More or
  better CBMC trials (`CBMC_Nth`, `CBMC_Ang`, `CBMC_Dih`) or recoil growth
  should help.

### `[lnacc]`: distance from threshold

This section is a histogram of ln(acceptance ratio) for attempts that had a
finite ratio.
- **Bins:** width 1. Values ≤ −50 go into the first bin and values ≥ 9 into
  the last.
- **`ratio_zero`:** counts the structural rejections, which have no finite
  ratio.

**How to read it:**
- **Most mass within a few units of 0:** the move is near threshold, and
  tuning pays off.
- **Most mass far below −20:** the move is orders of magnitude away, and it
  needs a different method.

### `[widom]`, `[weights]`, `[stages]`: Rosenbluth diagnostics

- **`[widom]`:** ln⟨W_new⟩ over all attempts of insertion-type moves (swap,
  intra-swap, targeted swap), zero weights included, together with
  `ess` = (ΣW)²/ΣW², the effective number of insertions.
  - **Comparing boxes:** the difference between two boxes estimates the
    difference in βμ_ex.
  - **When ess is small:** the estimate is dominated by a few insertions and
    is biased low.
- **`[weights]`:** histograms of ln W_new and ln W_old, and of the
  intermolecular energy βU of the new and old trial molecules.
- **`[stages]`:** one row per stage type (first or later) of the new
  configuration, with:
  - the fraction of trial sets that were all zero or all overlapping;
  - the mean fraction of overlapping trials;
  - **`mean_best_frac`:** the share of the total weight carried by the single
    best trial. A histogram of this share is included, along with the fraction
    of sets where it exceeds 0.9. Values near 1 mean the trial count is too
    small (`CBMC_First` or `CBMC_Nth`).
  - **`mean_ln_w`:** the mean ln(mean trial weight). The most negative stage
    limits the Rosenbluth weight.
- **`[stage_position]`:** ln(mean trial weight) by position in the growth
  sequence, for the new and old configurations, plus their paired difference
  (new − old). In swap moves the two are grown in the same order, so the
  difference shows which growth step loses weight.
  - **Deficit concentrated late in growth:** dead ends caused by earlier
    choices, a look-ahead problem. Recoil growth or a better growth order
    helps more than extra trials.
  - **Deficit concentrated early:** the first branches are poorly sampled.
    Check the trial counts and branch sampler for those steps.

### `[atom_split]`: which placed atom costs the weight

A growth step (`DCLinkedHedron`) places every atom bonded to its focus atom
together, as one trial. For example, in an all-atom alkane each step places
the next carbon and the hydrogens on the current carbon. This section splits
the energy of each trial among the atoms it places, to show whether growth is
limited by the backbone or by the atoms that ride along with it.

- **Classes:** an atom is *H* when its type name starts with `H`, and *heavy*
  otherwise. United-atom sites are heavy.
- **Energies:** βu per atom. *Inter* is the intermolecular plus real-space
  Coulomb energy and *intra* is the 1-5+ nonbonded energy. The bonded
  importance weights (branch geometry and torsion) are left out.
- **Dead trial:** total βu above 20 (weight below e⁻²⁰).
- **Scope:** new configuration only, one row per move, box and kind. It is
  written with the `weights` category, and `instrument_report.py` prints it
  under each swap and regrowth move with a one-line diagnosis.

| Column | Meaning |
|---|---|
| `sets`, `trials` | Growth steps recorded and trials in them. |
| `heavy_per_trial`, `H_per_trial` | Atoms placed per trial, by class. |
| `dead_frac` | Fraction of trials that are dead. |
| `dead_heavy_frac` | Fraction dead on the heavy atoms' energy alone. |
| `dead_noHinter_frac` | Fraction still dead with the H intermolecular energy removed. |
| `kill_<class>_<term>` | Among dead trials, the fraction whose largest single-atom βu is that class and term. |
| `u_<term>_<class>` | Mean βu per atom over all trials. Each value is clipped to ±100, so overlaps don't swamp the mean. |
| `win_<term>_<class>` | The same mean over the selected trial of each step. |
| `atom_over_<class>` | Fraction of atoms whose own βu exceeds 20. |
| `ln_all`, `ln_heavy`, `ln_noHinter`, `ln_noHintra` | Mean per step of ln(mean over trials of e^−βu), counting all atoms, heavy atoms only, all but the H intermolecular energy, and all but the H 1-5+ energy. Each trial's βu is capped at 100. |
| `sets_dead`, `sets_dead_heavy` | Fraction of steps in which every trial is dead, counting all atoms or heavy atoms only. |

**How to read it:**
- **`ln_heavy` vs `ln_all`:** the difference is the weight the H atoms cost per
  growth step. Comparing `ln_noHinter` and `ln_noHintra` with `ln_all` shows
  whether that cost comes from neighbouring molecules or from the chain itself.
- **`dead_heavy_frac` close to `dead_frac`:** dead trials are dead because of
  the heavy atom. Placing the backbone is the problem, and more trials or a
  look-ahead are the levers.
- **`sets_dead` well above `sets_dead_heavy`:** the H atoms turn steps that
  had a usable backbone position into steps with none.
- **Comparing models:** `ln_heavy` of an all-atom model against `ln_all` of
  the united-atom model at a similar state point shows whether the all-atom
  backbone is any harder to grow than the united-atom chain.

**Caveats:**
- The cap at 100 makes the `ln_*` columns lower bounds on the loss when many
  steps have every trial dead. Read them together with `sets_dead`.
- Recoil growth steps (`CBMC_Recoil`) and the first placement
  (`DCFreeHedron`, `DCSingle`) are not recorded. Steps selected with the
  look-ahead are recorded, but the look-ahead factor is not part of βu.
- The counterfactual columns drop energies from the trials that were drawn,
  where each H's position was tied to its heavy atom's. They show what the H
  atoms cost those trials. They don't predict what a growth that ignored the
  H atoms would select, or how the later steps would change.

### `[nemtmc]`, `[nemtmc_windows]`: non-equilibrium transfer

- **`[nemtmc]`:** per destination box, the number of attempts and
  acceptances, and a histogram of the total work βW.
- **`[nemtmc_windows]`:** the mean and variance of each window's work
  increment.
  - **Dissipation (GCMC):** half the sum of the insertion and deletion means
    of a window estimates its dissipated work. `--schedule` uses this to
    space the windows.
  - **GEMC:** both boxes contribute to each window's work, so the split
    between them isn't available.

## Coverage

| Covered | Moves |
|---|---|
| Full: stages, weights, Widom | swap (`MOL_TRANSFER`), intra-swap, targeted swap, intra-targeted swap |
| Stages and weights, no Widom | regrowth, crankshaft |
| Reasons and ln ratio; stage failures from all trial molecules | MEMC variants |
| Reasons and ln ratio | displacement, rotation (useful for NeMTMC relaxation) |
| Reasons, ln ratio, window work | NeMTMC |

Not covered: multiparticle and volume moves.

Stage records come from the new configuration only. The old configuration
always contains the actual molecule, so it cannot fail.

## Overhead

| Build | Overhead |
|---|---|
| Built without `-i` | None: the hooks compile to nothing. |
| Built with `-i`, `Instrument false` | One branch per hook. |
| Built with `-i`, `Instrument true` | One pass over each trial set's weights, plus a few map lookups per move. With `weights` on, each growth step also copies its per-atom trial energies once (`[atom_split]`). Small next to the energy evaluations. |

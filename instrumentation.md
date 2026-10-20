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
| Built with `-i`, `Instrument true` | One pass over each trial set's weights, plus a few map lookups per move. Small next to the energy evaluations. |

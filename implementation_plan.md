# GOMC Energy Loop Optimization Plan

## Goal

Improve the performance of the main energy calculation loops in GOMC by eliminating virtual dispatch, removing redundant allocations, enabling compiler auto-vectorization, and improving memory access patterns — all without changing data structures, molecule topology, or checkpoint formats.

---

## Phase 1: Eliminate Virtual Dispatch via Templated Energy Kernels

> [!IMPORTANT]
> This is the highest-impact, lowest-risk change. It requires **no modifications** to existing code — only new templated wrappers and a one-time runtime dispatch.

### Problem

Every pairwise energy/force/virial evaluation in the inner loop calls through a virtual function pointer:

```cpp
// 52 virtual call sites in CalculateEnergy.cpp
forcefield.particles->CalcEn(distSq, kind1, kind2, lambda);
forcefield.particles->CalcCoulomb(distSq, kind1, kind2, qi_qj, lambda, box);
forcefield.particles->CalcVir(distSq, kind1, kind2, lambda);
```

The concrete type (`FF_SHIFT`, `FF_SWITCH`, `FF_SWITCH_MARTINI`, `FF_EXP6`, or base `FFParticle`) is chosen once at startup in [Forcefield.cpp:94–103](file:///home/ai8111/GOMC/tiles/GOMC/src/Forcefield.cpp#L94-L103) and never changes. But the compiler cannot see through the vtable, so it:
- Cannot inline the math
- Cannot auto-vectorize
- Cannot propagate constants (`rCutSq`, `sigmaSq[]`, `epsilon_cn[]`)
- Cannot eliminate dead branches (`electrostatic`, `ewald`, `lambda >= 0.999999`)

### Solution

Template the hot-path energy functions on the concrete forcefield type. Dispatch once at runtime.

#### [NEW] `src/CalculateEnergyInlined.h`

A new header containing templated implementations of the hot-path functions. These are **free functions** (not methods on `CalculateEnergy`) that accept the concrete FF type by `const` reference, enabling full inlining:

```cpp
template<typename FFType, bool Electrostatic, bool Ewald>
SystemPotential BoxInterImpl(const FFType& ff, const Forcefield& forcefield,
                             /* same args as BoxInter */);

template<typename FFType, bool Electrostatic, bool Ewald>
SystemPotential BoxForceImpl(const FFType& ff, const Forcefield& forcefield,
                             /* same args as BoxForce */);

template<typename FFType, bool Electrostatic, bool Ewald>
Virial VirialCalcImpl(const FFType& ff, const Forcefield& forcefield,
                      /* same args as VirialCalc */);

template<typename FFType, bool Electrostatic, bool Ewald>
bool MoleculeInterImpl(const FFType& ff, const Forcefield& forcefield,
                       /* same args as MoleculeInter */);

template<typename FFType, bool Electrostatic, bool Ewald>
void ParticleInterImpl(const FFType& ff, const Forcefield& forcefield,
                       /* same args as ParticleInter */);
```

By making `Electrostatic` and `Ewald` template `bool` parameters, the compiler will completely eliminate the dead branches (`if (electrostatic) { ... }`) at compile time in the non-electrostatic instantiation.

#### [MODIFY] [CalculateEnergy.cpp](file:///home/ai8111/GOMC/tiles/GOMC/src/CalculateEnergy.cpp)

Add a dispatch function that `static_cast`s `forcefield.particles` to the known concrete type and calls the templated implementation. This replaces the body of `BoxInter`, `BoxForce`, `VirialCalc`, `MoleculeInter`, and `ParticleInter`:

```cpp
SystemPotential CalculateEnergy::BoxInter(SystemPotential potential,
                                          XYZArray const &coords,
                                          BoxDimensions const &boxAxes,
                                          const uint box) {
  if (box >= BOXES_WITH_U_NB) return potential;

  // Dispatch once — the entire inner loop is now inlineable
  #define DISPATCH(FFType) \
    if (electrostatic && ewald) \
      return BoxInterImpl<FFType, true, true>( \
        static_cast<const FFType&>(*forcefield.particles), forcefield, \
        potential, coords, boxAxes, box, /* ... */); \
    else if (electrostatic) \
      return BoxInterImpl<FFType, true, false>( \
        static_cast<const FFType&>(*forcefield.particles), forcefield, \
        potential, coords, boxAxes, box, /* ... */); \
    else \
      return BoxInterImpl<FFType, false, false>( \
        static_cast<const FFType&>(*forcefield.particles), forcefield, \
        potential, coords, boxAxes, box, /* ... */);

  switch (forcefield.vdwKind) {
    case 0: DISPATCH(FFParticle); break;       // VDW_STD_KIND
    case 1: DISPATCH(FF_SHIFT); break;         // VDW_SHIFT_KIND
    case 2:                                     // VDW_SWITCH_KIND
      if (forcefield.isMartini) { DISPATCH(FF_SWITCH_MARTINI); }
      else { DISPATCH(FF_SWITCH); }
      break;
    case 3: DISPATCH(FF_EXP6); break;          // VDW_EXP6_KIND
  }
  #undef DISPATCH
  return potential;
}
```

This generates 5 × 3 = 15 instantiations of `BoxInterImpl` (5 FF types × 3 electrostatic/ewald combos). Each one is a fully specialized, inlineable, auto-vectorizable kernel with zero virtual dispatch.

#### Files Changed

| File | Change |
|------|--------|
| [NEW] `src/CalculateEnergyInlined.h` | Templated implementations of 5 hot-path functions |
| [MODIFY] `src/CalculateEnergy.cpp` | Replace function bodies with dispatch-to-template |
| No other files touched | — |

---

## Phase 2: Fix `GetNeighborList()` Return-by-Copy

### Problem

[GetNeighborList](file:///home/ai8111/GOMC/tiles/GOMC/src/CellList.cpp#L287-L289) returns `std::vector<std::vector<int>>` **by value**, deep-copying the entire neighbor list. Called 3× per energy cycle ([BoxInter:173](file:///home/ai8111/GOMC/tiles/GOMC/src/CalculateEnergy.cpp#L173), [BoxForce:297](file:///home/ai8111/GOMC/tiles/GOMC/src/CalculateEnergy.cpp#L297), [VirialCalc:430](file:///home/ai8111/GOMC/tiles/GOMC/src/CalculateEnergy.cpp#L430)).

### Solution

Return by `const` reference:

#### [MODIFY] [CellList.h](file:///home/ai8111/GOMC/tiles/GOMC/src/CellList.h#L38)

```diff
- std::vector<std::vector<int>> GetNeighborList(uint box) const;
+ const std::vector<std::vector<int>>& GetNeighborList(uint box) const;
```

#### [MODIFY] [CellList.cpp](file:///home/ai8111/GOMC/tiles/GOMC/src/CellList.cpp#L287-L289)

```diff
- std::vector<std::vector<int>> CellList::GetNeighborList(uint box) const {
+ const std::vector<std::vector<int>>& CellList::GetNeighborList(uint box) const {
    return neighbors[box];
  }
```

#### [MODIFY] [CalculateEnergy.cpp](file:///home/ai8111/GOMC/tiles/GOMC/src/CalculateEnergy.cpp) (3 sites)

Change the local variable declaration at each call site from:
```diff
- std::vector<std::vector<int>> neighborList;
- neighborList = cellList.GetNeighborList(box);
+ const std::vector<std::vector<int>>& neighborList = cellList.GetNeighborList(box);
```

#### Files Changed

| File | Change |
|------|--------|
| [MODIFY] `src/CellList.h` | Change return type to `const&` |
| [MODIFY] `src/CellList.cpp` | Change return type to `const&` |
| [MODIFY] `src/CalculateEnergy.cpp` | Change 3 local variable declarations to `const&` |

---

## Phase 3: Restructure Inner Loop into Filter-then-Compute

### Problem

The current inner loop in [BoxInter](file:///home/ai8111/GOMC/tiles/GOMC/src/CalculateEnergy.cpp#L205-L250) interleaves branching (cutoff check, same-molecule check, electrostatic check, charge-zero check) with heavy math (CalcEn, CalcCoulomb). This prevents the compiler from vectorizing the math portion, even after Phase 1 eliminates virtual dispatch.

### Solution

Split the inner loop into two passes inside the templated `BoxInterImpl`:

**Pass 1 — Filter:** Iterate over neighbor pairs, compute distances, apply cutoff and exclusion checks. Pack surviving pairs' data (distSq, kind indices, charge product) into thread-local scratch buffers. This pass has branches but they're simple comparisons.

**Pass 2 — Compute:** Iterate over the scratch buffer with straight-line arithmetic only. No branches. The compiler can auto-vectorize this loop with AVX2/AVX-512.

```cpp
// Thread-local scratch buffers (allocated once, reused per particle)
thread_local std::vector<double> buf_distSq(256);
thread_local std::vector<uint>   buf_idx(256);    // FlatIndex(kind1, kind2)
thread_local std::vector<double> buf_qProd(256);
thread_local std::vector<double> buf_lambda(256);

// Pass 1: Filter
int count = 0;
for (int nIdx = startIdx; nIdx < endIdx; nIdx++) {
    int nP = cellVector[nIdx];
    if (currP >= nP || particleMol[currP] == particleMol[nP]) continue;
    
    double distSq;
    XYZ virC;
    if (!boxAxes.InRcut(distSq, virC, coords, currP, nP, box)) continue;
    
    buf_distSq[count] = distSq;
    buf_idx[count]    = ff.FlatIndex(particleKind[currP], particleKind[nP]);
    buf_qProd[count]  = particleCharge[currP] * particleCharge[nP] * num::qqFact;
    buf_lambda[count] = GetLambdaVDW(particleMol[currP], particleMol[nP], box);
    count++;
}

// Pass 2: Compute (auto-vectorizable — no branches, no virtual calls)
for (int i = 0; i < count; i++) {
    double rRat2 = ff.sigmaSq[buf_idx[i]] / buf_distSq[i];
    double attract = rRat2 * rRat2 * rRat2;
    double repulse = pow(rRat2, ff.n[buf_idx[i]] * 0.5);
    tempLJEn += ff.epsilon_cn[buf_idx[i]] * (repulse - attract) 
                - ff.shiftConst[buf_idx[i]];
    // Coulomb (compiled away if Electrostatic=false)...
}
```

> [!NOTE]
> The `pow()` call with a non-integer exponent (`n[idx] * 0.5`) will limit SIMD throughput. For the common case of n=12 (standard LJ), we can specialize: `repulse = attract * attract` (since `(σ/r)^12 = ((σ/r)^6)^2`). This special case can be detected at init time.

### Handling Free Energy (Soft-Core) Pairs

Pairs where `lambda < 0.999999` use a completely different math path involving `cbrt()` and extra `pow()` calls. These are rare (typically only one molecule in the system has lambda != 1). The filter pass should route these to a separate scalar loop:

```cpp
if (buf_lambda[i] >= 0.999999) {
    // standard pair → stays in SIMD buffer
} else {
    // soft-core pair → separate buffer, processed with scalar code
}
```

#### Files Changed

| File | Change |
|------|--------|
| [MODIFY] `src/CalculateEnergyInlined.h` | Implement filter-then-compute pattern inside templated functions |

---

## Phase 4: Linearize `EnumerateLocal` for Single-Molecule Moves

### Problem

The bulk OpenMP loops (`BoxInter`, `BoxForce`, `VirialCalc`) already use the linearized cell list produced by [GetCellListNeighbor](file:///home/ai8111/GOMC/tiles/GOMC/src/CellList.cpp#L258-L285) (`cellVector` + `cellStartIndex`). However, single-molecule energy evaluations — the most frequently called functions during MC simulation — still use the linked-list traversal:

```cpp
// MoleculeInter, ParticleInter, TargetedSwap all do this:
CellList::Neighbors n = cellList.EnumerateLocal(pos, box);
while (!n.Done()) {
    nIndex.push_back(*n);  // collect into vector, then OpenMP over it
    n.Next();              // list[at] → pointer chase → cache miss
}
```

Each `n.Next()` follows a random pointer (`list[at]`), causing L1 cache misses for every particle in the neighborhood.

### Solution

Maintain a persistent linearized cell list alongside the existing linked list. Update it incrementally when `AddMol`/`RemoveMol` are called.

#### [MODIFY] [CellList.h](file:///home/ai8111/GOMC/tiles/GOMC/src/CellList.h)

Add new members:
```cpp
// Persistent linearized cell list (per box)
std::vector<int> cellParticles[BOX_TOTAL];   // particles sorted by cell
std::vector<int> cellStart[BOX_TOTAL];       // start index per cell
```

Add a new enumeration method:
```cpp
// Returns a pair of iterators (begin, end) into cellParticles for the
// neighborhood of a position
std::pair<const int*, const int*> 
    EnumerateLocalLinear(int cell, int box) const;
```

#### [MODIFY] [CellList.cpp](file:///home/ai8111/GOMC/tiles/GOMC/src/CellList.cpp)

- In `GridAll` / `GridBox`: after building the linked list, also build the linearized arrays (same logic as `GetCellListNeighbor` but stored persistently).
- In `AddMol` / `RemoveMol`: rebuild the linearized arrays for the affected cells. Since molecules are small (typically <100 atoms), this is fast.

#### [MODIFY] [CalculateEnergy.cpp](file:///home/ai8111/GOMC/tiles/GOMC/src/CalculateEnergy.cpp)

Replace the `EnumerateLocal` + `push_back` + `OpenMP` pattern in `MoleculeInter` and `ParticleInter` with direct iteration over the linearized arrays:

```cpp
// Before (linked-list → vector → OpenMP):
CellList::Neighbors n = cellList.EnumerateLocal(pos, box);
while (!n.Done()) { nIndex.push_back(*n); n.Next(); }

// After (direct contiguous iteration):
for (int nc = 0; nc < 27; nc++) {
    int nCell = neighborList[cell][nc]; // or flat: neighborList[cell * 27 + nc]
    const int* begin = &cellList.cellParticles[box][cellList.cellStart[box][nCell]];
    const int* end   = &cellList.cellParticles[box][cellList.cellStart[box][nCell + 1]];
    for (const int* p = begin; p != end; ++p) {
        nIndex.push_back(*p);
    }
}
```

> [!WARNING]
> The `EnumerateLocal` / `Neighbors` class is also used in [TargetedSwap.h](file:///home/ai8111/GOMC/tiles/GOMC/src/moves/TargetedSwap.h) and [IntraTargetedSwap.h](file:///home/ai8111/GOMC/tiles/GOMC/src/moves/IntraTargetedSwap.h). These must be updated as well. The old linked-list `EnumerateLocal` can be kept as a compatibility fallback until all callers are migrated.

#### Files Changed

| File | Change |
|------|--------|
| [MODIFY] `src/CellList.h` | Add linearized arrays + new enumeration method |
| [MODIFY] `src/CellList.cpp` | Build/maintain linearized arrays |
| [MODIFY] `src/CalculateEnergy.cpp` | Use linear iteration in `MoleculeInter`, `ParticleInter` |
| [MODIFY] `src/moves/TargetedSwap.h` | Migrate from `EnumerateLocal` to linear iteration |
| [MODIFY] `src/moves/IntraTargetedSwap.h` | Migrate from `EnumerateLocal` to linear iteration |

---

## What This Plan Does NOT Do

| Item | Why excluded |
|------|-------------|
| Flatten `neighbors` to 1D | 27 ints per cell, already in L1 cache. Negligible gain vs. complexity of updating GPU paths. |
| `invCellSize` multiply trick | One division per `PositionToCell` call. Not in the hot loop. |
| Molecule spatial sorting | High complexity, high risk (topology, checkpoint, MoleculeLookup), moderate gain. Can be added later as Phase 5 after the algorithmic wins are locked in. |
| Hand-written SIMD intrinsics | Unnecessary — Phases 1+3 enable the compiler to auto-vectorize. Intrinsics are fragile and non-portable. |
| GPU (`GOMC_CUDA`) changes | The `#ifdef GOMC_CUDA` paths already have their own optimized kernels. Our CPU-side changes don't affect them. |

---

## Expected Performance Impact

| Phase | Estimated Speedup | Risk | Effort |
|-------|-------------------|------|--------|
| 1. Template dispatch | **2–4×** | Very Low (additive, no existing code modified) | ~300 lines new |
| 2. Fix return-by-copy | **5–10%** | Zero | ~10 lines changed |
| 3. Filter-then-compute | **1.5–2×** (compounds with Phase 1) | Low | ~200 lines in new header |
| 4. Linearize EnumerateLocal | **1.2–1.5×** for MC moves | Medium | ~150 lines |

**Combined estimate: 3–8× overall speedup** on the CPU energy calculation path.

---

## Verification Plan

### Automated Tests
- Build GOMC (CPU-only, without `GOMC_CUDA`) and run the existing test suite after each phase.
- Add a regression test that compares total system energy (inter + intra + Ewald) to machine precision (±1e-10) against a baseline run, ensuring the templated kernels produce bit-identical results.
- For Phase 3 (filter-then-compute), verify that the standard-path and soft-core-path produce identical energies to the original interleaved loop using a free energy test case.

### Manual Verification
- Run a benchmark simulation (e.g., LJ fluid NVT, ~10,000 atoms, 10,000 steps) before and after each phase. Measure wall-clock time and verify energy drift is unchanged.
- Profile with `perf stat` to confirm:
  - Phase 1: reduction in branch mispredictions and instruction cache misses
  - Phase 3: increase in SIMD instruction counts (e.g., `fp_arith_inst_retired.256b_packed_double`)
  - Phase 4: reduction in L1 data cache misses during MC moves

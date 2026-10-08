/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#include "CalculateEnergy.h" //header for this

#include <algorithm>
#if defined(__AVX512F__) && defined(__AVX512VL__)
#include <immintrin.h>
#define GOMC_HAVE_COMPRESS 1
#endif
#include <cassert>

#include "BasicTypes.h" //uint
#include "BoxDimensions.h"
#include "BoxDimensionsNonOrth.h"
#include "Coordinates.h"
#include "EnergyTypes.h"          //Energy structs
#include "EnsemblePreprocessor.h" //Flags
#include "Ewald.h"                //for ewald calculation
#include "EwaldCached.h"          //for ewald calculation
#include "Forcefield.h"           //
#include "ForcefieldDispatch.h"   // resolve the concrete FF type once
#include "GeomLib.h"
#include "MoleculeKind.h"
#include "MoleculeLookup.h"
#include "NoEwald.h" //for ewald calculation
#include "NumLib.h"
#include "StaticVals.h" //For init
#include "System.h"     //For init
#include "TrialMol.h"
#ifdef GOMC_CUDA
#include "CalculateEnergyCUDAKernel.cuh"
#include "CalculateForceCUDAKernel.cuh"
#include "ConstantDefinitionsCUDAKernel.cuh"
#endif
#include "GOMCEventsProfile.h"
#define NUMBER_OF_NEIGHBOR_CELL 27
// How many distances pass 1 of BoxInterTemplate computes at a time. Cells hold
// ~150 (OPC water) to ~280 (OpenFF ethane) atoms, so this is one to two chunks
// per cell, and the scratch array stays L1-resident at any occupancy.
#define GOMC_DISTSQ_CHUNK 256

//
//    CalculateEnergy.cpp
//    Energy Calculation functions for Monte Carlo simulation
//    Calculates using const references to a particular Simulation's members
//    Brock Jackman Sep. 2013
//
//    Updated to use radial-based intermolecular pressure
//    Jason Mick    Feb. 2014
//

using namespace geom;

CalculateEnergy::CalculateEnergy(StaticVals &stat, System &sys)
    : forcefield(stat.forcefield), mols(stat.mol),
      currentCoords(sys.coordinates), currentCOM(sys.com),
      lambdaRef(sys.lambdaRef), atomForceRef(sys.atomForceRef),
      molForceRef(sys.molForceRef),
#ifdef VARIABLE_PARTICLE_NUMBER
      molLookup(sys.molLookup),
#else
      molLookup(stat.molLookup),
#endif
      currentAxes(sys.boxDimRef), cellList(sys.cellList) {
}

void CalculateEnergy::Init(System &sys) {
  uint maxAtomInMol = 0;
  calcEwald = sys.GetEwald();
  electrostatic = forcefield.electrostatic;
  ewald = forcefield.ewald;
  multiParticleEnabled = sys.statV.multiParticleEnabled;
  for (uint m = 0; m < mols.count; ++m) {
    const MoleculeKind &molKind = mols.GetKind(m);
    if (molKind.NumAtoms() > maxAtomInMol)
      maxAtomInMol = molKind.NumAtoms();
    for (uint a = 0; a < molKind.NumAtoms(); ++a) {
      particleKind.push_back(molKind.AtomKind(a));
      particleMol.push_back(m);
      particleCharge.push_back(molKind.AtomCharge(a));
      particleIndex.push_back(int(a));
    }
  }
#ifdef GOMC_CUDA
  InitCoordinatesCUDA(forcefield.particles->getCUDAVars(),
                      currentCoords.Count(), maxAtomInMol, currentCOM.Count());
#endif
}

SystemPotential CalculateEnergy::SystemTotal() {
  GOMC_EVENT_START(1, GomcProfileEvent::EN_SYSTEM_TOTAL);
  SystemPotential pot =
      SystemInter(SystemPotential(), currentCoords, currentAxes);

  // system intra
  for (uint b = 0; b < BOX_TOTAL; ++b) {
    GOMC_EVENT_START(1, GomcProfileEvent::EN_BOX_INTRA);
    double bondEnergy[2] = {0};
    double bondEn = 0.0, nonbondEn = 0.0, correction = 0.0;
    MoleculeLookup::box_iterator thisMol = molLookup.BoxBegin(b);
    MoleculeLookup::box_iterator end = molLookup.BoxEnd(b);
    std::vector<uint> molID;

    while (thisMol != end) {
      molID.push_back(*thisMol);
      ++thisMol;
    }

#ifdef _OPENMP
#pragma omp parallel for default(none) private(bondEnergy) shared(b, molID)    \
    reduction(+ : bondEn, nonbondEn, correction)
#endif
    for (int i = 0; i < (int)molID.size(); i++) {
      // calculate nonbonded energy
      MoleculeIntra(molID[i], b, bondEnergy);
      bondEn += bondEnergy[0];
      nonbondEn += bondEnergy[1];
      // calculate correction term of electrostatic interaction
      correction += calcEwald->MolCorrection(molID[i], b);
    }

    pot.boxEnergy[b].intraBond = bondEn;
    pot.boxEnergy[b].intraNonbond = nonbondEn;
    // calculate self term of electrostatic interaction
    pot.boxEnergy[b].self = calcEwald->BoxSelf(b);
    pot.boxEnergy[b].correction = correction;

    GOMC_EVENT_STOP(1, GomcProfileEvent::EN_BOX_INTRA);
    // Calculate Virial
    pot.boxVirial[b] = VirialCalc(b);
  }

  pot.Total();

  if (pot.totalEnergy.total > 1.0e12) {
    std::cout << "\nWarning: Large energy detected due to the overlap in "
                 "initial configuration.\n"
                 "         The total energy will be recalculated at EqStep to "
                 "ensure the accuracy \n"
                 "         of the computed running energies.\n";
  }

  GOMC_EVENT_STOP(1, GomcProfileEvent::EN_SYSTEM_TOTAL);
  return pot;
}

SystemPotential CalculateEnergy::SystemInter(SystemPotential potential,
                                             XYZArray const &coords,
                                             BoxDimensions const &boxAxes) {
  for (uint b = 0; b < BOXES_WITH_U_NB; ++b) {
    // calculate LJ interaction and real term of electrostatic interaction
    potential = BoxInter(potential, coords, boxAxes, b);
    // calculate reciprocal term of electrostatic interaction
    potential.boxEnergy[b].recip = calcEwald->BoxReciprocal(b, false);
  }

  potential.Total();

  return potential;
}

//
// Permute the per-atom arrays into cell-list order.
//
// cellVector already groups atom indices by cell and sorts them within each
// cell, so it is exactly the permutation we want; what it does not do is move
// the data. Every `coords.x[nParticle]` in the pair walk was therefore a
// gather, which is what kept the (already branchless) minimum-image arithmetic
// running one lane wide. Copying the data into this order once, O(N) per call,
// turns the inner loop into contiguous loads.
//
// The buffers are members and only grow, so steady state does not allocate.
//
void CalculateEnergy::BuildCellOrdered(
    XYZArray const &coords, const std::vector<int> &cellVector,
    const std::vector<int> &mapParticleToCell) {
  const int n = (int)cellVector.size();
  if ((int)cellOrderX.size() < n) {
    cellOrderX.resize(n);
    cellOrderY.resize(n);
    cellOrderZ.resize(n);
    cellOrderCharge.resize(n);
    cellOrderMol.resize(n);
    cellOrderKind.resize(n);
    cellOrderCell.resize(n);
  }
  for (int k = 0; k < n; ++k) {
    const int p = cellVector[k];
    cellOrderX[k] = coords.x[p];
    cellOrderY[k] = coords.y[p];
    cellOrderZ[k] = coords.z[p];
    cellOrderCharge[k] = particleCharge[p];
    cellOrderMol[k] = particleMol[p];
    cellOrderKind[k] = particleKind[p];
    cellOrderCell[k] = mapParticleToCell[p];
  }
}

//
// True when the cell list prunes nothing.
//
// ResizeGrid clamps the grid to 3 cells per side and sizes cells by the
// cutoff, and RebuildNeighbors uses a fixed +-1 stencil. So as soon as a box
// edge is under 4 cutoffs the grid is 3x3x3 and a cell's 27 neighbors are all
// 27 cells: the "neighbor" set is the entire box. Measured on the GEMC
// benchmarks, only 20.4% (OPC) and 7.7% (OpenFF ethane) of the enumerated
// pairs are inside the cutoff, which is exactly the cutoff-sphere/box volume
// ratio -- the signature of enumerating everything.
//
// Larger systems (BPTI at 70 A, the K channel at 80x80x132) do get a real
// grid and are unaffected by the paths this gates.
//
bool CalculateEnergy::StencilCoversBox(const uint box) const {
  return cellList.CellsInBox(box) == NUMBER_OF_NEIGHBOR_CELL;
}

//
// Pack the whole box into contiguous per-atom arrays.
//
// Costs one walk of the cell list. ParticleInterTemplate used to do that walk
// once per trial position and then gather currentCoords through the resulting
// index list for every pair; packing once per call amortizes the walk over all
// trials and turns the gathers into unit-stride loads, which is what lets
// DistSqRange vectorize.
//
// Buffers are members and only grow, so steady state does not allocate.
//
int CalculateEnergy::BuildBoxPacked(XYZArray const &coords,
                                    const uint box) const {
  const int cap = (int)coords.Count();
  if ((int)boxPackX.size() < cap) {
    boxPackX.resize(cap);
    boxPackY.resize(cap);
    boxPackZ.resize(cap);
    boxPackCharge.resize(cap);
    boxPackKind.resize(cap);
    boxPackMol.resize(cap);
  }
  int n = 0;
  // Cell 0's neighbors are every cell in the box; see StencilCoversBox().
  CellList::Neighbors it = cellList.EnumerateLocal(0, box);
  while (!it.Done()) {
    const int p = *it;
    boxPackX[n] = coords.x[p];
    boxPackY[n] = coords.y[p];
    boxPackZ[n] = coords.z[p];
    boxPackCharge[n] = particleCharge[p];
    boxPackKind[n] = particleKind[p];
    boxPackMol[n] = particleMol[p];
    ++n;
    it.Next();
  }
  return n;
}

//
// Interaction energy of one box, over the cell list.
//
// Two nested passes per neighbor cell rather than one:
//
//   1. DistSqRange over a contiguous run of j atoms -- straight-line
//      arithmetic, no branches, contiguous in and out, so it vectorises.
//      This is where 80-92% of the work is: measured on the OPC GEMC
//      benchmark only 20.4% of the distances computed here fall inside the
//      cutoff (7.7% for OpenFF ethane), because the cell stencil covers the
//      whole box in the dense phase.
//   2. a scalar scan of those distances for the pairs that survive, running
//      the pair kernels unchanged.
//
// The pair set and the order it is visited in are identical to the single
// fused loop this replaces, so the energies are bit-for-bit unchanged:
// cellVector is sorted within each cell, which makes `currParticle <
// nParticle` monotone in the slot index, so the same test becomes a starting
// bound found by upper_bound rather than a per-candidate compare.
//
template <bool HasLambda, bool HasCharge, typename BoxType, typename FFType>
void CalculateEnergy::BoxInterTemplate(
    const FFType &ff, XYZArray const &coords, const BoxType &boxAxes,
    const uint box, double &tempREn, double &tempLJEn,
    const std::vector<int> &cellVector,
    const std::vector<int> &cellStartIndex,
    const std::vector<int> &mapParticleToCell,
    const std::vector<std::vector<int>> &neighborList, const int nPacked) {

  // HasLambda is resolved by the caller from Lambda::HasFraction(box). Only
  // NeMTMC ever sets a fractional molecule, so in an ordinary simulation it is
  // false for the whole run and everything below folds away.
  const int fracMol = HasLambda ? lambdaRef.GetMolIndex(box) : -1;
  const double fracVDW =
      HasLambda ? lambdaRef.GetLambdaVDW(fracMol, box) : 1.0;
  const double fracCoul =
      HasLambda ? lambdaRef.GetLambdaCoulomb(fracMol, box) : 1.0;

  // Whole-box path; see the comment in BoxInter(). Same pair set as the cell
  // walk below -- every unordered pair once -- reached as a single j > i scan
  // over packed arrays instead of 27 ranges each needing an upper_bound.
  if (nPacked >= 0) {
    const double *const pX = boxPackX.data();
    const double *const pY = boxPackY.data();
    const double *const pZ = boxPackZ.data();
    const double *const pQ = boxPackCharge.data();
    const int *const pKind = boxPackKind.data();
    const int *const pMol = boxPackMol.data();
    const double rCutSqBox = boxAxes.rCutSq[box];

    // schedule(static, 1) because this loop is triangular: atom i does
    // nPacked-i-1 distances, so the default blocked schedule would hand the
    // first thread several times the work of the last. Round-robin over a
    // linearly decreasing profile balances to within one iteration's work and
    // costs nothing at runtime, unlike dynamic.
#if defined _OPENMP && _OPENMP >= 201511 // check if OpenMP version is 4.5
#pragma omp parallel for schedule(static, 1) default(none)                     \
    shared(boxAxes, ff) reduction(+ : tempREn, tempLJEn)                       \
    firstprivate(box, num::qqFact, fracMol, fracVDW, fracCoul, nPacked, pX,    \
                     pY, pZ, pQ, pKind, pMol, rCutSqBox)
#endif
    for (int i = 0; i < nPacked; i++) {
      const int currMol = pMol[i];
      const int currKind = pKind[i];
      const double currQ = pQ[i];
      const double xi = pX[i], yi = pY[i], zi = pZ[i];

      // Scratch, per thread, ~6 KiB total and L1-resident. A masked
      // compressing store writes only the active lanes, so the survivor
      // buffers never need more than m slots; the 8 extra are slack against
      // an off-by-one, not a requirement.
      double distSq[GOMC_DISTSQ_CHUNK];
      double sDistSq[GOMC_DISTSQ_CHUNK + 8];
      double sQ[GOMC_DISTSQ_CHUNK + 8];
      int sKind[GOMC_DISTSQ_CHUNK + 8];
      int sMol[GOMC_DISTSQ_CHUNK + 8];

      for (int base = i + 1; base < nPacked; base += GOMC_DISTSQ_CHUNK) {
        const int m = std::min(GOMC_DISTSQ_CHUNK, nPacked - base);

        // ---- pass 1: contiguous, branch-free, vectorisable ----
        boxAxes.BoxType::DistSqRange(distSq, xi, yi, zi, pX + base, pY + base,
                                     pZ + base, m, box);

        // ---- pass 2: compact the survivors ----
        //
        // The annotate this replaces had 7.84% of the symbol on the cutoff
        // branch alone -- taken for ~20% of candidates, close to the worst
        // case for a predictor -- plus 11.35% on the loop control around it.
        // Compacting first turns that into a mask and lets pass 3 run over a
        // dense array with no cutoff test at all.
        //
        // Compaction preserves order, so pass 3 sees the pairs in the same
        // sequence as the fused loop did and the sums are bit-for-bit equal.
        //
        // Written with intrinsics because the branchless `ns += keep` form
        // was tried first and icpx 2025.1 did not recognise it: no vcompress
        // in the object and 17% MORE instructions than the branchy original.
        int ns = 0;
        int k = 0;
#ifdef GOMC_HAVE_COMPRESS
        {
          const __m512d vRCut = _mm512_set1_pd(rCutSqBox);
          const __m256i vCurrMol = _mm256_set1_epi32(currMol);
          for (; k + 8 <= m; k += 8) {
            const int j = base + k;
            const __m512d d = _mm512_loadu_pd(distSq + k);
            const __m256i mol =
                _mm256_loadu_si256((const __m256i *)(pMol + j));
            const __mmask8 keep =
                _mm512_cmp_pd_mask(d, vRCut, _CMP_LT_OQ) &
                _mm256_cmpneq_epi32_mask(mol, vCurrMol);
            _mm512_mask_compressstoreu_pd(sDistSq + ns, keep, d);
            _mm512_mask_compressstoreu_pd(sQ + ns, keep,
                                          _mm512_loadu_pd(pQ + j));
            _mm256_mask_compressstoreu_epi32(
                sKind + ns, keep,
                _mm256_loadu_si256((const __m256i *)(pKind + j)));
            _mm256_mask_compressstoreu_epi32(sMol + ns, keep, mol);
            ns += _mm_popcnt_u32((unsigned)keep);
          }
        }
#endif
        for (; k < m; k++) {
          const int j = base + k;
          if (currMol == pMol[j])
            continue;
          if (!(rCutSqBox > distSq[k]))
            continue;
          sDistSq[ns] = distSq[k];
          sQ[ns] = pQ[j];
          sKind[ns] = pKind[j];
          sMol[ns] = pMol[j];
          ++ns;
        }

        // ---- pass 3: kernels over a dense run, no cutoff test ----
        for (int s = 0; s < ns; s++) {
          if constexpr (HasLambda) {
            double lambdaVDW = 1.0;
            double lambdaCoulomb = 1.0;
            if (currMol == fracMol || sMol[s] == fracMol) {
              lambdaVDW = fracVDW;
              lambdaCoulomb = fracCoul;
            }

            if constexpr (HasCharge) {
              const double qi_qj_fact = currQ * sQ[s] * num::qqFact;
              if (qi_qj_fact != 0.0) {
                tempREn += ff.FFType::CalcCoulomb(sDistSq[s], currKind,
                                                  sKind[s], qi_qj_fact,
                                                  lambdaCoulomb, box);
              }
            }
            tempLJEn +=
                ff.FFType::CalcEn(sDistSq[s], currKind, sKind[s], lambdaVDW);
          } else {
            if constexpr (HasCharge) {
              const double qi_qj_fact = currQ * sQ[s] * num::qqFact;
              if (qi_qj_fact != 0.0) {
                tempREn += ff.CalcCoulombFull(sDistSq[s], qi_qj_fact, box);
              }
            }
            tempLJEn += ff.CalcEnFull(sDistSq[s], currKind, sKind[s]);
          }
        }
      }
    }
    return;
  }

  // Cell-ordered views; see BuildCellOrdered().
  const double *const ordX = cellOrderX.data();
  const double *const ordY = cellOrderY.data();
  const double *const ordZ = cellOrderZ.data();
  const double *const ordQ = cellOrderCharge.data();
  const int *const ordMol = cellOrderMol.data();
  const int *const ordKind = cellOrderKind.data();
  const int *const ordCell = cellOrderCell.data();
  const int *const cellVec = cellVector.data();
  const int *const cellStart = cellStartIndex.data();
  const int nAtoms = (int)cellVector.size();
  const double rCutSqBox = boxAxes.rCutSq[box];

#if defined _OPENMP && _OPENMP >= 201511 // check if OpenMP version is 4.5
#pragma omp parallel for default(none)                                         \
    shared(boxAxes, ff, neighborList)                                          \
    reduction(+ : tempREn, tempLJEn)                                           \
    firstprivate(box, num::qqFact, fracMol, fracVDW, fracCoul,                 \
                     ordX, ordY, ordZ, ordQ, ordMol, ordKind, ordCell,         \
                     cellVec, cellStart, nAtoms, rCutSqBox)
#endif
  // loop over all particles
  for (int i = 0; i < nAtoms; i++) {
    const int currParticle = cellVec[i];
    const int currMol = ordMol[i];
    const int currKind = ordKind[i];
    const double currQ = ordQ[i];
    const double xi = ordX[i], yi = ordY[i], zi = ordZ[i];
    // find the which cell currParticle belong to
    const int currCell = ordCell[i];

    // Scratch, per thread, ~6 KiB and L1-resident; see the whole-box path.
    double distSq[GOMC_DISTSQ_CHUNK];
    double sDistSq[GOMC_DISTSQ_CHUNK + 8];
    double sQ[GOMC_DISTSQ_CHUNK + 8];
    int sKind[GOMC_DISTSQ_CHUNK + 8];
    int sMol[GOMC_DISTSQ_CHUNK + 8];

    // loop over currCell neighboring cells
    for (int nCellIndex = 0; nCellIndex < NUMBER_OF_NEIGHBOR_CELL;
         nCellIndex++) {
      // find the index of neighboring cell
      const int neighborCell = neighborList[currCell][nCellIndex];
      // find the ending index in neighboring cell
      const int endIndex = cellStart[neighborCell + 1];
      // `currParticle < nParticle` used to be tested once per candidate. The
      // slots of a cell are sorted by atom index, so it is monotone here:
      // everything from the first slot that passes onwards also passes.
      const int firstIndex =
          (int)(std::upper_bound(cellVec + cellStart[neighborCell],
                                 cellVec + endIndex, currParticle) -
                cellVec);

      for (int base = firstIndex; base < endIndex;
           base += GOMC_DISTSQ_CHUNK) {
        const int m = std::min(GOMC_DISTSQ_CHUNK, endIndex - base);

        // ---- pass 1: contiguous, branch-free, vectorisable ----
        boxAxes.BoxType::DistSqRange(distSq, xi, yi, zi, ordX + base,
                                     ordY + base, ordZ + base, m, box);

        // ---- pass 2: compact the survivors ----
        //
        // Same transform as the whole-box path above, and for the same reason;
        // see the comment there. Two things differ here and are worth watching
        // when this is measured: the ranges are per-cell, so they are shorter
        // than 256 and the 8-wide compress has a larger scalar tail; but only
        // ~11% of candidates survive at a 5^3 grid against ~20% in the
        // degenerate case, so proportionally more of the branchy scan goes
        // away.
        int ns = 0;
        int k = 0;
#ifdef GOMC_HAVE_COMPRESS
        {
          const __m512d vRCut = _mm512_set1_pd(rCutSqBox);
          const __m256i vCurrMol = _mm256_set1_epi32(currMol);
          for (; k + 8 <= m; k += 8) {
            const int j = base + k;
            const __m512d d = _mm512_loadu_pd(distSq + k);
            const __m256i mol =
                _mm256_loadu_si256((const __m256i *)(ordMol + j));
            const __mmask8 keep =
                _mm512_cmp_pd_mask(d, vRCut, _CMP_LT_OQ) &
                _mm256_cmpneq_epi32_mask(mol, vCurrMol);
            _mm512_mask_compressstoreu_pd(sDistSq + ns, keep, d);
            _mm512_mask_compressstoreu_pd(sQ + ns, keep,
                                          _mm512_loadu_pd(ordQ + j));
            _mm256_mask_compressstoreu_epi32(
                sKind + ns, keep,
                _mm256_loadu_si256((const __m256i *)(ordKind + j)));
            _mm256_mask_compressstoreu_epi32(sMol + ns, keep, mol);
            ns += _mm_popcnt_u32((unsigned)keep);
          }
        }
#endif
        for (; k < m; k++) {
          const int j = base + k;
          if (currMol == ordMol[j])
            continue;
          if (!(rCutSqBox > distSq[k]))
            continue;
          sDistSq[ns] = distSq[k];
          sQ[ns] = ordQ[j];
          sKind[ns] = ordKind[j];
          sMol[ns] = ordMol[j];
          ++ns;
        }

        // ---- pass 3: kernels over a dense run, no cutoff test ----
        for (int sIdx = 0; sIdx < ns; sIdx++) {
          if constexpr (HasLambda) {
            double lambdaVDW = 1.0;
            double lambdaCoulomb = 1.0;
            if (currMol == fracMol || sMol[sIdx] == fracMol) {
              lambdaVDW = fracVDW;
              lambdaCoulomb = fracCoul;
            }

            if constexpr (HasCharge) {
              const double qi_qj_fact = currQ * sQ[sIdx] * num::qqFact;
              if (qi_qj_fact != 0.0) {
                tempREn += ff.FFType::CalcCoulomb(sDistSq[sIdx], currKind,
                                                  sKind[sIdx], qi_qj_fact,
                                                  lambdaCoulomb, box);
              }
            }
            tempLJEn += ff.FFType::CalcEn(sDistSq[sIdx], currKind, sKind[sIdx],
                                          lambdaVDW);
          } else {
            // Same values, reached without the lambda tests; see
            // FFAdapter::CalcEnFull.
            if constexpr (HasCharge) {
              const double qi_qj_fact = currQ * sQ[sIdx] * num::qqFact;
              if (qi_qj_fact != 0.0) {
                tempREn += ff.CalcCoulombFull(sDistSq[sIdx], qi_qj_fact, box);
              }
            }
            tempLJEn += ff.CalcEnFull(sDistSq[sIdx], currKind, sKind[sIdx]);
          }
        }
      }
    }
  }
}

template <typename BoxType, typename FFType>
void CalculateEnergy::BoxForceTemplate(
    const FFType &ff, XYZArray const &coords, XYZArray &atomForce,
    XYZArray &molForce,
    const BoxType &boxAxes, const uint box, double &tempREn, double &tempLJEn,
    const std::vector<int> &cellVector, const std::vector<int> &cellStartIndex,
    const std::vector<int> &mapParticleToCell,
    const std::vector<std::vector<int>> &neighborList) {

  double *aForcex = atomForce.x;
  double *aForcey = atomForce.y;
  double *aForcez = atomForce.z;
  double *mForcex = molForce.x;
  double *mForcey = molForce.y;
  double *mForcez = molForce.z;
  int atomCount = atomForce.Count();
  int molCount = molForce.Count();

  const bool hasFraction = lambdaRef.HasFraction(box);
  const int fracMol = hasFraction ? lambdaRef.GetMolIndex(box) : -1;
  const double fracVDW =
      hasFraction ? lambdaRef.GetLambdaVDW(fracMol, box) : 1.0;
  const double fracCoul =
      hasFraction ? lambdaRef.GetLambdaCoulomb(fracMol, box) : 1.0;

#if defined _OPENMP && _OPENMP >= 201511 // check if OpenMP version is 4.5
#pragma omp parallel for default(none)                                         \
    shared(boxAxes, cellStartIndex, cellVector, coords, ff,                    \
               mapParticleToCell, neighborList)                                \
    firstprivate(box, atomCount, molCount, num::qqFact, hasFraction, fracMol,  \
                     fracVDW, fracCoul)                                        \
    reduction(+ : tempREn, tempLJEn, aForcex[ : atomCount],                    \
                  aForcey[ : atomCount], aForcez[ : atomCount],                \
                  mForcex[ : molCount], mForcey[ : molCount],                  \
                  mForcez[ : molCount])
#endif
  for (int currParticleIdx = 0; currParticleIdx < (int)cellVector.size();
       currParticleIdx++) {
    int currParticle = cellVector[currParticleIdx];
    int currMol = particleMol[currParticle];
    int currCell = mapParticleToCell[currParticle];

    for (int nCellIndex = 0; nCellIndex < NUMBER_OF_NEIGHBOR_CELL;
         nCellIndex++) {
      int neighborCell = neighborList[currCell][nCellIndex];

      int endIndex = cellStartIndex[neighborCell + 1];
      for (int nParticleIndex = cellStartIndex[neighborCell];
           nParticleIndex < endIndex; nParticleIndex++) {
        int nParticle = cellVector[nParticleIndex];
        int nMol = particleMol[nParticle];

        if (currParticle < nParticle && currMol != nMol) {
          double distSq;
          XYZ virComponents, forceLJ, forceReal;
          if (boxAxes.InRcut(distSq, virComponents, coords, currParticle,
                             nParticle, box)) {

            double lambdaVDW = 1.0;
            double lambdaCoulomb = 1.0;
            if (hasFraction) {
              if (currMol == fracMol || nMol == fracMol) {
                lambdaVDW = fracVDW;
                lambdaCoulomb = fracCoul;
              }
            }

            if (electrostatic) {
              double qi_qj_fact = particleCharge[currParticle] *
                                  particleCharge[nParticle] * num::qqFact;
              if (qi_qj_fact != 0.0) {
                tempREn += ff.FFType::CalcCoulomb(
                    distSq, particleKind[currParticle], particleKind[nParticle],
                    qi_qj_fact, lambdaCoulomb, box);
                // Calculating the force
                forceReal =
                    virComponents * ff.FFType::CalcCoulombVir(
                                        distSq, particleKind[currParticle],
                                        particleKind[nParticle], qi_qj_fact,
                                        lambdaCoulomb, box);
              }
            }
            tempLJEn += ff.FFType::CalcEn(
                distSq, particleKind[currParticle], particleKind[nParticle],
                lambdaVDW);
            forceLJ = virComponents * ff.FFType::CalcVir(
                                          distSq, particleKind[currParticle],
                                          particleKind[nParticle], lambdaVDW);
            aForcex[currParticle] += forceLJ.x + forceReal.x;
            aForcey[currParticle] += forceLJ.y + forceReal.y;
            aForcez[currParticle] += forceLJ.z + forceReal.z;
            aForcex[nParticle] += -(forceLJ.x + forceReal.x);
            aForcey[nParticle] += -(forceLJ.y + forceReal.y);
            aForcez[nParticle] += -(forceLJ.z + forceReal.z);
            mForcex[particleMol[currParticle]] += (forceLJ.x + forceReal.x);
            mForcey[particleMol[currParticle]] += (forceLJ.y + forceReal.y);
            mForcez[particleMol[currParticle]] += (forceLJ.z + forceReal.z);
            mForcex[particleMol[nParticle]] += -(forceLJ.x + forceReal.x);
            mForcey[particleMol[nParticle]] += -(forceLJ.y + forceReal.y);
            mForcez[particleMol[nParticle]] += -(forceLJ.z + forceReal.z);
          }
        }
      }
    }
  }
}

template <typename BoxType, typename FFType>
void CalculateEnergy::VirialCalcTemplate(
    const FFType &ff, const BoxType &boxAxes, const uint box, double &vT11,
    double &vT12,
    double &vT13, double &vT22, double &vT23, double &vT33, double &rT11,
    double &rT12, double &rT13, double &rT22, double &rT23, double &rT33,
    const std::vector<int> &cellVector, const std::vector<int> &cellStartIndex,
    const std::vector<int> &mapParticleToCell,
    const std::vector<std::vector<int>> &neighborList, const int nPacked) {

  const bool hasFraction = lambdaRef.HasFraction(box);
  const int fracMol = hasFraction ? lambdaRef.GetMolIndex(box) : -1;
  const double fracVDW =
      hasFraction ? lambdaRef.GetLambdaVDW(fracMol, box) : 1.0;
  const double fracCoul =
      hasFraction ? lambdaRef.GetLambdaCoulomb(fracMol, box) : 1.0;

  // Whole-box path; see StencilCoversBox() and the comment in BoxInter().
  // Same two-pass shape as BoxInterTemplate, but pass 1 keeps the
  // minimum-image components as well, because the virial contracts them
  // against the molecule centre-of-mass separation.
  if (nPacked >= 0) {
    const double *const pX = boxPackX.data();
    const double *const pY = boxPackY.data();
    const double *const pZ = boxPackZ.data();
    const double *const pQ = boxPackCharge.data();
    const int *const pKind = boxPackKind.data();
    const int *const pMol = boxPackMol.data();
    const double rCutSqBox = boxAxes.rCutSq[box];

    // schedule(static, 1): triangular loop, see BoxInterTemplate.
#if defined _OPENMP && _OPENMP >= 201511 // check if OpenMP version is 4.5
#pragma omp parallel for schedule(static, 1) default(none)                     \
    shared(boxAxes, ff) firstprivate(box, hasFraction, fracMol, fracVDW,       \
                                         fracCoul, nPacked, pX, pY, pZ, pQ,    \
                                         pKind, pMol, rCutSqBox)               \
    reduction(+ : vT11, vT12, vT13, vT22, vT23, vT33, rT11, rT12, rT13, rT22,  \
                  rT23, rT33)
#endif
    for (int i = 0; i < nPacked; i++) {
      const int currMol = pMol[i];
      const int currKind = pKind[i];
      const double currQ = pQ[i];
      const double xi = pX[i], yi = pY[i], zi = pZ[i];

      // Per thread, 8 KiB, L1-resident.
      double distSq[GOMC_DISTSQ_CHUNK];
      double dxs[GOMC_DISTSQ_CHUNK], dys[GOMC_DISTSQ_CHUNK],
          dzs[GOMC_DISTSQ_CHUNK];

      for (int base = i + 1; base < nPacked; base += GOMC_DISTSQ_CHUNK) {
        const int m = std::min(GOMC_DISTSQ_CHUNK, nPacked - base);

        // ---- pass 1: contiguous, branch-free, vectorisable ----
        boxAxes.BoxType::DistVecRange(distSq, dxs, dys, dzs, xi, yi, zi,
                                      pX + base, pY + base, pZ + base, m, box);

        // ---- pass 2: the pairs that survive ----
        for (int k = 0; k < m; k++) {
          const int j = base + k;
          const int nMol = pMol[j];
          if (currMol == nMol)
            continue;
          if (!(rCutSqBox > distSq[k]))
            continue;

          // distance between the centres of mass of the two molecules
          XYZ comC = currentCOM.Difference(currMol, nMol);
          comC = boxAxes.BoxType::MinImage(comC, box);

          double lambdaVDW = 1.0;
          double lambdaCoulomb = 1.0;
          if (hasFraction) {
            if (currMol == fracMol || nMol == fracMol) {
              lambdaVDW = fracVDW;
              lambdaCoulomb = fracCoul;
            }
          }

          if (electrostatic) {
            const double qi_qj = currQ * pQ[j];
            // skip particle pairs with no charge
            if (qi_qj != 0.0) {
              const double pRF = ff.FFType::CalcCoulombVir(
                  distSq[k], currKind, pKind[j], qi_qj, lambdaCoulomb, box);
              rT11 += pRF * (dxs[k] * comC.x);
              rT22 += pRF * (dys[k] * comC.y);
              rT33 += pRF * (dzs[k] * comC.z);
            }
          }

          const double pVF =
              ff.FFType::CalcVir(distSq[k], currKind, pKind[j], lambdaVDW);
          vT11 += pVF * (dxs[k] * comC.x);
          vT22 += pVF * (dys[k] * comC.y);
          vT33 += pVF * (dzs[k] * comC.z);
        }
      }
    }
    return;
  }

#if defined _OPENMP && _OPENMP >= 201511 // check if OpenMP version is 4.5
#pragma omp parallel for default(none) shared(                                 \
        cellStartIndex, cellVector, ff, mapParticleToCell, neighborList,       \
            boxAxes)                                                           \
    firstprivate(box, hasFraction, fracMol, fracVDW, fracCoul)                 \
    reduction(+ : vT11, vT12, vT13, vT22, vT23, vT33, rT11, rT12, rT13, rT22,  \
                  rT23, rT33)
#endif
  for (int currParticleIdx = 0; currParticleIdx < (int)cellVector.size();
       currParticleIdx++) {
    int currParticle = cellVector[currParticleIdx];
    int currMol = particleMol[currParticle];
    int currCell = mapParticleToCell[currParticle];

    for (int nCellIndex = 0; nCellIndex < NUMBER_OF_NEIGHBOR_CELL;
         nCellIndex++) {
      int neighborCell = neighborList[currCell][nCellIndex];

      int endIndex = cellStartIndex[neighborCell + 1];
      for (int nParticleIndex = cellStartIndex[neighborCell];
           nParticleIndex < endIndex; nParticleIndex++) {
        int nParticle = cellVector[nParticleIndex];
        int nMol = particleMol[nParticle];

        // make sure the pairs are unique and they belong to different molecules
        if (currParticle < nParticle && currMol != nMol) {
          double distSq;
          XYZ virC;
          if (boxAxes.InRcut(distSq, virC, currentCoords, currParticle,
                             nParticle, box)) {
            // calculate the distance between com of two molecules
            XYZ comC = currentCOM.Difference(currMol, nMol);
            // calculate the minimum image between com of two molecules
            comC = boxAxes.BoxType::MinImage(comC, box);

            double lambdaVDW = 1.0;
            double lambdaCoulomb = 1.0;
            if (hasFraction) {
              if (currMol == fracMol || nMol == fracMol) {
                lambdaVDW = fracVDW;
                lambdaCoulomb = fracCoul;
              }
            }

            if (electrostatic) {
              double qi_qj =
                  particleCharge[currParticle] * particleCharge[nParticle];

              // skip particle pairs with no charge
              if (qi_qj != 0.0) {
                double pRF = ff.FFType::CalcCoulombVir(
                    distSq, particleKind[currParticle], particleKind[nParticle],
                    qi_qj, lambdaCoulomb, box);
                // calculate the top diagonal of pressure tensor
                rT11 += pRF * (virC.x * comC.x);
                rT22 += pRF * (virC.y * comC.y);
                rT33 += pRF * (virC.z * comC.z);
              }
            }

            double pVF = ff.FFType::CalcVir(
                distSq, particleKind[currParticle], particleKind[nParticle],
                lambdaVDW);
            // calculate the top diagonal of pressure tensor
            vT11 += pVF * (virC.x * comC.x);
            vT22 += pVF * (virC.y * comC.y);
            vT33 += pVF * (virC.z * comC.z);
          }
        }
      }
    }
  }
}

// templates functions for intermolecular interactions for single
// molecule moves.  Replacing currentAxes with boxAxes
template <typename BoxType, typename FFType>
bool CalculateEnergy::MoleculeInterTemplate(const FFType &ff,
                                            Intermolecular &inter_LJ,
                                            Intermolecular &inter_coulomb,
                                            XYZArray const &molCoords,
                                            const uint molIndex, const uint box,
                                            const BoxType &boxAxes) const {
  double tempREn = 0.0, tempLJEn = 0.0;
  bool overlap = false;

  if (box < BOXES_WITH_U_NB) {
    GOMC_EVENT_START(1, GomcProfileEvent::EN_MOL_INTER);
    uint length = mols.GetKind(molIndex).NumAtoms();
    uint start = mols.MolStart(molIndex);
    // Check to see if we have a fract  ional molecule in the box outside
    // and identify it outside of the main energy loop.
    // we do this lookup once instead of each time through the loop.
    const bool hasFraction = lambdaRef.HasFraction(box);
    const int fracMol = hasFraction ? lambdaRef.GetMolIndex(box) : -1;
    const double fracVDW =
        hasFraction ? lambdaRef.GetLambdaVDW(fracMol, box) : 1.0;
    const double fracCoul =
        hasFraction ? lambdaRef.GetLambdaCoulomb(fracMol, box) : 1.0;

    // Whole-box path. This function walks the cell list twice per atom of the
    // moving molecule -- once at the old position to subtract, once at the new
    // one to add -- so for OPC water it was doing eight full-box pointer
    // chases per trial move, each feeding a scalar InRcut through a gathered
    // index list. When the stencil selects every cell all eight walks see the
    // same set, so one pack serves all of them and the arithmetic vectorizes.
    //
    // The loop stays parallel. I first made it serial, reasoning that a
    // region dispatching four iterations could not be worth its fork/join
    // against a kmp_flag_64::wait that was 15.8% of runtime. That was
    // backwards, and measuring said so: serial cost 7% of wall clock and
    // pushed the wait UP to 24.6%, because that symbol counts thread
    // *idleness*. Three threads spinning through a serial region is more
    // idleness, not less. The wait is a symptom of too little parallel work
    // per region in this system, and the cure is more parallelism, not less.
    if (StencilCoversBox(box)) {
      const int nPacked = BuildBoxPacked(currentCoords, box);
      const double *const pX = boxPackX.data();
      const double *const pY = boxPackY.data();
      const double *const pZ = boxPackZ.data();
      const double *const pQ = boxPackCharge.data();
      const int *const pKind = boxPackKind.data();
      const int *const pMol = boxPackMol.data();
      const double rCutSqBox = boxAxes.rCutSq[box];
      const double rCutLowSq = forcefield.rCutLowSq;

      double oldREn = 0.0, oldLJEn = 0.0, newREn = 0.0, newLJEn = 0.0;

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(boxAxes, ff, molCoords)          \
    firstprivate(box, molIndex, num::qqFact, length, start, hasFraction,       \
                     fracMol, fracVDW, fracCoul, nPacked, pX, pY, pZ, pQ,      \
                     pKind, pMol, rCutSqBox, rCutLowSq)                        \
    reduction(+ : oldREn, oldLJEn, newREn, newLJEn) reduction(| : overlap)
#endif
      for (uint p = 0; p < length; ++p) {
        // Scratch for pass 1, per thread. 2 KiB, L1-resident.
        double distSq[GOMC_DISTSQ_CHUNK];
        const uint atom = start + p;
        const int currKind = particleKind[atom];
        const double currQ = particleCharge[atom];

        // side 0: the old position, which this move removes.
        // side 1: the trial position, which it adds.
        for (int side = 0; side < 2; ++side) {
          const double xi = side ? molCoords.x[p] : currentCoords.x[atom];
          const double yi = side ? molCoords.y[p] : currentCoords.y[atom];
          const double zi = side ? molCoords.z[p] : currentCoords.z[atom];
          double sumREn = 0.0, sumLJEn = 0.0;

          for (int base = 0; base < nPacked; base += GOMC_DISTSQ_CHUNK) {
            const int m = std::min(GOMC_DISTSQ_CHUNK, nPacked - base);

            // ---- pass 1: contiguous, branch-free, vectorisable ----
            boxAxes.BoxType::DistSqRange(distSq, xi, yi, zi, pX + base,
                                         pY + base, pZ + base, m, box);

            // ---- pass 2: the pairs that survive ----
            for (int k = 0; k < m; k++) {
              if (!(rCutSqBox > distSq[k]))
                continue;
              const int j = base + k;

              double lambdaVDW = 1.0;
              double lambdaCoulomb = 1.0;
              if (hasFraction) {
                int nMol = pMol[j];
                if (molIndex == fracMol && nMol == fracMol) {
                  lambdaVDW = fracVDW * fracVDW;
                  lambdaCoulomb = fracCoul * fracCoul;
                } else if (molIndex == fracMol || nMol == fracMol) {
                  lambdaVDW = fracVDW;
                  lambdaCoulomb = fracCoul;
                }
              }

              // Only the trial position can overlap; the old one is where the
              // molecule already sits.
              if (side && distSq[k] < rCutLowSq) {
                overlap = true;
              }

              if (electrostatic) {
                const double qi_qj_fact = currQ * pQ[j] * num::qqFact;
                if (qi_qj_fact != 0.0) {
                  sumREn += ff.FFType::CalcCoulomb(distSq[k], currKind,
                                                   pKind[j], qi_qj_fact,
                                                   lambdaCoulomb, box);
                }
              }
              sumLJEn +=
                  ff.FFType::CalcEn(distSq[k], currKind, pKind[j], lambdaVDW);
            }
          }

          if (side) {
            newREn += sumREn;
            newLJEn += sumLJEn;
          } else {
            oldREn += sumREn;
            oldLJEn += sumLJEn;
          }
        }
      }

      tempREn = newREn - oldREn;
      tempLJEn = newLJEn - oldLJEn;
      GOMC_EVENT_STOP(1, GomcProfileEvent::EN_MOL_INTER);
      inter_LJ.energy = tempLJEn;
      inter_coulomb.energy = tempREn;
      return overlap;
    }

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(boxAxes, ff, molCoords)          \
    firstprivate(box, molIndex, num::qqFact, length, start, hasFraction,       \
                     fracMol, fracVDW, fracCoul)                               \
    reduction(+ : tempREn, tempLJEn) reduction(| : overlap)
#endif
    for (uint p = 0; p < length; ++p) {
      double pREn = 0.0, pLJEn = 0.0;
      uint atom = start + p;
      CellList::Neighbors n = cellList.EnumerateLocal(currentCoords[atom], box);

      std::vector<uint> nIndex;
      while (!n.Done()) {
        nIndex.push_back(*n);
        n.Next();
      }

      for (int i = 0; i < (int)nIndex.size(); i++) {
        double distSq = 0.0;
        XYZ virComponents;
        if (boxAxes.InRcut(distSq, virComponents, currentCoords, atom,
                           nIndex[i], box)) {

          double lambdaVDW = 1.0;
          double lambdaCoulomb = 1.0;
          if (hasFraction) {
            int nMol = particleMol[nIndex[i]];
            if (molIndex == fracMol && nMol == fracMol) {
              lambdaVDW = fracVDW * fracVDW;
              lambdaCoulomb = fracCoul * fracCoul;
            } else if (molIndex == fracMol || nMol == fracMol) {
              lambdaVDW = fracVDW;
              lambdaCoulomb = fracCoul;
            }
          }

          if (electrostatic) {
            double qi_qj_fact =
                particleCharge[atom] * particleCharge[nIndex[i]] * num::qqFact;

            if (qi_qj_fact != 0.0) {
              pREn += -ff.FFType::CalcCoulomb(
                  distSq, particleKind[atom], particleKind[nIndex[i]],
                  qi_qj_fact, lambdaCoulomb, box);
            }
          }
          pLJEn += -ff.FFType::CalcEn(
              distSq, particleKind[atom], particleKind[nIndex[i]], lambdaVDW);
        }
      }

      n = cellList.EnumerateLocal(molCoords[p], box);
      nIndex.clear();
      while (!n.Done()) {
        nIndex.push_back(*n);
        n.Next();
      }

      for (int i = 0; i < (int)nIndex.size(); i++) {
        double distSq = 0.0;
        XYZ virComponents;
        if (boxAxes.InRcut(distSq, virComponents, molCoords, p, currentCoords,
                           nIndex[i], box)) {
          double lambdaVDW = 1.0;
          double lambdaCoulomb = 1.0;
          if (hasFraction) {
            int nMol = particleMol[nIndex[i]];
            if (molIndex == fracMol && nMol == fracMol) {
              lambdaVDW = fracVDW * fracVDW;
              lambdaCoulomb = fracCoul * fracCoul;
            } else if (molIndex == fracMol || nMol == fracMol) {
              lambdaVDW = fracVDW;
              lambdaCoulomb = fracCoul;
            }
          }

          if (distSq < forcefield.rCutLowSq) {
            overlap |= true;
          }

          if (electrostatic) {
            double qi_qj_fact =
                particleCharge[atom] * particleCharge[nIndex[i]] * num::qqFact;

            if (qi_qj_fact != 0.0) {
              pREn += ff.FFType::CalcCoulomb(
                  distSq, particleKind[atom], particleKind[nIndex[i]],
                  qi_qj_fact, lambdaCoulomb, box);
            }
          }
          pLJEn += ff.FFType::CalcEn(
              distSq, particleKind[atom], particleKind[nIndex[i]], lambdaVDW);
        }
      }

      tempREn += pREn;
      tempLJEn += pLJEn;
    }
    GOMC_EVENT_STOP(1, GomcProfileEvent::EN_MOL_INTER);
  }

  inter_LJ.energy = tempLJEn;
  inter_coulomb.energy = tempREn;
  return overlap;
}

template <typename BoxType, typename FFType>
void CalculateEnergy::ParticleNonbondedTemplate(
    const FFType &ff, double *inter, cbmc::TrialMol const &trialMol,
    XYZArray const &trialPos,
    const uint partIndex, const uint box, const uint trials,
    const BoxType &boxAxes) const {
  if (box >= BOXES_WITH_U_B)
    return;

  GOMC_EVENT_START(1, GomcProfileEvent::EN_CBMC_INTRA_NB);
  const MoleculeKind &kind = trialMol.GetKind();
  const uint *partner = kind.sortedNB.Begin(partIndex);
  const uint *end = kind.sortedNB.End(partIndex);
  while (partner != end) {
    if (trialMol.AtomExists(*partner)) {
      for (uint t = 0; t < trials; ++t) {
        double distSq;
        if (boxAxes.InRcut(distSq, trialPos, t, trialMol.GetCoords(), *partner,
                           box)) {
          inter[t] += ff.FFType::CalcEn(
              distSq, kind.AtomKind(partIndex), kind.AtomKind(*partner), 1.0);
          if (electrostatic) {
            double qi_qj_fact = kind.AtomCharge(partIndex) *
                                kind.AtomCharge(*partner) * num::qqFact;

            if (qi_qj_fact != 0.0) {
              ff.FFType::CalcCoulombAdd_1_4(inter[t], distSq,
                                                       qi_qj_fact, true);
            }
          }
        }
      }
    }
    ++partner;
  }
  GOMC_EVENT_STOP(1, GomcProfileEvent::EN_CBMC_INTRA_NB);
}

template <bool HasCharge, typename BoxType, typename FFType>
void CalculateEnergy::ParticleInterTemplate(const FFType &ff, double *en,
                                            double *real,
                                            XYZArray const &trialPos,
                                            bool *overlap, const uint partIndex,
                                            const uint molIndex, const uint box,
                                            const uint trials,
                                            const BoxType &boxAxes) const {
  if (box >= BOXES_WITH_U_NB)
    return;

  GOMC_EVENT_START(1, GomcProfileEvent::EN_CBMC_INTER);
  // double tempLJ, tempReal;
  MoleculeKind const &thisKind = mols.GetKind(molIndex);
  uint kindI = thisKind.AtomKind(partIndex);
  double kindICharge = thisKind.AtomCharge(partIndex);
  // std::vector<uint> nIndex;

  const bool hasFraction = lambdaRef.HasFraction(box);
  const int fracMol = hasFraction ? lambdaRef.GetMolIndex(box) : -1;
  const double fracVDW =
      hasFraction ? lambdaRef.GetLambdaVDW(fracMol, box) : 1.0;
  const double fracCoul =
      hasFraction ? lambdaRef.GetLambdaCoulomb(fracMol, box) : 1.0;

  // Whole-box path. When the stencil selects every cell, each trial's
  // neighbor set is the same set -- the box -- so the cell-list walk that used
  // to run once per trial runs once per call, and the pair loop reads packed
  // arrays instead of gathering currentCoords through the chased index list.
  // That is what lets DistSqRange vectorize here, the way it already does in
  // BoxInterTemplate.
  if (StencilCoversBox(box)) {
    const int nPacked = BuildBoxPacked(currentCoords, box);
    const double *const pX = boxPackX.data();
    const double *const pY = boxPackY.data();
    const double *const pZ = boxPackZ.data();
    const double *const pQ = boxPackCharge.data();
    const int *const pKind = boxPackKind.data();
    const int *const pMol = boxPackMol.data();
    const double rCutSqBox = boxAxes.rCutSq[box];
    const double rCutLowSq = forcefield.rCutLowSq;
#ifdef _OPENMP
#pragma omp parallel for default(none)                                         \
    shared(overlap, trialPos, boxAxes, en, ff, real)                           \
    firstprivate(kindICharge, kindI, box, molIndex, num::qqFact, trials,       \
                     hasFraction, fracMol, fracVDW, fracCoul, nPacked, pX, pY, \
                     pZ, pQ, pKind, pMol, rCutSqBox, rCutLowSq)
#endif
    for (uint t = 0; t < trials; ++t) {
      double tempReal = 0.0;
      double tempLJ = 0.0;
      bool over = false;
      const double xi = trialPos.x[t], yi = trialPos.y[t], zi = trialPos.z[t];

      // Scratch, per thread, ~6 KiB and L1-resident.
      double distSq[GOMC_DISTSQ_CHUNK];
      double sDistSq[GOMC_DISTSQ_CHUNK + 8];
      double sQ[GOMC_DISTSQ_CHUNK + 8];
      int sKind[GOMC_DISTSQ_CHUNK + 8];
      int sMol[GOMC_DISTSQ_CHUNK + 8];

      for (int base = 0; base < nPacked; base += GOMC_DISTSQ_CHUNK) {
        const int m = std::min(GOMC_DISTSQ_CHUNK, nPacked - base);

        // ---- pass 1: contiguous, branch-free, vectorisable ----
        boxAxes.BoxType::DistSqRange(distSq, xi, yi, zi, pX + base, pY + base,
                                     pZ + base, m, box);

        // ---- pass 2: compact the survivors ----
        //
        // As in BoxInterTemplate. Simpler here: the moving molecule was taken
        // out of the cell list before this call, so there is no same-molecule
        // test to fold into the mask -- only the cutoff. Survival is ~20%,
        // which is the regime where this pays; see the note on the cell path
        // about why it does not at ~11%.
        int ns = 0;
        int k = 0;
#ifdef GOMC_HAVE_COMPRESS
        {
          const __m512d vRCut = _mm512_set1_pd(rCutSqBox);
          for (; k + 8 <= m; k += 8) {
            const int j = base + k;
            const __m512d d = _mm512_loadu_pd(distSq + k);
            const __mmask8 keep = _mm512_cmp_pd_mask(d, vRCut, _CMP_LT_OQ);
            _mm512_mask_compressstoreu_pd(sDistSq + ns, keep, d);
            _mm512_mask_compressstoreu_pd(sQ + ns, keep,
                                          _mm512_loadu_pd(pQ + j));
            _mm256_mask_compressstoreu_epi32(
                sKind + ns, keep,
                _mm256_loadu_si256((const __m256i *)(pKind + j)));
            _mm256_mask_compressstoreu_epi32(
                sMol + ns, keep,
                _mm256_loadu_si256((const __m256i *)(pMol + j)));
            ns += _mm_popcnt_u32((unsigned)keep);
          }
        }
#endif
        for (; k < m; k++) {
          if (!(rCutSqBox > distSq[k]))
            continue;
          const int j = base + k;
          sDistSq[ns] = distSq[k];
          sQ[ns] = pQ[j];
          sKind[ns] = pKind[j];
          sMol[ns] = pMol[j];
          ++ns;
        }

        // ---- pass 3: kernels over a dense run, no cutoff test ----
        for (int sIdx = 0; sIdx < ns; sIdx++) {
          const double dsq = sDistSq[sIdx];

          double lambdaVDW = 1.0;
          double lambdaCoulomb = 1.0;
          if (hasFraction) {
            int nMol = sMol[sIdx];
            if (molIndex == fracMol && nMol == fracMol) {
              lambdaVDW = fracVDW * fracVDW;
              lambdaCoulomb = fracCoul * fracCoul;
            } else if (molIndex == fracMol || nMol == fracMol) {
              lambdaVDW = fracVDW;
              lambdaCoulomb = fracCoul;
            }
          }

          if (dsq < rCutLowSq) {
            over = true;
          }
          tempLJ += ff.FFType::CalcEn(dsq, kindI, sKind[sIdx], lambdaVDW);
          if constexpr (HasCharge) {
            double qi_qj_fact = sQ[sIdx] * kindICharge * num::qqFact;

            if (qi_qj_fact != 0.0) {
              tempReal += ff.FFType::CalcCoulomb(dsq, kindI, sKind[sIdx],
                                                 qi_qj_fact, lambdaCoulomb,
                                                 box);
            }
          }
        }
      }
      overlap[t] |= over;
      en[t] += tempLJ;
      real[t] += tempReal;
    }
    GOMC_EVENT_STOP(1, GomcProfileEvent::EN_CBMC_INTER);
    return;
  }

// use OpenMP to distribute the workload over CBMC trials
#ifdef _OPENMP
#pragma omp parallel for default(none)                                         \
    shared(overlap, trialPos, boxAxes, en, ff, real)                           \
    firstprivate(kindICharge, kindI, box, molIndex, num::qqFact, trials,       \
                     hasFraction, fracMol, fracVDW, fracCoul)
#endif
  for (uint t = 0; t < trials; ++t) {
    // Each thread gets it's own copy of nIndex, tempReal and tempLJ
    std::vector<uint> nIndex;
    double tempReal = 0.0;
    double tempLJ = 0.0;
    nIndex.clear();
    tempReal = 0.0;
    tempLJ = 0.0;
    CellList::Neighbors n = cellList.EnumerateLocal(trialPos[t], box);
    while (!n.Done()) {
      nIndex.push_back(*n);
      n.Next();
    }

    for (int i = 0; i < (int)nIndex.size(); i++) {
      double distSq = 0.0;
      if (boxAxes.InRcut(distSq, trialPos, t, currentCoords, nIndex[i], box)) {

        double lambdaVDW = 1.0;
        double lambdaCoulomb = 1.0;
        if (hasFraction) {
          int nMol = particleMol[nIndex[i]];
          if (molIndex == fracMol && nMol == fracMol) {
            lambdaVDW = fracVDW * fracVDW;
            lambdaCoulomb = fracCoul * fracCoul;
          } else if (molIndex == fracMol || nMol == fracMol) {
            lambdaVDW = fracVDW;
            lambdaCoulomb = fracCoul;
          }
        }

        if (distSq < forcefield.rCutLowSq) {
          overlap[t] |= true;
        }
        tempLJ += ff.FFType::CalcEn(
            distSq, kindI, particleKind[nIndex[i]], lambdaVDW);
        if (electrostatic) {
          double qi_qj_fact =
              particleCharge[nIndex[i]] * kindICharge * num::qqFact;

          if (qi_qj_fact != 0.0) {
            tempReal += ff.FFType::CalcCoulomb(
                distSq, kindI, particleKind[nIndex[i]], qi_qj_fact,
                lambdaCoulomb, box);
          }
        }
      }
    }
    en[t] += tempLJ;
    real[t] += tempReal;
  }
  GOMC_EVENT_STOP(1, GomcProfileEvent::EN_CBMC_INTER);
}

// Calculate the inter energy for Box. Fractional molecule are not allowed in
// this function. Need to implement the GPU function
SystemPotential CalculateEnergy::BoxInter(SystemPotential potential,
                                          XYZArray const &coords,
                                          BoxDimensions const &boxAxes,
                                          const uint box) {
  // Handles reservoir box case, returning zeroed structure if
  // interactions are off.
  if (box >= BOXES_WITH_U_NB)
    return potential;

  GOMC_EVENT_START(1, GomcProfileEvent::EN_BOX_INTER);
  double tempREn = 0.0, tempLJEn = 0.0;

  std::vector<int> cellVector, cellStartIndex, mapParticleToCell;
  std::vector<std::vector<int>> neighborList;

  // Whole-box path. When the stencil selects every cell (StencilCoversBox),
  // the 27 per-cell ranges are the whole box between them, so none of the CSR
  // machinery earns its keep: GetCellListNeighbor walks the linked list and
  // sorts all 27 cells, GetNeighborList deep-copies a vector<vector<int>>, and
  // the pair walk then does 27 upper_bound searches per atom to recover a
  // bound it could have had for free. Pack once and walk j > i instead. That
  // also takes the contiguous run handed to DistSqRange from one cell (~146
  // atoms here) to the whole box.
  //
  // nPacked < 0 means "not taken"; the CUDA kernels always want the CSR form.
  int nPacked = -1;
#ifndef GOMC_CUDA
  if (StencilCoversBox(box)) {
    nPacked = BuildBoxPacked(coords, box);
  }
  if (nPacked < 0)
#endif
  {
    cellList.GetCellListNeighbor(box, currentCoords.Count(), cellVector,
                                 cellStartIndex, mapParticleToCell);
    neighborList = cellList.GetNeighborList(box);
#ifndef GOMC_CUDA
    // Permute the per-atom data into cell order for the pair walk below.
    BuildCellOrdered(coords, cellVector, mapParticleToCell);
#endif
  }

#ifdef GOMC_CUDA
  // update unitcell in GPU
  UpdateCellBasisCUDA(forcefield.particles->getCUDAVars(), box,
                      boxAxes.cellBasis[box].x, boxAxes.cellBasis[box].y,
                      boxAxes.cellBasis[box].z);

  if (!boxAxes.orthogonal[box]) {
    // In this case, boxAxes is really an object of type BoxDimensionsNonOrth,
    // so cast and copy the additional data to the GPU
    const BoxDimensionsNonOrth *NonOrthAxes =
        static_cast<const BoxDimensionsNonOrth *>(&boxAxes);
    UpdateInvCellBasisCUDA(forcefield.particles->getCUDAVars(), box,
                           NonOrthAxes->cellBasis_Inv[box].x,
                           NonOrthAxes->cellBasis_Inv[box].y,
                           NonOrthAxes->cellBasis_Inv[box].z);
  }

  CallBoxInterGPU(forcefield.particles->getCUDAVars(), cellVector,
                  cellStartIndex, neighborList, coords, boxAxes, electrostatic,
                  particleCharge, particleKind, particleMol, tempREn, tempLJEn,
                  forcefield.sc_coul, forcefield.sc_sigma_6,
                  forcefield.sc_alpha, forcefield.sc_power, box);
#else
  // Resolve the concrete forcefield and box types once, then run a kernel
  // with no virtual dispatch in the pair loop.
  // Resolve the fractional-molecule case here too, so the pair loop is
  // compiled without any lambda handling in the overwhelmingly common case
  // where there is none. See BoxInterTemplate's HasLambda parameter.
  const bool hasFraction = lambdaRef.HasFraction(box);
  // `electrostatic` is fixed for the whole run, but the annotate showed it
  // being re-loaded from `this` and branched on for every surviving pair --
  // 8.56% of this symbol. Resolving it here costs instantiations and buys a
  // pair loop with no charge test in it at all.
  DispatchForcefield(forcefield, [&](const auto &ffRef) {
    using FFT = std::decay_t<decltype(ffRef)>;
    if (boxAxes.orthogonal[box]) {
      if (hasFraction) {
        if (electrostatic)
          BoxInterTemplate<true, true, BoxDimensions, FFT>(
              ffRef, coords, boxAxes, box, tempREn, tempLJEn, cellVector,
              cellStartIndex, mapParticleToCell, neighborList, nPacked);
        else
          BoxInterTemplate<true, false, BoxDimensions, FFT>(
              ffRef, coords, boxAxes, box, tempREn, tempLJEn, cellVector,
              cellStartIndex, mapParticleToCell, neighborList, nPacked);
      } else {
        if (electrostatic)
          BoxInterTemplate<false, true, BoxDimensions, FFT>(
              ffRef, coords, boxAxes, box, tempREn, tempLJEn, cellVector,
              cellStartIndex, mapParticleToCell, neighborList, nPacked);
        else
          BoxInterTemplate<false, false, BoxDimensions, FFT>(
              ffRef, coords, boxAxes, box, tempREn, tempLJEn, cellVector,
              cellStartIndex, mapParticleToCell, neighborList, nPacked);
      }
    } else {
      const BoxDimensionsNonOrth &nonOrth =
          static_cast<const BoxDimensionsNonOrth &>(boxAxes);
      if (hasFraction) {
        if (electrostatic)
          BoxInterTemplate<true, true, BoxDimensionsNonOrth, FFT>(
              ffRef, coords, nonOrth, box, tempREn, tempLJEn, cellVector,
              cellStartIndex, mapParticleToCell, neighborList, nPacked);
        else
          BoxInterTemplate<true, false, BoxDimensionsNonOrth, FFT>(
              ffRef, coords, nonOrth, box, tempREn, tempLJEn, cellVector,
              cellStartIndex, mapParticleToCell, neighborList, nPacked);
      } else {
        if (electrostatic)
          BoxInterTemplate<false, true, BoxDimensionsNonOrth, FFT>(
              ffRef, coords, nonOrth, box, tempREn, tempLJEn, cellVector,
              cellStartIndex, mapParticleToCell, neighborList, nPacked);
        else
          BoxInterTemplate<false, false, BoxDimensionsNonOrth, FFT>(
              ffRef, coords, nonOrth, box, tempREn, tempLJEn, cellVector,
              cellStartIndex, mapParticleToCell, neighborList, nPacked);
      }
    }
  });
#endif

  // setting energy and virial of LJ interaction
  potential.boxEnergy[box].inter = tempLJEn;
  // setting energy and virial of coulomb interaction
  potential.boxEnergy[box].real = tempREn;

  GOMC_EVENT_STOP(1, GomcProfileEvent::EN_BOX_INTER);
  // set correction energy and virial
  if (forcefield.useLRC) {
    EnergyCorrection(potential, boxAxes, box);
  }

  potential.Total();
  return potential;
}

SystemPotential
CalculateEnergy::BoxForce(SystemPotential potential, XYZArray const &coords,
                          XYZArray &atomForce, XYZArray &molForce,
                          BoxDimensions const &boxAxes, const uint box) {
  // Handles reservoir box case, returning zeroed structure if
  // interactions are off.
  if (box >= BOXES_WITH_U_NB)
    return potential;

  GOMC_EVENT_START(1, GomcProfileEvent::EN_BOX_FORCE);

  double tempREn = 0.0, tempLJEn = 0.0;
  // make a pointer to atom force and mol force for OpenMP
  double *aForcex = atomForce.x;
  double *aForcey = atomForce.y;
  double *aForcez = atomForce.z;
  double *mForcex = molForce.x;
  double *mForcey = molForce.y;
  double *mForcez = molForce.z;
  int atomCount = atomForce.Count();
  int molCount = molForce.Count();

  // Reset Force Arrays
  ResetForce(atomForce, molForce, box);

  std::vector<int> cellVector, cellStartIndex, mapParticleToCell;
  std::vector<std::vector<int>> neighborList;
  cellList.GetCellListNeighbor(box, coords.Count(), cellVector, cellStartIndex,
                               mapParticleToCell);
  neighborList = cellList.GetNeighborList(box);

#ifdef GOMC_CUDA
  // update unitcell in GPU
  UpdateCellBasisCUDA(forcefield.particles->getCUDAVars(), box,
                      boxAxes.cellBasis[box].x, boxAxes.cellBasis[box].y,
                      boxAxes.cellBasis[box].z);

  if (!boxAxes.orthogonal[box]) {
    // In this case, boxAxes is really an object of type BoxDimensionsNonOrth,
    // so cast and copy the additional data to the GPU
    const BoxDimensionsNonOrth *NonOrthAxes =
        static_cast<const BoxDimensionsNonOrth *>(&boxAxes);
    UpdateInvCellBasisCUDA(forcefield.particles->getCUDAVars(), box,
                           NonOrthAxes->cellBasis_Inv[box].x,
                           NonOrthAxes->cellBasis_Inv[box].y,
                           NonOrthAxes->cellBasis_Inv[box].z);
  }

  CallBoxForceGPU(forcefield.particles->getCUDAVars(), cellVector,
                  cellStartIndex, neighborList, mapParticleToCell, coords,
                  boxAxes, electrostatic, particleCharge, particleKind,
                  particleMol, tempREn, tempLJEn, aForcex, aForcey, aForcez,
                  mForcex, mForcey, mForcez, atomCount, molCount,
                  forcefield.sc_coul, forcefield.sc_sigma_6,
                  forcefield.sc_alpha, forcefield.sc_power, box);

#else
  DispatchForcefield(forcefield, [&](const auto &ffRef) {
    using FFT = std::decay_t<decltype(ffRef)>;
    if (boxAxes.orthogonal[box]) {
      BoxForceTemplate<BoxDimensions, FFT>(
          ffRef, coords, atomForce, molForce, boxAxes, box, tempREn, tempLJEn,
          cellVector, cellStartIndex, mapParticleToCell, neighborList);
    } else {
      BoxForceTemplate<BoxDimensionsNonOrth, FFT>(
          ffRef, coords, atomForce, molForce,
          static_cast<const BoxDimensionsNonOrth &>(boxAxes), box, tempREn,
          tempLJEn, cellVector, cellStartIndex, mapParticleToCell,
          neighborList);
    }
  });
#endif

  // setting energy and virial of LJ interaction
  potential.boxEnergy[box].inter = tempLJEn;
  // setting energy and virial of coulomb interaction
  potential.boxEnergy[box].real = tempREn;

  GOMC_EVENT_STOP(1, GomcProfileEvent::EN_BOX_FORCE);
  return potential;
}

// NOTE: The calculation of W12, W13, and W23 is expensive and would not be
// required for pressure and surface tension calculation. So, they have been
// commented out. If you need to calculate them, uncomment them.
Virial CalculateEnergy::VirialCalc(const uint box) {
  // store virial and energy of reference and modify the virial
  Virial tempVir;
  // no need to calculate the virial for reservoir
  if (box >= BOXES_WITH_U_NB)
    return tempVir;

  GOMC_EVENT_START(1, GomcProfileEvent::EN_BOX_VIRIAL);

  // tensors for VDW and real part of electrostatic
  double vT11 = 0.0, vT12 = 0.0, vT13 = 0.0;
  double vT22 = 0.0, vT23 = 0.0, vT33 = 0.0;
  double rT11 = 0.0, rT12 = 0.0, rT13 = 0.0;
  double rT22 = 0.0, rT23 = 0.0, rT33 = 0.0;

  std::vector<int> cellVector, cellStartIndex, mapParticleToCell;
  std::vector<std::vector<int>> neighborList;

  // Whole-box path, as in BoxInter(); see StencilCoversBox().
  int nPacked = -1;
#ifndef GOMC_CUDA
  if (StencilCoversBox(box)) {
    nPacked = BuildBoxPacked(currentCoords, box);
  }
  if (nPacked < 0)
#endif
  {
    cellList.GetCellListNeighbor(box, currentCoords.Count(), cellVector,
                                 cellStartIndex, mapParticleToCell);
    neighborList = cellList.GetNeighborList(box);
  }

#ifdef GOMC_CUDA
  // update unitcell in GPU
  UpdateCellBasisCUDA(
      forcefield.particles->getCUDAVars(), box, currentAxes.cellBasis[box].x,
      currentAxes.cellBasis[box].y, currentAxes.cellBasis[box].z);

  if (!currentAxes.orthogonal[box]) {
    // In this case, currentAxes is really an object of type
    // BoxDimensionsNonOrth,
    // so cast and copy the additional data to the GPU
    const BoxDimensionsNonOrth *NonOrthAxes =
        static_cast<const BoxDimensionsNonOrth *>(&currentAxes);
    UpdateInvCellBasisCUDA(forcefield.particles->getCUDAVars(), box,
                           NonOrthAxes->cellBasis_Inv[box].x,
                           NonOrthAxes->cellBasis_Inv[box].y,
                           NonOrthAxes->cellBasis_Inv[box].z);
  }

  CallBoxInterForceGPU(forcefield.particles->getCUDAVars(), cellVector,
                       cellStartIndex, neighborList, mapParticleToCell,
                       currentCoords, currentCOM, currentAxes, electrostatic,
                       particleCharge, particleKind, particleMol, rT11, rT12,
                       rT13, rT22, rT23, rT33, vT11, vT12, vT13, vT22, vT23,
                       vT33, forcefield.sc_coul, forcefield.sc_sigma_6,
                       forcefield.sc_alpha, forcefield.sc_power, box);
#else
  DispatchForcefield(forcefield, [&](const auto &ffRef) {
    using FFT = std::decay_t<decltype(ffRef)>;
    if (currentAxes.orthogonal[box]) {
      VirialCalcTemplate<BoxDimensions, FFT>(
          ffRef, currentAxes, box, vT11, vT12, vT13, vT22, vT23, vT33, rT11,
          rT12, rT13, rT22, rT23, rT33, cellVector, cellStartIndex,
          mapParticleToCell, neighborList, nPacked);
    } else {
      VirialCalcTemplate<BoxDimensionsNonOrth, FFT>(
          ffRef, static_cast<const BoxDimensionsNonOrth &>(currentAxes), box,
          vT11, vT12, vT13, vT22, vT23, vT33, rT11, rT12, rT13, rT22, rT23,
          rT33, cellVector, cellStartIndex, mapParticleToCell, neighborList,
          nPacked);
    }
  });
#endif

  // set the all tensor values
  tempVir.interTens[0][0] = vT11;
  tempVir.interTens[0][1] = vT12;
  tempVir.interTens[0][2] = vT13;

  tempVir.interTens[1][0] = vT12;
  tempVir.interTens[1][1] = vT22;
  tempVir.interTens[1][2] = vT23;

  tempVir.interTens[2][0] = vT13;
  tempVir.interTens[2][1] = vT23;
  tempVir.interTens[2][2] = vT33;

  if (electrostatic) {
    // real part of electrostatic
    tempVir.realTens[0][0] = rT11 * num::qqFact;
    tempVir.realTens[0][1] = rT12 * num::qqFact;
    tempVir.realTens[0][2] = rT13 * num::qqFact;

    tempVir.realTens[1][0] = rT12 * num::qqFact;
    tempVir.realTens[1][1] = rT22 * num::qqFact;
    tempVir.realTens[1][2] = rT23 * num::qqFact;

    tempVir.realTens[2][0] = rT13 * num::qqFact;
    tempVir.realTens[2][1] = rT23 * num::qqFact;
    tempVir.realTens[2][2] = rT33 * num::qqFact;
  }

  // setting virial of LJ
  tempVir.inter = vT11 + vT22 + vT33;
  // setting virial of coulomb
  tempVir.real = (rT11 + rT22 + rT33) * num::qqFact;

  GOMC_EVENT_STOP(1, GomcProfileEvent::EN_BOX_VIRIAL);

  if (forcefield.useLRC || forcefield.useIPC) {
    VirialCorrection(tempVir, currentAxes, box);
  }

  // calculate reciprocal term of force
  tempVir = calcEwald->VirialReciprocal(tempVir, box);

  tempVir.Total();
  return tempVir;
}

bool CalculateEnergy::MoleculeInter(Intermolecular &inter_LJ,
                                    Intermolecular &inter_coulomb,
                                    XYZArray const &molCoords,
                                    const uint molIndex, const uint box) const {
  bool overlap = false;
  DispatchForcefield(forcefield, [&](const auto &ffRef) {
    using FFT = std::decay_t<decltype(ffRef)>;
    if (currentAxes.orthogonal[box]) {
      overlap = MoleculeInterTemplate<BoxDimensions, FFT>(
          ffRef, inter_LJ, inter_coulomb, molCoords, molIndex, box,
          currentAxes);
    } else {
      overlap = MoleculeInterTemplate<BoxDimensionsNonOrth, FFT>(
          ffRef, inter_LJ, inter_coulomb, molCoords, molIndex, box,
          static_cast<const BoxDimensionsNonOrth &>(currentAxes));
    }
  });
  return overlap;
}

void CalculateEnergy::ParticleNonbonded(double *inter,
                                        cbmc::TrialMol const &trialMol,
                                        XYZArray const &trialPos,
                                        const uint partIndex, const uint box,
                                        const uint trials) const {
  DispatchForcefield(forcefield, [&](const auto &ffRef) {
    using FFT = std::decay_t<decltype(ffRef)>;
    if (currentAxes.orthogonal[box]) {
      ParticleNonbondedTemplate<BoxDimensions, FFT>(
          ffRef, inter, trialMol, trialPos, partIndex, box, trials,
          currentAxes);
    } else {
      ParticleNonbondedTemplate<BoxDimensionsNonOrth, FFT>(
          ffRef, inter, trialMol, trialPos, partIndex, box, trials,
          static_cast<const BoxDimensionsNonOrth &>(currentAxes));
    }
  });
}

void CalculateEnergy::ParticleInter(double *en, double *real,
                                    XYZArray const &trialPos, bool *overlap,
                                    const uint partIndex, const uint molIndex,
                                    const uint box, const uint trials) const {
  // `electrostatic` as a template parameter, as in BoxInterTemplate. Hoisting
  // it into a local was tried first and was not enough: inside the OpenMP
  // outlined region the firstprivate copy gets spilled and re-loaded anyway,
  // and the disassembly still showed four byte-loads from memory in the hot
  // clone. The template removes the test rather than the load.
  DispatchForcefield(forcefield, [&](const auto &ffRef) {
    using FFT = std::decay_t<decltype(ffRef)>;
    if (currentAxes.orthogonal[box]) {
      if (electrostatic)
        ParticleInterTemplate<true, BoxDimensions, FFT>(
            ffRef, en, real, trialPos, overlap, partIndex, molIndex, box,
            trials, currentAxes);
      else
        ParticleInterTemplate<false, BoxDimensions, FFT>(
            ffRef, en, real, trialPos, overlap, partIndex, molIndex, box,
            trials, currentAxes);
    } else {
      const BoxDimensionsNonOrth &nonOrth =
          static_cast<const BoxDimensionsNonOrth &>(currentAxes);
      if (electrostatic)
        ParticleInterTemplate<true, BoxDimensionsNonOrth, FFT>(
            ffRef, en, real, trialPos, overlap, partIndex, molIndex, box,
            trials, nonOrth);
      else
        ParticleInterTemplate<false, BoxDimensionsNonOrth, FFT>(
            ffRef, en, real, trialPos, overlap, partIndex, molIndex, box,
            trials, nonOrth);
    }
  });
}

// Calculates the change in the TC from adding numChange atoms of a kind
Intermolecular CalculateEnergy::MoleculeTailChange(const uint box,
                                                   const uint kind,
                                                   const bool add) const {
  Intermolecular delta;

  if (box < BOXES_WITH_U_NB) {
    double sign = (add ? 1.0 : -1.0);
    uint mkIdxII = kind * mols.GetKindsCount() + kind;
    for (uint j = 0; j < mols.GetKindsCount(); ++j) {
      uint mkIdxIJ = j * mols.GetKindsCount() + kind;
      double rhoDeltaIJ_2 = sign * 2.0 *
                            (double)(molLookup.NumKindInBox(j, box)) *
                            currentAxes.volInv[box];
      delta.energy += mols.pairEnCorrections[mkIdxIJ] * rhoDeltaIJ_2;
    }

    // We already calculated part of the change for this type in the loop
    delta.energy += mols.pairEnCorrections[mkIdxII] * currentAxes.volInv[box];
  }
  return delta;
}

// Calculates the change in the Virial TC from adding numChange atoms of a kind
Intermolecular CalculateEnergy::MoleculeTailVirChange(const uint box,
                                                      const uint kind,
                                                      const bool add) const {
  Intermolecular delta;

  if (box < BOXES_WITH_U_NB) {
    double sign = (add ? 1.0 : -1.0);
    uint mkIdxII = kind * mols.GetKindsCount() + kind;
    for (uint j = 0; j < mols.GetKindsCount(); ++j) {
      uint mkIdxIJ = j * mols.GetKindsCount() + kind;
      double rhoDeltaIJ_2 = sign * 2.0 *
                            (double)(molLookup.NumKindInBox(j, box)) *
                            currentAxes.volInv[box];
      delta.virial += mols.pairVirCorrections[mkIdxIJ] * rhoDeltaIJ_2;
    }

    // We already calculated part of the change for this type in the loop
    delta.virial += mols.pairVirCorrections[mkIdxII] * currentAxes.volInv[box];
  }
  return delta;
}

// Calculates intramolecular energy of a full molecule
void CalculateEnergy::MoleculeIntra(const uint molIndex, const uint box,
                                    double *bondEn) const {
  GOMC_EVENT_START(1, GomcProfileEvent::EN_MOL_INTRA);
  bondEn[0] = 0.0, bondEn[1] = 0.0;

  MoleculeKind &molKind = mols.kinds[mols.kIndex[molIndex]];
  // *2 because we'll be storing inverse bond vectors
  XYZArray bondVec(molKind.bondList.count * 2);

  BondVectors(bondVec, molKind, molIndex, box);
  MolBond(bondEn[0], molKind, bondVec, molIndex, box);
  MolAngle(bondEn[0], molKind, bondVec, box);
  MolDihedral(bondEn[0], molKind, bondVec, box);
  MolNonbond(bondEn[1], molKind, molIndex, box);
  MolNonbond_1_4(bondEn[1], molKind, molIndex, box);
  MolNonbond_1_3(bondEn[1], molKind, molIndex, box);
  GOMC_EVENT_STOP(1, GomcProfileEvent::EN_MOL_INTRA);
}

// used in molecule exchange for calculating bonded and intraNonbonded energy
Energy CalculateEnergy::MoleculeIntra(cbmc::TrialMol const &mol) const {
  GOMC_EVENT_START(1, GomcProfileEvent::EN_MOL_INTRA);
  double bondEn = 0.0, intraNonbondEn = 0.0;
  // *2 because we'll be storing inverse bond vectors
  const MoleculeKind &molKind = mol.GetKind();
  uint count = molKind.bondList.count;
  XYZArray bondVec(count * 2);
  std::vector<bool> bondExist(count * 2, false);

  BondVectors(bondVec, mol, bondExist, molKind);
  MolBond(bondEn, mol, bondVec, bondExist, molKind);
  MolAngle(bondEn, mol, bondVec, bondExist, molKind);
  MolDihedral(bondEn, mol, bondVec, bondExist, molKind);
  MolNonbond(intraNonbondEn, mol, molKind);
  MolNonbond_1_4(intraNonbondEn, mol, molKind);
  MolNonbond_1_3(intraNonbondEn, mol, molKind);
  GOMC_EVENT_STOP(1, GomcProfileEvent::EN_MOL_INTRA);
  return Energy(bondEn, intraNonbondEn, 0.0, 0.0, 0.0, 0.0, 0.0);
}

void CalculateEnergy::BondVectors(XYZArray &vecs, MoleculeKind const &molKind,
                                  const uint molIndex, const uint box) const {
  for (uint i = 0; i < molKind.bondList.count; ++i) {
    uint p1 = mols.start[molIndex] + molKind.bondList.part1[i];
    uint p2 = mols.start[molIndex] + molKind.bondList.part2[i];
    XYZ dist = currentCoords.Difference(p2, p1);
    dist = currentAxes.MinImage(dist, box);

    // store inverse vectors at i+count
    vecs.Set(i, dist);
    vecs.Set(i + molKind.bondList.count, -dist.x, -dist.y, -dist.z);
  }
}

void CalculateEnergy::BondVectors(XYZArray &vecs, cbmc::TrialMol const &mol,
                                  std::vector<bool> &bondExist,
                                  MoleculeKind const &molKind) const {
  uint box = mol.GetBox();
  uint count = molKind.bondList.count;
  for (uint i = 0; i < count; ++i) {
    uint p1 = molKind.bondList.part1[i];
    uint p2 = molKind.bondList.part2[i];
    if (mol.AtomExists(p1) && mol.AtomExists(p2)) {
      bondExist[i] = true;
      bondExist[i + count] = true;
      XYZ dist = mol.GetCoords().Difference(p2, p1);
      dist = currentAxes.MinImage(dist, box);
      // store inverse vectors at i+count
      vecs.Set(i, dist);
      vecs.Set(i + count, -dist.x, -dist.y, -dist.z);
    }
  }
}

void CalculateEnergy::MolBond(double &energy, MoleculeKind const &molKind,
                              XYZArray const &vecs, const uint molIndex,
                              const uint box) const {
  if (box >= BOXES_WITH_U_B)
    return;

  for (uint b = 0; b < molKind.bondList.count; ++b) {
    double molLength = vecs.Get(b).Length();
    energy += forcefield.bonds.Calc(molKind.bondList.kinds[b], molLength);
    /*if(std::abs(molLength - eqLength) > 0.02) {
      uint p1 = molKind.bondList.part1[b];
      uint p2 = molKind.bondList.part2[b];
      double eqLength = forcefield.bonds.Length(molKind.bondList.kinds[b]);
      printf("Warning: Box%d, %6d %4s,", box, molIndex, molKind.name.c_str());
      printf("%3s-%-3s bond: Par-file ", molKind.atomNames[p1].c_str(),
          molKind.atomNames[p2].c_str());
      printf("%2.3f A, PDB file %2.3f A!\n", eqLength, molLength);
    }*/
  }
}

void CalculateEnergy::MolBond(double &energy, cbmc::TrialMol const &mol,
                              XYZArray const &vecs,
                              std::vector<bool> const &bondExist,
                              MoleculeKind const &molKind) const {
  if (mol.GetBox() >= BOXES_WITH_U_B)
    return;

  uint count = molKind.bondList.count;
  for (uint b = 0; b < count; ++b) {
    if (bondExist[b]) {
      energy += forcefield.bonds.Calc(molKind.bondList.kinds[b],
                                      vecs.Get(b).Length());
    }
  }
}

void CalculateEnergy::MolAngle(double &energy, MoleculeKind const &molKind,
                               XYZArray const &vecs, const uint box) const {
  if (box >= BOXES_WITH_U_B)
    return;
  for (uint a = 0; a < molKind.angles.Count(); ++a) {
    // Note: need to reverse the second bond to get angle properly.
    double theta = Theta(vecs.Get(molKind.angles.GetBond(a, 0)),
                         -vecs.Get(molKind.angles.GetBond(a, 1)));
    energy += forcefield.angles->Calc(molKind.angles.GetKind(a), theta);
  }
}

void CalculateEnergy::MolAngle(double &energy, cbmc::TrialMol const &mol,
                               XYZArray const &vecs,
                               std::vector<bool> const &bondExist,
                               MoleculeKind const &molKind) const {
  if (mol.GetBox() >= BOXES_WITH_U_B)
    return;

  uint count = molKind.angles.Count();
  for (uint a = 0; a < count; ++a) {
    if (bondExist[molKind.angles.GetBond(a, 0)] &&
        bondExist[molKind.angles.GetBond(a, 1)]) {
      // Note: need to reverse the second bond to get angle properly.
      double theta = Theta(vecs.Get(molKind.angles.GetBond(a, 0)),
                           -vecs.Get(molKind.angles.GetBond(a, 1)));
      energy += forcefield.angles->Calc(molKind.angles.GetKind(a), theta);
    }
  }
}

void CalculateEnergy::MolDihedral(double &energy, MoleculeKind const &molKind,
                                  XYZArray const &vecs, const uint box) const {
  if (box >= BOXES_WITH_U_B)
    return;
  for (uint d = 0; d < molKind.dihedrals.Count(); ++d) {
    double phi = Phi(vecs.Get(molKind.dihedrals.GetBond(d, 0)),
                     vecs.Get(molKind.dihedrals.GetBond(d, 1)),
                     vecs.Get(molKind.dihedrals.GetBond(d, 2)));
    energy += forcefield.dihedrals.Calc(molKind.dihedrals.GetKind(d), phi);
  }
}

void CalculateEnergy::MolDihedral(double &energy, cbmc::TrialMol const &mol,
                                  XYZArray const &vecs,
                                  std::vector<bool> const &bondExist,
                                  MoleculeKind const &molKind) const {
  if (mol.GetBox() >= BOXES_WITH_U_B)
    return;

  uint count = molKind.dihedrals.Count();
  for (uint d = 0; d < count; ++d) {
    if (bondExist[molKind.dihedrals.GetBond(d, 0)] &&
        bondExist[molKind.dihedrals.GetBond(d, 1)] &&
        bondExist[molKind.dihedrals.GetBond(d, 2)]) {
      double phi = Phi(vecs.Get(molKind.dihedrals.GetBond(d, 0)),
                       vecs.Get(molKind.dihedrals.GetBond(d, 1)),
                       vecs.Get(molKind.dihedrals.GetBond(d, 2)));
      energy += forcefield.dihedrals.Calc(molKind.dihedrals.GetKind(d), phi);
    }
  }
}

// Calculate 1-N nonbonded intra energy
void CalculateEnergy::MolNonbond(double &energy, MoleculeKind const &molKind,
                                 const uint molIndex, const uint box) const {
  if (box >= BOXES_WITH_U_B)
    return;

  double distSq;
  double qi_qj_fact;

  for (uint i = 0; i < molKind.nonBonded.count; ++i) {
    uint p1 = mols.start[molIndex] + molKind.nonBonded.part1[i];
    uint p2 = mols.start[molIndex] + molKind.nonBonded.part2[i];
    if (currentAxes.InRcut(distSq, currentCoords, p1, p2, box)) {
      energy += forcefield.particles->CalcEn(
          distSq, molKind.AtomKind(molKind.nonBonded.part1[i]),
          molKind.AtomKind(molKind.nonBonded.part2[i]), 1.0);
      if (electrostatic) {
        qi_qj_fact = num::qqFact *
                     molKind.AtomCharge(molKind.nonBonded.part1[i]) *
                     molKind.AtomCharge(molKind.nonBonded.part2[i]);

        if (qi_qj_fact != 0.0) {
          forcefield.particles->CalcCoulombAdd_1_4(energy, distSq, qi_qj_fact,
                                                   true);
        }
      }
    }
  }
}

// Calculate 1-N nonbonded intra energy using pos
void CalculateEnergy::MolNonbond(double &energy, cbmc::TrialMol const &mol,
                                 MoleculeKind const &molKind) const {
  if (mol.GetBox() >= BOXES_WITH_U_B)
    return;

  double distSq;
  double qi_qj_fact;
  uint count = molKind.nonBonded.count;

  for (uint i = 0; i < count; ++i) {
    uint p1 = molKind.nonBonded.part1[i];
    uint p2 = molKind.nonBonded.part2[i];
    if (mol.AtomExists(p1) && mol.AtomExists(p2)) {
      if (currentAxes.InRcut(distSq, mol.GetCoords(), p1, p2, mol.GetBox())) {
        energy += forcefield.particles->CalcEn(distSq, molKind.AtomKind(p1),
                                               molKind.AtomKind(p2), 1.0);
        if (electrostatic) {
          qi_qj_fact =
              num::qqFact * molKind.AtomCharge(1) * molKind.AtomCharge(p2);

          if (qi_qj_fact != 0.0) {
            forcefield.particles->CalcCoulombAdd_1_4(energy, distSq, qi_qj_fact,
                                                     true);
          }
        }
      }
    }
  }
}

// Calculate 1-4 nonbonded intra energy
void CalculateEnergy::MolNonbond_1_4(double &energy,
                                     MoleculeKind const &molKind,
                                     const uint molIndex,
                                     const uint box) const {
  if (box >= BOXES_WITH_U_B)
    return;

  double distSq;
  double qi_qj_fact;

  for (uint i = 0; i < molKind.nonBonded_1_4.count; ++i) {
    uint p1 = mols.start[molIndex] + molKind.nonBonded_1_4.part1[i];
    uint p2 = mols.start[molIndex] + molKind.nonBonded_1_4.part2[i];
    if (currentAxes.InRcut(distSq, currentCoords, p1, p2, box)) {
      forcefield.particles->CalcAdd_1_4(
          energy, distSq, molKind.AtomKind(molKind.nonBonded_1_4.part1[i]),
          molKind.AtomKind(molKind.nonBonded_1_4.part2[i]));
      if (electrostatic) {
        qi_qj_fact = num::qqFact *
                     molKind.AtomCharge(molKind.nonBonded_1_4.part1[i]) *
                     molKind.AtomCharge(molKind.nonBonded_1_4.part2[i]);

        if (qi_qj_fact != 0.0) {
          forcefield.particles->CalcCoulombAdd_1_4(energy, distSq, qi_qj_fact,
                                                   false);
        }
      }
    }
  }
}

// Calculate 1-4 nonbonded intra energy using pos
void CalculateEnergy::MolNonbond_1_4(double &energy, cbmc::TrialMol const &mol,
                                     MoleculeKind const &molKind) const {
  if (mol.GetBox() >= BOXES_WITH_U_B)
    return;

  double distSq;
  double qi_qj_fact;
  uint count = molKind.nonBonded_1_4.count;

  for (uint i = 0; i < count; ++i) {
    uint p1 = molKind.nonBonded_1_4.part1[i];
    uint p2 = molKind.nonBonded_1_4.part2[i];
    if (mol.AtomExists(p1) && mol.AtomExists(p2)) {
      if (currentAxes.InRcut(distSq, mol.GetCoords(), p1, p2, mol.GetBox())) {
        forcefield.particles->CalcAdd_1_4(energy, distSq, molKind.AtomKind(p1),
                                          molKind.AtomKind(p2));
        if (electrostatic) {
          qi_qj_fact =
              num::qqFact * molKind.AtomCharge(p1) * molKind.AtomCharge(p2);

          if (qi_qj_fact != 0.0) {
            forcefield.particles->CalcCoulombAdd_1_4(energy, distSq, qi_qj_fact,
                                                     false);
          }
        }
      }
    }
  }
}

// Calculate 1-3 nonbonded intra energy
void CalculateEnergy::MolNonbond_1_3(double &energy,
                                     MoleculeKind const &molKind,
                                     const uint molIndex,
                                     const uint box) const {
  if (box >= BOXES_WITH_U_B)
    return;

  double distSq;
  double qi_qj_fact;

  for (uint i = 0; i < molKind.nonBonded_1_3.count; ++i) {
    uint p1 = mols.start[molIndex] + molKind.nonBonded_1_3.part1[i];
    uint p2 = mols.start[molIndex] + molKind.nonBonded_1_3.part2[i];
    if (currentAxes.InRcut(distSq, currentCoords, p1, p2, box)) {
      forcefield.particles->CalcAdd_1_4(
          energy, distSq, molKind.AtomKind(molKind.nonBonded_1_3.part1[i]),
          molKind.AtomKind(molKind.nonBonded_1_3.part2[i]));
      if (electrostatic) {
        qi_qj_fact = num::qqFact *
                     molKind.AtomCharge(molKind.nonBonded_1_3.part1[i]) *
                     molKind.AtomCharge(molKind.nonBonded_1_3.part2[i]);

        if (qi_qj_fact != 0.0) {
          forcefield.particles->CalcCoulombAdd_1_4(energy, distSq, qi_qj_fact,
                                                   false);
        }
      }
    }
  }
}

// Calculate 1-3 nonbonded intra energy
void CalculateEnergy::MolNonbond_1_3(double &energy, cbmc::TrialMol const &mol,
                                     MoleculeKind const &molKind) const {
  if (mol.GetBox() >= BOXES_WITH_U_B)
    return;

  double distSq;
  double qi_qj_fact;
  uint count = molKind.nonBonded_1_3.count;

  for (uint i = 0; i < count; ++i) {
    uint p1 = molKind.nonBonded_1_3.part1[i];
    uint p2 = molKind.nonBonded_1_3.part2[i];
    if (mol.AtomExists(p1) && mol.AtomExists(p2)) {
      if (currentAxes.InRcut(distSq, mol.GetCoords(), p1, p2, mol.GetBox())) {
        forcefield.particles->CalcAdd_1_4(energy, distSq, molKind.AtomKind(p1),
                                          molKind.AtomKind(p2));
        if (electrostatic) {
          qi_qj_fact =
              num::qqFact * molKind.AtomCharge(p1) * molKind.AtomCharge(p2);

          if (qi_qj_fact != 0.0) {
            forcefield.particles->CalcCoulombAdd_1_4(energy, distSq, qi_qj_fact,
                                                     false);
          }
        }
      }
    }
  }
}

// Calculate 1-3 nonbonded intra energy
double CalculateEnergy::IntraEnergy_1_3(const double distSq, const uint atom1,
                                        const uint atom2,
                                        const uint molIndex) const {
  if (!forcefield.OneThree)
    return 0.0;

  double eng = 0.0;

  MoleculeKind const &thisKind = mols.GetKind(molIndex);
  uint kind1 = thisKind.AtomKind(atom1);
  uint kind2 = thisKind.AtomKind(atom2);

  if (electrostatic) {
    double qi_qj_fact =
        num::qqFact * thisKind.AtomCharge(atom1) * thisKind.AtomCharge(atom2);

    if (qi_qj_fact != 0.0) {
      forcefield.particles->CalcCoulombAdd_1_4(eng, distSq, qi_qj_fact, false);
    }
  }
  forcefield.particles->CalcAdd_1_4(eng, distSq, kind1, kind2);

  if (std::isnan(eng))
    eng = num::BIGNUM;

  return eng;
}

// Calculate 1-4 nonbonded intra energy
double CalculateEnergy::IntraEnergy_1_4(const double distSq, const uint atom1,
                                        const uint atom2,
                                        const uint molIndex) const {
  if (!forcefield.OneFour)
    return 0.0;

  double eng = 0.0;

  MoleculeKind const &thisKind = mols.GetKind(molIndex);
  uint kind1 = thisKind.AtomKind(atom1);
  uint kind2 = thisKind.AtomKind(atom2);

  if (electrostatic) {
    double qi_qj_fact =
        num::qqFact * thisKind.AtomCharge(atom1) * thisKind.AtomCharge(atom2);

    if (qi_qj_fact != 0.0) {
      forcefield.particles->CalcCoulombAdd_1_4(eng, distSq, qi_qj_fact, false);
    }
  }
  forcefield.particles->CalcAdd_1_4(eng, distSq, kind1, kind2);

  if (std::isnan(eng))
    eng = num::BIGNUM;

  return eng;
}

//! Calculates energy tail corrections for the box
void CalculateEnergy::EnergyCorrection(SystemPotential &pot,
                                       BoxDimensions const &boxAxes,
                                       const uint box) const {
  if (box >= BOXES_WITH_U_NB) {
    return;
  }

  double en = 0.0;
  for (uint i = 0; i < mols.GetKindsCount(); ++i) {
    uint numI = molLookup.NumKindInBox(i, box);
    for (uint j = 0; j < mols.GetKindsCount(); ++j) {
      uint numJ = molLookup.NumKindInBox(j, box);
      en += mols.pairEnCorrections[i * mols.GetKindsCount() + j] * numI * numJ *
            boxAxes.volInv[box];
    }
  }

  if (!forcefield.freeEnergy) {
    pot.boxEnergy[box].tailCorrection = en;
  }
#if ENSEMBLE == NVT || ENSEMBLE == NPT
  else {
    // Get the kind and lambda value
    uint fk = mols.GetMolKind(lambdaRef.GetMolIndex(box));
    double lambdaVDW = lambdaRef.GetLambdaVDW(lambdaRef.GetMolIndex(box), box);
    // remove the LRC for one molecule with lambda = 1
    en += MoleculeTailChange(box, fk, false).energy;

    // Add the LRC for fractional molecule
    for (uint i = 0; i < mols.GetKindsCount(); ++i) {
      uint molNum = molLookup.NumKindInBox(i, box);
      if (i == fk) {
        --molNum; // We have one less molecule (it is fractional molecule)
      }
      double rhoDeltaIJ_2 = 2.0 * (double)(molNum)*currentAxes.volInv[box];
      en += lambdaVDW * mols.pairEnCorrections[fk * mols.GetKindsCount() + i] *
            rhoDeltaIJ_2;
    }
    // We already calculated part of the change for this type in the loop
    en += lambdaVDW * mols.pairEnCorrections[fk * mols.GetKindsCount() + fk] *
          currentAxes.volInv[box];
    pot.boxEnergy[box].tailCorrection = en;
  }
#endif
}

//! Calculates energy corrections for the box
double CalculateEnergy::EnergyCorrection(const uint box,
                                         const uint *kCount) const {
  if (box >= BOXES_WITH_U_NB) {
    return 0.0;
  }

  double tailCorrection = 0.0;
  for (uint i = 0; i < mols.kindsCount; ++i) {
    for (uint j = 0; j < mols.kindsCount; ++j) {
      tailCorrection += mols.pairEnCorrections[i * mols.kindsCount + j] *
                        kCount[i] * kCount[j] * currentAxes.volInv[box];
    }
  }
  return tailCorrection;
}

void CalculateEnergy::VirialCorrection(Virial &virial,
                                       BoxDimensions const &boxAxes,
                                       const uint box) const {
  if (box >= BOXES_WITH_U_NB) {
    return;
  }
  double vir = 0.0;

  for (uint i = 0; i < mols.GetKindsCount(); ++i) {
    uint numI = molLookup.NumKindInBox(i, box);
    for (uint j = 0; j < mols.GetKindsCount(); ++j) {
      uint numJ = molLookup.NumKindInBox(j, box);
      vir += mols.pairVirCorrections[i * mols.GetKindsCount() + j] * numI *
             numJ * boxAxes.volInv[box];
    }
  }

  if (!forcefield.freeEnergy) {
    virial.tailCorrection = vir;
  }
#if ENSEMBLE == NVT || ENSEMBLE == NPT
  else {
    // Get the kind and lambda value
    uint fk = mols.GetMolKind(lambdaRef.GetMolIndex(box));
    double lambdaVDW = lambdaRef.GetLambdaVDW(lambdaRef.GetMolIndex(box), box);
    // remove the LRC for one molecule with lambda = 1
    vir += MoleculeTailVirChange(box, fk, false).virial;

    // Add the LRC for fractional molecule
    for (uint i = 0; i < mols.GetKindsCount(); ++i) {
      uint molNum = molLookup.NumKindInBox(i, box);
      if (i == fk) {
        --molNum; // We have one less molecule (it is fractional molecule)
      }
      double rhoDeltaIJ_2 = 2.0 * (double)(molNum)*currentAxes.volInv[box];
      vir += lambdaVDW *
             mols.pairVirCorrections[fk * mols.GetKindsCount() + i] *
             rhoDeltaIJ_2;
    }
    // We already calculated part of the change for this type in the loop
    vir += lambdaVDW * mols.pairVirCorrections[fk * mols.GetKindsCount() + fk] *
           currentAxes.volInv[box];
    virial.tailCorrection = vir;
  }
#endif
}

//! Calculate Torque
void CalculateEnergy::CalculateTorque(std::vector<uint> &moleculeIndex,
                                      XYZArray const &coordinates,
                                      XYZArray const &com,
                                      XYZArray const &atomForce,
                                      XYZArray const &atomForceRec,
                                      XYZArray &molTorque, const uint box) {
  if (multiParticleEnabled && (box < BOXES_WITH_U_NB)) {
    GOMC_EVENT_START(1, GomcProfileEvent::BOX_TORQUE);
    // make a pointer to mol torque for OpenMP
    double *torquex = molTorque.x;
    double *torquey = molTorque.y;
    double *torquez = molTorque.z;

#if defined _OPENMP
#pragma omp parallel for default(none)                                         \
    shared(atomForce, atomForceRec, com, coordinates, moleculeIndex, torquex,  \
               torquey, torquez) firstprivate(box)
#endif
    for (int m = 0; m < (int)moleculeIndex.size(); m++) {
      int mIndex = moleculeIndex[m];
      int length = mols.GetKind(mIndex).NumAtoms();
      int start = mols.MolStart(mIndex);
      double tx = 0.0;
      double ty = 0.0;
      double tz = 0.0;
      // atom iterator
      for (int p = start; p < start + length; p++) {
        XYZ distFromCOM = coordinates.Difference(p, com, mIndex);
        distFromCOM = currentAxes.MinImage(distFromCOM, box);
        XYZ tempTorque = Cross(distFromCOM, atomForce[p] + atomForceRec[p]);

        tx += tempTorque.x;
        ty += tempTorque.y;
        tz += tempTorque.z;
      }
      torquex[mIndex] = tx;
      torquey[mIndex] = ty;
      torquez[mIndex] = tz;
    }
  }
  GOMC_EVENT_STOP(1, GomcProfileEvent::BOX_TORQUE);
}

void CalculateEnergy::ResetForce(XYZArray &atomForce, XYZArray &molForce,
                                 uint box) {
  if (multiParticleEnabled) {
    uint length, start;

    // molecule iterator
    MoleculeLookup::box_iterator thisMol = molLookup.BoxBegin(box);
    MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);

    while (thisMol != end) {
      length = mols.GetKind(*thisMol).NumAtoms();
      start = mols.MolStart(*thisMol);

      molForce.Set(*thisMol, 0.0, 0.0, 0.0);
      for (uint p = start; p < start + length; p++) {
        atomForce.Set(p, 0.0, 0.0, 0.0);
      }
      thisMol++;
    }
  }
}

uint CalculateEnergy::NumberOfParticlesInsideBox(uint box) {
  uint numberOfAtoms = 0;

  for (int k = 0; k < (int)mols.GetKindsCount(); k++) {
    MoleculeKind const &thisKind = mols.kinds[k];
    numberOfAtoms += thisKind.NumAtoms() * molLookup.NumKindInBox(k, box);
  }

  return numberOfAtoms;
}

bool CalculateEnergy::FindMolInCavity(std::vector<std::vector<uint>> &mol,
                                      const XYZ &center, const XYZ &cavDim,
                                      const XYZArray &invCav, const uint box,
                                      const uint kind, const uint exRatio) {
  uint k;
  mol.clear();
  mol.resize(molLookup.GetNumKind());
  double maxLength = cavDim.Max();

  if (maxLength <= currentAxes.rCut[box]) {
    CellList::Neighbors n = cellList.EnumerateLocal(center, box);
    while (!n.Done()) {
      if (currentAxes.InCavity(currentCOM.Get(particleMol[*n]), center, cavDim,
                               invCav, box)) {
        uint molIndex = particleMol[*n];
        // if molecule can be transfer between boxes
        if (!molLookup.IsNoSwap(molIndex)) {
          k = mols.GetMolKind(molIndex);
          bool exist =
              std::find(mol[k].begin(), mol[k].end(), molIndex) != mol[k].end();
          if (!exist)
            mol[k].push_back(molIndex);
        }
      }
      n.Next();
    }
  } else {
    MoleculeLookup::box_iterator n = molLookup.BoxBegin(box);
    MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);
    while (n != end) {
      if (currentAxes.InCavity(currentCOM.Get(*n), center, cavDim, invCav,
                               box)) {
        uint molIndex = *n;
        // if molecule can be transfer between boxes
        if (!molLookup.IsNoSwap(molIndex)) {
          k = mols.GetMolKind(molIndex);
          bool exist =
              std::find(mol[k].begin(), mol[k].end(), molIndex) != mol[k].end();
          if (!exist)
            mol[k].push_back(molIndex);
        }
      }
      n++;
    }
  }

  // If the is exRate and more molecule kind in cavity, return true.
  if (mol[kind].size() >= exRatio)
    return true;
  else
    return false;
}

void CalculateEnergy::SingleMoleculeInter(
    Energy &interEnOld, Energy &interEnNew, const double lambdaOldVDW,
    const double lambdaNewVDW, const double lambdaOldCoulomb,
    const double lambdaNewCoulomb, const uint molIndex, const uint box) const {
  double tempREnOld = 0.0, tempLJEnOld = 0.0;
  double tempREnNew = 0.0, tempLJEnNew = 0.0;
  if (box < BOXES_WITH_U_NB) {
    uint length = mols.GetKind(molIndex).NumAtoms();
    uint start = mols.MolStart(molIndex);

    for (uint p = 0; p < length; ++p) {
      uint atom = start + p;
      CellList::Neighbors n = cellList.EnumerateLocal(currentCoords[atom], box);

      std::vector<uint> nIndex;
      // store atom index in neighboring cell
      while (!n.Done()) {
        if (particleMol[*n] != (int)molIndex) {
          nIndex.push_back(*n);
        }
        n.Next();
      }

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(nIndex)                          \
    firstprivate(atom, box, lambdaNewCoulomb, lambdaOldCoulomb, lambdaOldVDW,  \
                     lambdaNewVDW, num::qqFact)                                \
    reduction(+ : tempREnOld, tempLJEnOld, tempREnNew, tempLJEnNew)
#endif
      for (int i = 0; i < (int)nIndex.size(); i++) {
        double distSq = 0.0;
        XYZ virComponents;
        if (currentAxes.InRcut(distSq, virComponents, currentCoords, atom,
                               nIndex[i], box)) {
          if (electrostatic) {
            double qi_qj_fact =
                particleCharge[atom] * particleCharge[nIndex[i]] * num::qqFact;
            if (qi_qj_fact != 0.0) {
              tempREnNew += forcefield.particles->CalcCoulomb(
                  distSq, particleKind[atom], particleKind[nIndex[i]],
                  qi_qj_fact, lambdaNewCoulomb, box);
              tempREnOld += forcefield.particles->CalcCoulomb(
                  distSq, particleKind[atom], particleKind[nIndex[i]],
                  qi_qj_fact, lambdaOldCoulomb, box);
            }
          }

          tempLJEnNew += forcefield.particles->CalcEn(
              distSq, particleKind[atom], particleKind[nIndex[i]],
              lambdaNewVDW);
          tempLJEnOld += forcefield.particles->CalcEn(
              distSq, particleKind[atom], particleKind[nIndex[i]],
              lambdaOldVDW);
        }
      }
    }
  }

  interEnNew.inter = tempLJEnNew;
  interEnNew.real = tempREnNew;
  interEnOld.inter = tempLJEnOld;
  interEnOld.real = tempREnOld;
}

double CalculateEnergy::GetLambdaVDW(uint molA, uint molB, uint box) const {
  double lambda = 1.0;
  lambda *= lambdaRef.GetLambdaVDW(molA, box);
  lambda *= lambdaRef.GetLambdaVDW(molB, box);
  return lambda;
}

double CalculateEnergy::GetLambdaCoulomb(uint molA, uint molB, uint box) const {
  double lambda = 1.0;
  lambda *= lambdaRef.GetLambdaCoulomb(molA, box);
  lambda *= lambdaRef.GetLambdaCoulomb(molB, box);
  // no need for sq root for inter energy. Always one of the molecules has
  // lambda 1
  return lambda;
}

// Calculates the change in the TC from adding numChange atoms of a kind
double CalculateEnergy::MoleculeTailChange(const uint box, const uint kind,
                                           const std::vector<uint> &kCount,
                                           const double lambdaOld,
                                           const double lambdaNew) const {
  if (box >= BOXES_WITH_U_NB) {
    return 0.0;
  }

  double tcDiff = 0.0;
  uint ktot = mols.GetKindsCount();
  for (uint i = 0; i < ktot; ++i) {
    // We should have only one molecule of fractional kind
    double rhoDeltaIJ_2 = 2.0 * (double)(kCount[i]) * currentAxes.volInv[box];
    uint index = kind * ktot + i;
    tcDiff +=
        (lambdaNew - lambdaOld) * mols.pairEnCorrections[index] * rhoDeltaIJ_2;
  }
  uint index = kind * ktot + kind;
  tcDiff += (lambdaNew - lambdaOld) * mols.pairEnCorrections[index] *
            currentAxes.volInv[box];

  return tcDiff;
}

// Calculate the change in energy due to lambda
void CalculateEnergy::EnergyChange(Energy *energyDiff, Energy &dUdL_VDW,
                                   Energy &dUdL_Coul,
                                   const std::vector<double> &lambda_VDW,
                                   const std::vector<double> &lambda_Coul,
                                   const uint iState, const uint molIndex,
                                   const uint box) const {
  if (box >= BOXES_WITH_U_NB) {
    return;
  }

  GOMC_EVENT_START(1, GomcProfileEvent::FREE_ENERGY);
  uint length = mols.GetKind(molIndex).NumAtoms();
  uint start = mols.MolStart(molIndex);
  uint lambdaSize = lambda_VDW.size();
  double *tempLJEnDiff = new double[lambdaSize];
  double *tempREnDiff = new double[lambdaSize];
  double dudl_VDW = 0.0, dudl_Coul = 0.0;
  std::fill_n(tempLJEnDiff, lambdaSize, 0.0);
  std::fill_n(tempREnDiff, lambdaSize, 0.0);

  // Calculate the vdw, short range electrostatic energy
  for (uint p = 0; p < length; ++p) {
    uint atom = start + p;
    CellList::Neighbors n = cellList.EnumerateLocal(currentCoords[atom], box);

    std::vector<uint> nIndex;
    // store atom index in neighboring cell
    while (!n.Done()) {
      if (particleMol[*n] != (int)molIndex) {
        nIndex.push_back(*n);
      }
      n.Next();
    }

#if defined _OPENMP && _OPENMP >= 201511 // check if OpenMP version is 4.5
#pragma omp parallel for default(none) shared(lambda_Coul, lambda_VDW, nIndex) \
    firstprivate(box, atom, iState, lambdaSize, num::qqFact)                   \
    reduction(+ : dudl_VDW, dudl_Coul, tempREnDiff[ : lambdaSize],             \
                  tempLJEnDiff[ : lambdaSize])
#endif
    for (int i = 0; i < (int)nIndex.size(); i++) {
      double distSq = 0.0;
      XYZ virComponents;
      if (currentAxes.InRcut(distSq, virComponents, currentCoords, atom,
                             nIndex[i], box)) {
        double qi_qj_fact = 0.0, energyOldCoul = 0.0;
        // Calculate the energy of current state
        double energyOldVDW = forcefield.particles->CalcEn(
            distSq, particleKind[atom], particleKind[nIndex[i]],
            lambda_VDW[iState]);
        // Calculate du/dl in VDW for current state
        dudl_VDW += forcefield.particles->CalcdEndL(distSq, particleKind[atom],
                                                    particleKind[nIndex[i]],
                                                    lambda_VDW[iState]);

        if (electrostatic) {
          qi_qj_fact =
              particleCharge[atom] * particleCharge[nIndex[i]] * num::qqFact;
          if (qi_qj_fact != 0.0) {
            energyOldCoul = forcefield.particles->CalcCoulomb(
                distSq, particleKind[atom], particleKind[nIndex[i]], qi_qj_fact,
                lambda_Coul[iState], box);
            // Calculate du/dl in Coulomb for current state.
            dudl_Coul += forcefield.particles->CalcCoulombdEndL(
                distSq, particleKind[atom], particleKind[nIndex[i]], qi_qj_fact,
                lambda_Coul[iState], box);
          }
        }

        for (int s = 0; s < (int)lambdaSize; s++) {
          // Calculate the energy of other state
          tempLJEnDiff[s] += forcefield.particles->CalcEn(
              distSq, particleKind[atom], particleKind[nIndex[i]],
              lambda_VDW[s]);
          tempLJEnDiff[s] += -energyOldVDW;
          if (electrostatic && qi_qj_fact != 0.0) {
            tempREnDiff[s] += forcefield.particles->CalcCoulomb(
                distSq, particleKind[atom], particleKind[nIndex[i]], qi_qj_fact,
                lambda_Coul[s], box);
            tempREnDiff[s] += -energyOldCoul;
          }
        }
      }
    }
  }

  dUdL_VDW.inter = dudl_VDW;
  dUdL_Coul.real = dudl_Coul;
  for (int s = 0; s < (int)lambdaSize; s++) {
    energyDiff[s].inter += tempLJEnDiff[s];
    energyDiff[s].real += tempREnDiff[s];
  }
  delete[] tempLJEnDiff;
  delete[] tempREnDiff;

  if (forcefield.useLRC) {
    // Need to calculate change in LRC
    ChangeLRC(energyDiff, dUdL_VDW, lambda_VDW, iState, molIndex, box);
  }
  // Need to calculate change in self
  calcEwald->ChangeSelf(energyDiff, dUdL_Coul, lambda_Coul, iState, molIndex,
                        box);
  // Need to calculate change in correction
  calcEwald->ChangeCorrection(energyDiff, dUdL_Coul, lambda_Coul, iState,
                              molIndex, box);
  // Need to calculate change in Reciprocal
  calcEwald->ChangeRecip(energyDiff, dUdL_Coul, lambda_Coul, iState, molIndex,
                         box);
  GOMC_EVENT_STOP(1, GomcProfileEvent::FREE_ENERGY);
}

// Calculate the change in LRC for each state
void CalculateEnergy::ChangeLRC(Energy *energyDiff, Energy &dUdL_VDW,
                                const std::vector<double> &lambda_VDW,
                                const uint iState, const uint molIndex,
                                const uint box) const {
  // Get the kind and lambda value
  uint fk = mols.GetMolKind(molIndex);
  double lambda_istate = lambda_VDW[iState];

  // Add the LRC for fractional molecule
  for (size_t s = 0; s < lambda_VDW.size(); s++) {
    double lambdaVDW = lambda_VDW[s];
    for (uint i = 0; i < mols.GetKindsCount(); ++i) {
      uint molNum = molLookup.NumKindInBox(i, box);
      if (i == fk) {
        --molNum; // We have one less molecule (it is fractional molecule)
      }
      double rhoDeltaIJ_2 = 2.0 * (double)(molNum)*currentAxes.volInv[box];
      energyDiff[s].tailCorrection +=
          mols.pairEnCorrections[fk * mols.GetKindsCount() + i] * rhoDeltaIJ_2 *
          (lambdaVDW - lambda_istate);
      if (s == iState) {
        // Calculate du/dl in VDW LRC for current state
        dUdL_VDW.tailCorrection +=
            mols.pairEnCorrections[fk * mols.GetKindsCount() + i] *
            rhoDeltaIJ_2;
      }
    }
    energyDiff[s].tailCorrection +=
        mols.pairEnCorrections[fk * mols.GetKindsCount() + fk] *
        currentAxes.volInv[box] * (lambdaVDW - lambda_istate);
    if (s == iState) {
      // Calculate du/dl in VDW LRC for current state
      dUdL_VDW.tailCorrection +=
          mols.pairEnCorrections[fk * mols.GetKindsCount() + fk] *
          currentAxes.volInv[box];
    }
  }
}

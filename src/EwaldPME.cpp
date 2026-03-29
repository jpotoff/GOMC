#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
#include <cstring>

#include "EwaldPME.h"
#include "BSpline.h"
#include "BoxDimensions.h"
#include "EnergyTypes.h"
#include "NumLib.h"
#include "StaticVals.h"
#include "System.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

EwaldPME::EwaldPME(StaticVals &stat, System &sys) : Ewald(stat, sys) {
  // ff is the protected Forcefield& member inherited from Ewald
  pmeOrder = ff.pmeSplineOrder;
  refreshInterval = ff.pmeRefreshFreq;

  S_ref = nullptr;
  greenFunc = nullptr;
  chargeMesh = nullptr;
  potentialMesh = nullptr;
  scratchMesh = nullptr;
  S_delta = nullptr;
  S_trial = nullptr;
  greenFunc_trial = nullptr;

  for (uint b = 0; b < BOX_TOTAL; b++) {
    fwdPlan[b] = nullptr;
    bwdPlan[b] = nullptr;
    tempEnergyRecip[b] = 0.0;
    tempVirialRecip[b].Zero();
  }

  pendingUpdate = false;
  forceFullUpdate = false;
  cachedBox = 0;
  cachedNAtoms = 0;
}

EwaldPME::~EwaldPME() {
  for (uint b = 0; b < BOX_TOTAL; b++) {
    if (S_ref && S_ref[b]) fftw_free(S_ref[b]);
    if (greenFunc && greenFunc[b]) delete[] greenFunc[b];
    if (chargeMesh && chargeMesh[b]) fftw_free(chargeMesh[b]);
    if (potentialMesh && potentialMesh[b]) fftw_free(potentialMesh[b]);
    if (fwdPlan[b]) fftw_destroy_plan(fwdPlan[b]);
    if (bwdPlan[b]) fftw_destroy_plan(bwdPlan[b]);
    if (S_trial && S_trial[b]) fftw_free(S_trial[b]);
    if (greenFunc_trial && greenFunc_trial[b]) delete[] greenFunc_trial[b];
  }
  delete[] S_ref;
  delete[] greenFunc;
  delete[] chargeMesh;
  delete[] potentialMesh;
  delete[] S_trial;
  delete[] greenFunc_trial;
}

void EwaldPME::Init() {
  Ewald::Init(); // call base class initialization
}

void EwaldPME::AllocMem() {
  // Ewald base class configures the arrays for atoms and coordinates.
  Ewald::AllocMem();

  S_ref = new fftw_complex *[BOX_TOTAL];
  greenFunc = new double *[BOX_TOTAL];
  chargeMesh = new double *[BOX_TOTAL];
  potentialMesh = new double *[BOX_TOTAL];
  scratchMesh = new double *[BOX_TOTAL];
  S_delta = new fftw_complex *[BOX_TOTAL];
  S_trial = new fftw_complex *[BOX_TOTAL];
  greenFunc_trial = new double *[BOX_TOTAL];

  for (uint b = 0; b < BOX_TOTAL; b++) {
    S_ref[b] = nullptr;
    greenFunc[b] = nullptr;
    chargeMesh[b] = nullptr;
    potentialMesh[b] = nullptr;
    scratchMesh[b] = nullptr;
    S_delta[b] = nullptr;
    fwdPlan[b] = nullptr;
    bwdPlan[b] = nullptr;
    scratchPlan[b] = nullptr;
    S_trial[b] = nullptr;
    greenFunc_trial[b] = nullptr;
    K[b][0] = K[b][1] = K[b][2] = 0;
    K_trial[b][0] = K_trial[b][1] = K_trial[b][2] = 0;
  }
}
void EwaldPME::BoxReciprocalSetup(uint box, XYZArray const &molCoords) {
  if (box >= BOXES_WITH_U_NB)
    return;

  // Set grid dimensions based on user-defined pmeGridSpacing
  // Kx, Ky, Kz should be at least pmeOrder
  double gridSpacing = ff.pmeGridSpacing <= 0.0 ? 1.0 : ff.pmeGridSpacing;

  int k0 = (int)ceil(trialAxes[box].axis.Get(box).x / gridSpacing);
  int k1 = (int)ceil(trialAxes[box].axis.Get(box).y / gridSpacing);
  int k2 = (int)ceil(trialAxes[box].axis.Get(box).z / gridSpacing);

  K_trial[box][0] = (pmeOrder > k0) ? pmeOrder : k0;
  K_trial[box][1] = (pmeOrder > k1) ? pmeOrder : k1;
  K_trial[box][2] = (pmeOrder > k2) ? pmeOrder : k2;

  for (int i = 0; i < 3; ++i)
    if (K_trial[box][i] % 2 != 0)
      K_trial[box][i]++;

  int Kx = K_trial[box][0];
  int Ky = K_trial[box][1];
  int Kz = K_trial[box][2];
  int N = Kx * Ky * Kz;
  int N_complex = Kx * Ky * (Kz / 2 + 1);

  bool kChanged = (Kx != K[box][0] || Ky != K[box][1] || Kz != K[box][2]);

  if (kChanged || fwdPlan[box] == nullptr) {
    // Destroy old plans if they exist
    if (fwdPlan[box]) fftw_destroy_plan(fwdPlan[box]);
    if (bwdPlan[box]) fftw_destroy_plan(bwdPlan[box]);
    if (scratchPlan[box]) fftw_destroy_plan(scratchPlan[box]);

    // Free old buffers if they exist
    if (S_ref[box]) fftw_free(S_ref[box]);
    if (S_trial[box]) fftw_free(S_trial[box]);
    if (chargeMesh[box]) fftw_free(chargeMesh[box]);
    if (potentialMesh[box]) fftw_free(potentialMesh[box]);
    if (scratchMesh[box]) fftw_free(scratchMesh[box]);
    if (S_delta[box]) fftw_free(S_delta[box]);
    if (greenFunc[box]) delete[] greenFunc[box];
    if (greenFunc_trial[box]) delete[] greenFunc_trial[box];

    // Allocate new buffers
    S_ref[box] = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N_complex);
    S_trial[box] = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N_complex);
    chargeMesh[box] = (double *)fftw_malloc(sizeof(double) * N);
    potentialMesh[box] = (double *)fftw_malloc(sizeof(double) * N);
    scratchMesh[box] = (double *)fftw_malloc(sizeof(double) * N);
    S_delta[box] = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N_complex);
    greenFunc[box] = new double[N_complex];
    greenFunc_trial[box] = new double[N_complex];

    // Create new plans bound to these specific buffers
    fwdPlan[box] = fftw_plan_dft_r2c_3d(Kx, Ky, Kz, chargeMesh[box], S_trial[box],
                                        FFTW_ESTIMATE);
    bwdPlan[box] = fftw_plan_dft_c2r_3d(Kx, Ky, Kz, S_trial[box], potentialMesh[box],
                                        FFTW_ESTIMATE);
    scratchPlan[box] = fftw_plan_dft_r2c_3d(Kx, Ky, Kz, scratchMesh[box], S_delta[box],
                                            FFTW_ESTIMATE);
    
    // Sync permanent grid dimensions
    K[box][0] = Kx; K[box][1] = Ky; K[box][2] = Kz;
  }

  // Temporarily point greenFunc[box] to the trial buffer for calculation
  double *gt = greenFunc[box];
  greenFunc[box] = greenFunc_trial[box];

  ComputeGreenFunction(box);
  ComputeChargeMesh(box, molCoords);
  fftw_execute(fwdPlan[box]); // writes to S_trial[box]

  // Sum energy from S_trial
  tempEnergyRecip[box] =
      SumMeshEnergy(box, S_trial[box], &tempVirialRecip[box]);

  // Restore greenFunc[box] pointer
  greenFunc[box] = gt;
}

void EwaldPME::BoxReciprocalSums(uint box, XYZArray const &molCoords) {
  if (box >= BOXES_WITH_U_NB)
    return;
  // This is a full refresh. Use Setup logic then immediately accept results.
  RecipInit(box, currentAxes);
  BoxReciprocalSetup(box, molCoords);
  
  // Immediately accept results into reference
  UpdateRecipVec(box);
}

double EwaldPME::BoxReciprocal(uint box, bool isNewVolume) const {
  if (box >= BOXES_WITH_U_NB)
    return 0.0;
  return isNewVolume ? tempEnergyRecip[box] : sysPotRef.boxEnergy[box].recip;
}

void EwaldPME::RecipInit(uint box, BoxDimensions const &axes) {
  Ewald::RecipInit(box, axes);
  trialAxes[box] = axes; // Cache the trial axes for use by mesh/Green functions
}

void EwaldPME::UpdateRecipVec(uint box) {
  if (box >= BOXES_WITH_U_NB)
    return;

  // We no longer swap pointers in the array.
  // Instead we copy data from trial to reference.
  int nk = K[box][0] * K[box][1] * (K[box][2] / 2 + 1);
  std::memcpy(S_ref[box], S_trial[box], sizeof(fftw_complex) * nk);
  std::memcpy(greenFunc[box], greenFunc_trial[box], sizeof(double) * nk);

  // Update sysPotRef
  const_cast<SystemPotential &>(sysPotRef).boxEnergy[box].recip =
      tempEnergyRecip[box];
  const_cast<SystemPotential &>(sysPotRef).boxVirial[box].recip =
      tempVirialRecip[box].recip;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      const_cast<SystemPotential &>(sysPotRef).boxVirial[box].recipTens[i][j] =
          tempVirialRecip[box].recipTens[i][j];
    }
  }
  
  // Sync trialAxes so subsequent per-molecule moves see consistent geometry.
  trialAxes[box] = currentAxes;
  
  // Potential mesh must be recomputed for the new state
  UpdatePotentialMesh(box);
}

void EwaldPME::UpdateRecip(uint box) {
  if (box >= BOXES_WITH_U_NB || !pendingUpdate || box != cachedBox)
    return;

  if (forceFullUpdate) {
    BoxReciprocalSetup(box, currentCoords);
    int nk = K[box][0] * K[box][1] * (K[box][2] / 2 + 1);
    std::memcpy(S_ref[box], S_trial[box], sizeof(fftw_complex) * nk);
    UpdatePotentialMesh(box);
  } else {
    // Apply incremental update using DeltaERecip with updateSRef = true
    const XYZArray *pNew = (cachedSignNew != 0.0) ? &cachedNewCoords : nullptr;
    const XYZArray *pOld = (cachedSignOld != 0.0) ? &cachedOldCoords : nullptr;
    double exactRecip = DeltaERecip(box, pNew, cachedSignNew, pOld, cachedSignOld,
                                    cachedAtomIndices.data(), cachedCharges.data(), cachedNAtoms,
                                    true);
    const_cast<SystemPotential &>(sysPotRef).boxEnergy[box].recip = exactRecip;
  }

  // NOTE: By explicitly overriding sysPotRef.boxEnergy[box].recip with the newly
  // calculated exact mesh reciprocal energy from DeltaERecip(..., true), we 
  // eliminate interpolation drift. This is mathematically necessary because VolumeTransfer 
  // evaluates the trial volume's BoxReciprocal() exact energy and compares it 
  // against sysPotRef. Any accumulated interpolation drift artificially biases NPT moves.

  pendingUpdate = false;
  forceFullUpdate = false;
}

void EwaldPME::Maintain(const ulong step) {
  if (refreshInterval == 0 || (step + 1) % refreshInterval != 0)
    return;

  for (uint b = 0; b < BOXES_WITH_U_NB; b++) {
    BoxReciprocalSums(b, currentCoords);
  }
}

// Called by GOMC when a volume move is rejected, to undo the trial state.
// The base Ewald class no-ops here, but EwaldPME must restore chargeMesh because
// BoxReciprocalSetup overwrites it with the trial charge density and leaves it
// corrupted after a rejection (currentCoords and box revert, but chargeMesh does not).
void EwaldPME::RestoreMol(int molIndex) {
  cachedSignNew = 0.0;
  pendingUpdate = false;
}

void EwaldPME::exgMolCache() {
  Ewald::exgMolCache(); // call base (currently a no-op, but keeps the chain intact)
  for (uint box = 0; box < BOXES_WITH_U_NB; ++box) {
    if (chargeMesh[box] != nullptr) {
      // Check if K dimensions were corrupted by a rejected volume trial
      double gridSpacing = ff.pmeGridSpacing <= 0.0 ? 1.0 : ff.pmeGridSpacing;
      int k0 = (int)ceil(currentAxes.axis.Get(box).x / gridSpacing);
      int k1 = (int)ceil(currentAxes.axis.Get(box).y / gridSpacing);
      int k2 = (int)ceil(currentAxes.axis.Get(box).z / gridSpacing);

      int Kx = (pmeOrder > k0) ? pmeOrder : k0;
      int Ky = (pmeOrder > k1) ? pmeOrder : k1;
      int Kz = (pmeOrder > k2) ? pmeOrder : k2;

      if (Kx % 2 != 0) Kx++;
      if (Ky % 2 != 0) Ky++;
      if (Kz % 2 != 0) Kz++;

      if (K[box][0] != Kx || K[box][1] != Ky || K[box][2] != Kz) {
        // K dimensions were permanently altered during BoxReciprocalSetup but the move was rejected.
        // The previously committed S_ref and potentialMesh were destroyed. Rebuild them completely.
        BoxReciprocalSums(box, currentCoords);
      } else {
        // Restore trialAxes and Green function to the committed (current) volume.
        RecipInit(box, currentAxes);
        // Rebuild chargeMesh to reflect the committed (current) coordinates.
        ComputeChargeMesh(box, currentCoords);
        // CRITICAL: Recompute S_ref from the restored chargeMesh.
        // BoxReciprocalSetup wrote the trial FFT into S_trial; the fwdPlan target
        // is S_trial. We must write the current-state FFT back to S_ref.
        // Execute the forward FFT (writes to S_trial as per the plan)...
        fftw_execute(fwdPlan[box]);
        // ...then copy S_trial → S_ref (same as UpdateRecipVec does on acceptance).
        int nk = K[box][0] * K[box][1] * (K[box][2] / 2 + 1);
        std::memcpy(S_ref[box], S_trial[box], sizeof(fftw_complex) * nk);
        // Rebuild the real-space potential mesh so interpolation reads fresh data.
        UpdatePotentialMesh(box);
        // Recompute exact energy and store in sysPotRef to stay consistent.
        const_cast<SystemPotential &>(sysPotRef).boxEnergy[box].recip =
            SumMeshEnergy(box, S_ref[box]);
      }
    }
  }
}

Virial EwaldPME::VirialReciprocal(Virial &virial, uint box) const {
  if (box >= BOXES_WITH_U_NB)
    return virial;

  virial.recip = sysPotRef.boxVirial[box].recip;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      virial.recipTens[i][j] = sysPotRef.boxVirial[box].recipTens[i][j];
    }
  }
  return virial;
}

void EwaldPME::ComputeChargeMesh(uint box, XYZArray const &molCoords) {
  if (box >= BOXES_WITH_U_NB)
    return;

  int Kx = K[box][0], Ky = K[box][1], Kz = K[box][2];
  double *mesh = chargeMesh[box];
  std::fill(mesh, mesh + (Kx * Ky * Kz), 0.0);

  std::vector<unsigned int> molIds;
  for (auto it = molLookup.BoxBegin(box); it != molLookup.BoxEnd(box); ++it) {
    molIds.push_back(*it);
  }

#ifdef _OPENMP
#pragma omp parallel for default(shared)
#endif
  for (int m = 0; m < (int)molIds.size(); ++m) {
    AddMoleculeToMesh(box, molIds[m], molCoords, mesh);
  }
}

void EwaldPME::AddMoleculeToMesh(uint box, uint molIndex,
                                 XYZArray const &molCoords, double *mesh) {
  int Kx = K[box][0], Ky = K[box][1], Kz = K[box][2];
  MoleculeKind const &thisKind = mols.GetKind(molIndex);
  uint startAtom = mols.MolStart(molIndex);
  double lambdaCoef = GetLambdaCoef(molIndex, box);
  double bs_x[20], bs_y[20], bs_z[20];

  for (uint i = 0; i < thisKind.NumAtoms(); ++i) {
    uint atomIndex = startAtom + i;
    if (particleHasNoCharge[atomIndex])
      continue;

    double charge = thisKind.AtomCharge(i) * lambdaCoef;
    XYZ r = molCoords.Get(atomIndex);
    XYZ s = trialAxes[box].TransformUnSlant(r, box);
    double sx = s.x / trialAxes[box].axis.Get(box).x;
    double sy = s.y / trialAxes[box].axis.Get(box).y;
    double sz = s.z / trialAxes[box].axis.Get(box).z;

    sx -= std::floor(sx);
    sy -= std::floor(sy);
    sz -= std::floor(sz);

    double ux = sx * Kx, uy = sy * Ky, uz = sz * Kz;
    bspline::EvalAll(pmeOrder, ux - std::floor(ux), bs_x);
    bspline::EvalAll(pmeOrder, uy - std::floor(uy), bs_y);
    bspline::EvalAll(pmeOrder, uz - std::floor(uz), bs_z);

    int nx = (int)std::floor(ux) - pmeOrder + 1;
    int ny = (int)std::floor(uy) - pmeOrder + 1;
    int nz = (int)std::floor(uz) - pmeOrder + 1;

    for (int ix = 0; ix < pmeOrder; ++ix) {
      int gx = ((nx + ix) % Kx + Kx) % Kx;
      double wx = bs_x[pmeOrder - 1 - ix];
      for (int iy = 0; iy < pmeOrder; ++iy) {
        int gy = ((ny + iy) % Ky + Ky) % Ky;
        double wy = wx * bs_y[pmeOrder - 1 - iy];
        for (int iz = 0; iz < pmeOrder; ++iz) {
          int gz = ((nz + iz) % Kz + Kz) % Kz;
          double w = charge * wy * bs_z[pmeOrder - 1 - iz];
          int idx = (gx * Ky + gy) * Kz + gz;
#ifdef _OPENMP
#pragma omp atomic update
#endif
          mesh[idx] += w;
        }
      }
    }
  }
}
void EwaldPME::ComputeGreenFunction(uint box) {
  if (box >= BOXES_WITH_U_NB)
    return;

  int Kx = K[box][0];
  int Ky = K[box][1];
  int Kz = K[box][2];

  std::vector<double> bx = bspline::BSplineModuli(Kx, pmeOrder);
  std::vector<double> by = bspline::BSplineModuli(Ky, pmeOrder);
  std::vector<double> bz = bspline::BSplineModuli(Kz, pmeOrder);

  double volume = trialAxes[box].volume[box];
  // Standard Ewald scaling: 4*PI * qqFact / V
  double fac = 4.0 * M_PI * num::qqFact / volume;

  double alphaSq = ff.alphaSq[box];

  int halfKz = Kz / 2 + 1;
  int nk = Kx * Ky * halfKz;
  double *Gptr = greenFunc[box];

#ifdef _OPENMP
#pragma omp parallel for default(shared)
#endif
  for (int i = 0; i < nk; ++i) {
    int iz = i % halfKz;
    int iy_iz = i / halfKz;
    int ix = iy_iz / Ky;
    int iy = iy_iz % Ky;

    int kx_int = (ix <= Kx / 2) ? ix : ix - Kx;
    int ky_int = (iy <= Ky / 2) ? iy : iy - Ky;

    if (ix == 0 && iy == 0 && iz == 0) {
      Gptr[i] = 0.0;
      continue;
    }

    double kx_cart = 2.0 * M_PI * kx_int / trialAxes[box].axis.Get(box).x;
    double ky_cart = 2.0 * M_PI * ky_int / trialAxes[box].axis.Get(box).y;
    double kz_cart = 2.0 * M_PI * iz / trialAxes[box].axis.Get(box).z;
    double kSq = kx_cart * kx_cart + ky_cart * ky_cart + kz_cart * kz_cart;

    double BFunc = bx[ix] * by[iy] * bz[iz];
    double expTerm = std::exp(-kSq / (4.0 * alphaSq));
    Gptr[i] = BFunc * expTerm * fac / kSq;
  }
}

double EwaldPME::SumMeshEnergy(uint box, const fftw_complex *S,
                               Virial *virial) const {
  if (box >= BOXES_WITH_U_NB)
    return 0.0;

  int Kx = K[box][0];
  int Ky = K[box][1];
  int Kz = K[box][2];

  double energy = 0.0;
  double wT11 = 0.0, wT22 = 0.0, wT33 = 0.0;
  double constVal = 1.0 / (4.0 * ff.alphaSq[box]);

  int halfKz = (Kz / 2 + 1);
  int nk = Kx * Ky * halfKz;
  double *Gptr = greenFunc[box];

#ifdef _OPENMP
#pragma omp parallel for default(shared)                                       \
    reduction(+ : energy, wT11, wT22, wT33)
#endif
  for (int i = 0; i < nk; ++i) {
    int iz = i % halfKz;
    int iy_iz = i / halfKz;
    int ix = iy_iz / Ky;
    int iy = iy_iz % Ky;

    if (ix == 0 && iy == 0 && iz == 0)
      continue;

    double S_sq = S[i][0] * S[i][0] + S[i][1] * S[i][1];

    // Multiply by 2.0 for everything except the iz=0 and iz=Kz/2 planes
    double weight = (iz == 0 || (Kz % 2 == 0 && iz == Kz / 2)) ? 1.0 : 2.0;

    double G = Gptr[i];
    double term = 0.5 * weight * G * S_sq;
    energy += term;

    if (virial) {
      int kx_int = (ix <= Kx / 2) ? ix : ix - Kx;
      int ky_int = (iy <= Ky / 2) ? iy : iy - Ky;
      double kx_cart = 2.0 * M_PI * kx_int / trialAxes[box].axis.Get(box).x;
      double ky_cart = 2.0 * M_PI * ky_int / trialAxes[box].axis.Get(box).y;
      double kz_cart = 2.0 * M_PI * iz / trialAxes[box].axis.Get(box).z;
      double kSq = kx_cart * kx_cart + ky_cart * ky_cart + kz_cart * kz_cart;

      double common = 2.0 * (constVal + 1.0 / kSq);
      wT11 += term * (1.0 - common * kx_cart * kx_cart);
      wT22 += term * (1.0 - common * ky_cart * ky_cart);
      wT33 += term * (1.0 - common * kz_cart * kz_cart);
    }
  }

  if (virial) {
    virial->recipTens[0][0] = wT11;
    virial->recipTens[1][1] = wT22;
    virial->recipTens[2][2] = wT33;
    virial->recip = wT11 + wT22 + wT33;
  }

  return energy;
}
void EwaldPME::UpdatePotentialMesh(uint box) {
  if (box >= BOXES_WITH_U_NB || S_ref[box] == nullptr)
    return;
  int Kx = K[box][0], Ky = K[box][1], Kz = K[box][2];
  int halfKz = Kz / 2 + 1;
  int nk = Kx * Ky * halfKz;
  double *G = greenFunc[box];
  fftw_complex *stref = S_ref[box];

  // We use S_trial as a temporary buffer to avoid modifying S_ref.
  // We need to compute V(m) = G(m) * S_ref(m)
  for (int i = 0; i < nk; ++i) {
    S_trial[box][i][0] = G[i] * stref[i][0];
    S_trial[box][i][1] = G[i] * stref[i][1];
  }

  // Backward FFT to get potential in real space
  fftw_execute(bwdPlan[box]);

  // Normalize by number of grid points (standard for IFFT)
  // HOWEVER: the real space dot product \sum q \phi(r) equals \sum_m S(m) V(m)
  // ONLY IF the iFFT is unnormalized (i.e., just \sum V(m) exp(i m r)).
  // FFTW backward transform calculates exactly this. Dividing by N breaks Parseval's theorem!
  // We MUST leave potentialMesh unnormalized.
  // int N = Kx * Ky * Kz;
  // double invN = 1.0 / (double)N;
  // double *mesh = potentialMesh[box];
  // for (int i = 0; i < N; ++i) {
  //   mesh[i] *= invN;
  // }
}

void EwaldPME::UpdateAtomInMesh(uint box, const double *charges, uint nAtoms,
                                const XYZArray &coords, double sign) {
  int Kx = K[box][0], Ky = K[box][1], Kz = K[box][2];
  double bs_x[20], bs_y[20], bs_z[20];
  for (uint i = 0; i < nAtoms; ++i) {
    double charge = charges[i] * sign;
    XYZ r = coords.Get(i);
    XYZ s = currentAxes.TransformUnSlant(r, box);
    double sx = s.x / currentAxes.axis.Get(box).x;
    double sy = s.y / currentAxes.axis.Get(box).y;
    double sz = s.z / currentAxes.axis.Get(box).z;
    sx -= floor(sx); sy -= floor(sy); sz -= floor(sz);

    double ux = sx * Kx, uy = sy * Ky, uz = sz * Kz;
    bspline::EvalAll(pmeOrder, ux - floor(ux), bs_x);
    bspline::EvalAll(pmeOrder, uy - floor(uy), bs_y);
    bspline::EvalAll(pmeOrder, uz - floor(uz), bs_z);

    int nx = (int)floor(ux) - pmeOrder + 1;
    int ny = (int)floor(uy) - pmeOrder + 1;
    int nz = (int)floor(uz) - pmeOrder + 1;

    for (int ix = 0; ix < pmeOrder; ++ix) {
      int gx = ((nx + ix) % Kx + Kx) % Kx;
      double wx = bs_x[pmeOrder - 1 - ix];
      for (int iy = 0; iy < pmeOrder; ++iy) {
        int gy = ((ny + iy) % Ky + Ky) % Ky;
        double wy = wx * bs_y[pmeOrder - 1 - iy];
        for (int iz = 0; iz < pmeOrder; ++iz) {
          int gz = ((nz + iz) % Kz + Kz) % Kz;
          double w = charge * wy * bs_z[pmeOrder - 1 - iz];
          int idx = (gx * Ky + gy) * Kz + gz;
          chargeMesh[box][idx] += w;
        }
      }
    }
  }
}

double EwaldPME::InterpolatePotential(uint box, const XYZ &r) const {
  int Kx = K[box][0], Ky = K[box][1], Kz = K[box][2];
  double *vMesh = potentialMesh[box];

  XYZ s = currentAxes.TransformUnSlant(r, box);
  double sx = s.x / currentAxes.axis.Get(box).x;
  double sy = s.y / currentAxes.axis.Get(box).y;
  double sz = s.z / currentAxes.axis.Get(box).z;
  sx -= floor(sx);
  sy -= floor(sy);
  sz -= floor(sz);

  double ux = sx * Kx, uy = sy * Ky, uz = sz * Kz;
  double bs_x[20], bs_y[20], bs_z[20];
  bspline::EvalAll(pmeOrder, ux - floor(ux), bs_x);
  bspline::EvalAll(pmeOrder, uy - floor(uy), bs_y);
  bspline::EvalAll(pmeOrder, uz - floor(uz), bs_z);

  int nx = (int)floor(ux) - pmeOrder + 1;
  int ny = (int)floor(uy) - pmeOrder + 1;
  int nz = (int)floor(uz) - pmeOrder + 1;

  double atomPotential = 0.0;
  for (int ix = 0; ix < pmeOrder; ++ix) {
    int gx = ((nx + ix) % Kx + Kx) % Kx;
    double wx = bs_x[pmeOrder - 1 - ix];
    for (int iy = 0; iy < pmeOrder; ++iy) {
      int gy = ((ny + iy) % Ky + Ky) % Ky;
      double wy = wx * bs_y[pmeOrder - 1 - iy];
      for (int iz = 0; iz < pmeOrder; ++iz) {
        int gz = ((nz + iz) % Kz + Kz) % Kz;
        double w = wy * bs_z[pmeOrder - 1 - iz];
        int idx = (gx * Ky + gy) * Kz + gz;
        atomPotential += w * vMesh[idx];
      }
    }
  }
  return atomPotential;
}

// ---------------------------------------------------------------------------
// Core incremental energy update.
// ΔE = Σ_m C(m) · [Re(S_ref(m)·ΔS*(m)) + ½|ΔS(m)|²]
// If updateSRef: S_ref(m) += ΔS(m)  (called on accept)
// ---------------------------------------------------------------------------
double EwaldPME::ComputeDeltaSsq(uint box, const XYZArray *newCoords, double sign_new,
                                 const XYZArray *oldCoords, double sign_old,
                                 const uint *atomIndices, const double *charges,
                                 uint nAtoms) const {
  if (nAtoms == 0)
    return 0.0;

  int Kx = K[box][0], Ky = K[box][1], Kz = K[box][2];
  int N = Kx * Ky * Kz;
  double *mesh = const_cast<double *>(scratchMesh[box]);
  std::fill(mesh, mesh + N, 0.0);

  double bs_x[20], bs_y[20], bs_z[20];
  auto addAtoms = [&](const XYZArray *coords, double sign) {
    if (!coords) return;
    for (uint i = 0; i < nAtoms; ++i) {
      if (particleHasNoCharge[atomIndices[i]]) continue;
      double charge = charges[i] * sign;
      XYZ r = coords->Get(i);
      XYZ s = currentAxes.TransformUnSlant(r, box);
      double sx = s.x / currentAxes.axis.Get(box).x;
      double sy = s.y / currentAxes.axis.Get(box).y;
      double sz = s.z / currentAxes.axis.Get(box).z;
      sx -= floor(sx); sy -= floor(sy); sz -= floor(sz);
      double ux = sx * Kx, uy = sy * Ky, uz = sz * Kz;
      bspline::EvalAll(pmeOrder, ux - floor(ux), bs_x);
      bspline::EvalAll(pmeOrder, uy - floor(uy), bs_y);
      bspline::EvalAll(pmeOrder, uz - floor(uz), bs_z);
      int nx = (int)floor(ux) - pmeOrder + 1;
      int ny = (int)floor(uy) - pmeOrder + 1;
      int nz = (int)floor(uz) - pmeOrder + 1;
      for (int ix = 0; ix < pmeOrder; ++ix) {
        int gx = ((nx + ix) % Kx + Kx) % Kx;
        double wx = bs_x[pmeOrder - 1 - ix];
        for (int iy = 0; iy < pmeOrder; ++iy) {
          int gy = ((ny + iy) % Ky + Ky) % Ky;
          double wy = wx * bs_y[pmeOrder - 1 - iy];
          for (int iz = 0; iz < pmeOrder; ++iz) {
            int gz = ((nz + iz) % Kz + Kz) % Kz;
            double w = charge * wy * bs_z[pmeOrder - 1 - iz];
            mesh[(gx * Ky + gy) * Kz + gz] += w;
          }
        }
      }
    }
  };

  addAtoms(newCoords, sign_new);
  addAtoms(oldCoords, sign_old);

  fftw_execute(scratchPlan[box]);
  return SumMeshEnergy(box, S_delta[box]);
}

// ---------------------------------------------------------------------------
// Core incremental energy update.
// ΔE = Σ_m C(m) · [Re(S_ref(m)·ΔS*(m)) + ½|ΔS(m)|²]
// If updateSRef: S_ref(m) += ΔS(m)  (called on accept)
// ---------------------------------------------------------------------------
double EwaldPME::DeltaERecip(uint box, const XYZArray *newCoords,
                             double sign_new, const XYZArray *oldCoords,
                             double sign_old, const uint *atomIndices,
                             const double *charges, uint nAtoms,
                             bool updateSRef) {
  if (box >= BOXES_WITH_U_NB || S_ref == nullptr || S_ref[box] == nullptr)
    return 0.0;

  double dE = 0.0;
  // Background interaction part: Interpolate potential Mesh at new/old positions
  // ΔE_back = Σ q_i_new φ(r_i_new) - Σ q_i_old φ(r_i_old)
  if (newCoords) {
    for (uint a = 0; a < nAtoms; ++a) {
      if (!particleHasNoCharge[atomIndices[a]]) {
        dE += sign_new * charges[a] *
              InterpolatePotential(box, newCoords->Get(a));
      }
    }
  }
  if (oldCoords) {
    for (uint a = 0; a < nAtoms; ++a) {
      if (!particleHasNoCharge[atomIndices[a]]) {
        dE += sign_old * charges[a] *
              InterpolatePotential(box, oldCoords->Get(a));
      }
    }
  }

  // Self-energy part: 0.5 * Σ G(m) |ΔS(m)|^2
  dE += ComputeDeltaSsq(box, newCoords, sign_new, oldCoords, sign_old, 
                        atomIndices, charges, nAtoms);

  // Optionally commit the update to S_ref (on move acceptance).
  // Incrementally update chargeMesh then re-FFT to get new S_ref.
  // chargeMesh is maintained correctly between per-molecule moves:
  // it holds Q(r) for the currently committed coordinates and is restored
  // after volume trials by BoxReciprocalSetup's save/restore.
  if (updateSRef) {
    if (oldCoords) UpdateAtomInMesh(box, charges, nAtoms, *oldCoords, sign_old);
    if (newCoords) UpdateAtomInMesh(box, charges, nAtoms, *newCoords, sign_new);
    fftw_execute(fwdPlan[box]);
    
    // Copy S_trial[box] to S_ref[box] for future references
    int nk = K[box][0] * K[box][1] * (K[box][2] / 2 + 1);
    std::memcpy(S_ref[box], S_trial[box], sizeof(fftw_complex) * nk);
    for (int i = 0; i < nk; ++i) {
      S_ref[box][i][0] = S_trial[box][i][0];
      S_ref[box][i][1] = S_trial[box][i][1];
    }
    UpdatePotentialMesh(box);
  }

  // Always return the exact energy of the successfully committed S_ref
  // (S_trial is destroyed during UpdatePotentialMesh's c2r transform)
  if (updateSRef) {
    return SumMeshEnergy(box, S_ref[box]);
  }
  return dE;
}

// ---------------------------------------------------------------------------
// Per-move overrides — all delegate to DeltaERecip
// ---------------------------------------------------------------------------

double EwaldPME::MolReciprocal(XYZArray const &molCoords, const uint molIndex,
                               const uint box) {
  if (box >= BOXES_WITH_U_NB || S_ref == nullptr || S_ref[box] == nullptr)
    return 0.0;

  MoleculeKind const &kind = mols.GetKind(molIndex);
  uint startAtom = mols.MolStart(molIndex);
  uint nAtoms = kind.NumAtoms();
  double lambdaCoef = GetLambdaCoef(molIndex, box);

  std::vector<uint> idx(nAtoms);
  std::vector<double> chg(nAtoms);
  for (uint i = 0; i < nAtoms; ++i) {
    idx[i] = startAtom + i;
    chg[i] = kind.AtomCharge(i) * lambdaCoef;
  }

  // Old positions are in currentCoords; new positions in molCoords
  // ΔE = E(new) - E(old)  => sign_new=+1, sign_old=-1
  XYZArray localOld(nAtoms);
  for (uint i = 0; i < nAtoms; ++i) {
    localOld.Set(i, currentCoords.Get(idx[i]));
  }
  cachedBox = box;
  cachedSignNew = +1.0;
  cachedSignOld = -1.0;
  cachedNewCoords = molCoords;
  cachedOldCoords = localOld;
  cachedAtomIndices = idx;
  cachedCharges = chg;
  cachedNAtoms = nAtoms;
  pendingUpdate = true;
  forceFullUpdate = false;

  return DeltaERecip(box, &molCoords, +1.0, &localOld, -1.0, idx.data(),
                     chg.data(), nAtoms, /*updateSRef=*/false);
}

double EwaldPME::SwapDestRecip(const cbmc::TrialMol &newMol, const uint box,
                               const int molIndex) {
  if (box >= BOXES_WITH_U_NB || S_ref == nullptr || S_ref[box] == nullptr)
    return 0.0;

  MoleculeKind const &kind = newMol.GetKind();
  uint nAtoms = kind.NumAtoms();
  uint startAtom = mols.MolStart(molIndex);

  std::vector<uint> idx(nAtoms);
  std::vector<double> chg(nAtoms);
  for (uint i = 0; i < nAtoms; ++i) {
    idx[i] = startAtom + i;
    chg[i] = kind.AtomCharge(i);
  }

  // Insertion into dest box: new coords only (no old positions)
  const XYZArray &coords = newMol.GetCoords();

  cachedBox = box;
  cachedSignNew = +1.0;
  cachedSignOld = 0.0;
  cachedNewCoords = coords;
  cachedAtomIndices = idx;
  cachedCharges = chg;
  cachedNAtoms = nAtoms;
  pendingUpdate = true;
  forceFullUpdate = false;

  return DeltaERecip(box, &coords, +1.0, nullptr, 0.0, idx.data(), chg.data(),
                     nAtoms, false);
}

double EwaldPME::SwapSourceRecip(const cbmc::TrialMol &oldMol, const uint box,
                                 const int molIndex) {
  if (box >= BOXES_WITH_U_NB || S_ref == nullptr || S_ref[box] == nullptr)
    return 0.0;

  MoleculeKind const &kind = oldMol.GetKind();
  uint nAtoms = kind.NumAtoms();
  uint startAtom = mols.MolStart(molIndex);

  std::vector<uint> idx(nAtoms);
  std::vector<double> chg(nAtoms);
  for (uint i = 0; i < nAtoms; ++i) {
    idx[i] = startAtom + i;
    chg[i] = kind.AtomCharge(i);
  }

  // Removal from source box: old coords only (negate)
  const XYZArray &coords = oldMol.GetCoords();

  cachedBox = box;
  cachedSignNew = 0.0;
  cachedSignOld = -1.0;
  cachedOldCoords = coords;
  cachedAtomIndices = idx;
  cachedCharges = chg;
  cachedNAtoms = nAtoms;
  pendingUpdate = true;
  forceFullUpdate = false;

  return DeltaERecip(box, nullptr, 0.0, &coords, -1.0, idx.data(), chg.data(),
                     nAtoms, false);
}

double EwaldPME::ChangeLambdaRecip(XYZArray const &molCoords,
                                   const double lambdaOld,
                                   const double lambdaNew, const uint molIndex,
                                   const uint box) {
  if (box >= BOXES_WITH_U_NB || S_ref == nullptr || S_ref[box] == nullptr)
    return 0.0;

  MoleculeKind const &kind = mols.GetKind(molIndex);
  uint startAtom = mols.MolStart(molIndex);
  uint nAtoms = kind.NumAtoms();

  // Same coordinates, charge scales from lambdaOld to lambdaNew
  // ΔS = (lambdaNew - lambdaOld) · Σ_a q_a · B(m, r_a)
  // We reuse DeltaERecip with sign_new=(lambdaNew-lambdaOld), sign_old=0
  // by passing identical coord arrays and only new coords with net charge delta
  double dLambda = lambdaNew - lambdaOld;

  std::vector<uint> idx(nAtoms);
  std::vector<double> chg(nAtoms);
  for (uint i = 0; i < nAtoms; ++i) {
    idx[i] = startAtom + i;
    chg[i] = kind.AtomCharge(i) * dLambda;
  }

  // Inject additional charge at same positions
  cachedBox = box;
  cachedSignNew = +1.0;
  cachedSignOld = 0.0;
  cachedNewCoords = molCoords;
  cachedAtomIndices = idx;
  cachedCharges = chg;
  cachedNAtoms = nAtoms;
  pendingUpdate = true;
  forceFullUpdate = false;

  return DeltaERecip(box, &molCoords, +1.0, nullptr, 0.0, idx.data(),
                     chg.data(), nAtoms, false);
}

double
EwaldPME::MolExchangeReciprocal(const std::vector<cbmc::TrialMol> &newMol,
                                const std::vector<cbmc::TrialMol> &oldMol,
                                const std::vector<uint> &molIndexNew,
                                const std::vector<uint> &molIndexOld,
                                bool first_call) {

  double dE = 0.0;

  if (!newMol.empty() || !oldMol.empty()) {
    cachedBox = !newMol.empty() ? newMol[0].GetBox() : oldMol[0].GetBox();
    pendingUpdate = true;
    forceFullUpdate = true; // Use full refresh for complex exchange moves
  }

  // Insertions (into dest boxes)
  for (uint i = 0; i < newMol.size(); ++i) {
    uint box = newMol[i].GetBox();
    if (box >= BOXES_WITH_U_NB || S_ref == nullptr || S_ref[box] == nullptr)
      continue;
    MoleculeKind const &kind = newMol[i].GetKind();
    uint nAtoms = kind.NumAtoms();
    uint startAtom = mols.MolStart(molIndexNew[i]);
    std::vector<uint> idx(nAtoms);
    std::vector<double> chg(nAtoms);
    for (uint a = 0; a < nAtoms; ++a) {
      idx[a] = startAtom + a;
      chg[a] = kind.AtomCharge(a);
    }
    const XYZArray &coords = newMol[i].GetCoords();
    dE += DeltaERecip(box, &coords, +1.0, nullptr, 0.0, idx.data(), chg.data(),
                      nAtoms, false);
  }

  // Removals (from source boxes)
  for (uint i = 0; i < oldMol.size(); ++i) {
    uint box = oldMol[i].GetBox();
    if (box >= BOXES_WITH_U_NB || S_ref == nullptr || S_ref[box] == nullptr)
      continue;
    MoleculeKind const &kind = oldMol[i].GetKind();
    uint nAtoms = kind.NumAtoms();
    uint startAtom = mols.MolStart(molIndexOld[i]);
    std::vector<uint> idx(nAtoms);
    std::vector<double> chg(nAtoms);
    for (uint a = 0; a < nAtoms; ++a) {
      idx[a] = startAtom + a;
      chg[a] = kind.AtomCharge(a);
    }
    const XYZArray &coords = oldMol[i].GetCoords();
    dE += DeltaERecip(box, nullptr, 0.0, &coords, -1.0, idx.data(), chg.data(),
                      nAtoms, false);
  }

  return dE;
}

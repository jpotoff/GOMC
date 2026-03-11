#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

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
  S_trial = nullptr;
  greenFunc_trial = nullptr;

  for (uint b = 0; b < BOX_TOTAL; b++) {
    fwdPlan[b] = nullptr;
    tempEnergyRecip[b] = 0.0;
    tempVirialRecip[b].Zero();
  }
}

EwaldPME::~EwaldPME() {}

void EwaldPME::Init() {
  Ewald::Init(); // call base class initialization
}

void EwaldPME::AllocMem() {
  // Ewald base class configures the arrays for atoms and coordinates.
  Ewald::AllocMem();

  S_ref = new fftw_complex *[BOX_TOTAL];
  greenFunc = new double *[BOX_TOTAL];
  chargeMesh = new double *[BOX_TOTAL];
  S_trial = new fftw_complex *[BOX_TOTAL];
  greenFunc_trial = new double *[BOX_TOTAL];

  for (uint b = 0; b < BOX_TOTAL; b++) {
    S_ref[b] = nullptr;
    greenFunc[b] = nullptr;
    chargeMesh[b] = nullptr;
    fwdPlan[b] = nullptr;
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

  int k0 = (int)ceil(currentAxes.axis.Get(box).x / gridSpacing);
  int k1 = (int)ceil(currentAxes.axis.Get(box).y / gridSpacing);
  int k2 = (int)ceil(currentAxes.axis.Get(box).z / gridSpacing);

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

  // Use trial buffers
  if (S_trial[box])
    fftw_free(S_trial[box]);
  if (greenFunc_trial[box])
    delete[] greenFunc_trial[box];
  if (chargeMesh[box])
    fftw_free(chargeMesh[box]);
  if (fwdPlan[box])
    fftw_destroy_plan(fwdPlan[box]);

  S_trial[box] = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N_complex);
  greenFunc_trial[box] = new double[N_complex];
  chargeMesh[box] = (double *)fftw_malloc(sizeof(double) * N);

  fwdPlan[box] = fftw_plan_dft_r2c_3d(Kx, Ky, Kz, chargeMesh[box], S_trial[box],
                                      FFTW_ESTIMATE);

  // Temporarily swap K and GreenFunc pointers so ComputeGreenFunction and
  // SumMeshEnergy work on trial data
  for (int i = 0; i < 3; ++i) {
    int kt = K[box][i];
    K[box][i] = K_trial[box][i];
    K_trial[box][i] = kt;
  }
  double *gt = greenFunc[box];
  greenFunc[box] = greenFunc_trial[box];
  greenFunc_trial[box] = gt;

  fftw_complex *st = S_ref[box];
  S_ref[box] = S_trial[box];
  S_trial[box] = st;

  ComputeGreenFunction(box);
  ComputeChargeMesh(box, molCoords);
  fftw_execute(fwdPlan[box]);

  tempEnergyRecip[box] = SumMeshEnergy(box, S_ref[box], &tempVirialRecip[box]);

  // Swap back - UpdateRecipVec will do the final swap on acceptance
  for (int i = 0; i < 3; ++i) {
    int kt = K[box][i];
    K[box][i] = K_trial[box][i];
    K_trial[box][i] = kt;
  }
  gt = greenFunc[box];
  greenFunc[box] = greenFunc_trial[box];
  greenFunc_trial[box] = gt;

  st = S_ref[box];
  S_ref[box] = S_trial[box];
  S_trial[box] = st;
}

void EwaldPME::BoxReciprocalSums(uint box, XYZArray const &molCoords) {
  if (box >= BOXES_WITH_U_NB)
    return;
  // Use trial buffers as a scratch mesh to avoid overwriting S_ref prematurely
  // during a full refresh.
  ComputeChargeMesh(box, molCoords);
  fftw_execute(fwdPlan[box]);

  // Sum energy from S_trial (where fftw_execute wrote)
  tempEnergyRecip[box] =
      SumMeshEnergy(box, S_trial[box], &tempVirialRecip[box]);

  // Update sysPotRef and S_ref
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

  // Synchronize S_ref with S_trial
  int nk = K[box][0] * K[box][1] * (K[box][2] / 2 + 1);
  for (int i = 0; i < nk; ++i) {
    S_ref[box][i][0] = S_trial[box][i][0];
    S_ref[box][i][1] = S_trial[box][i][1];
  }
}

double EwaldPME::BoxReciprocal(uint box, bool isNewVolume) const {
  if (box >= BOXES_WITH_U_NB)
    return 0.0;
  return isNewVolume ? tempEnergyRecip[box] : sysPotRef.boxEnergy[box].recip;
}

void EwaldPME::UpdateRecipVec(uint box) {
  if (box >= BOXES_WITH_U_NB)
    return;

  // Swap K
  for (int i = 0; i < 3; ++i) {
    int kt = K[box][i];
    K[box][i] = K_trial[box][i];
    K_trial[box][i] = kt;
  }

  // Swap pointers
  fftw_complex *st = S_ref[box];
  S_ref[box] = S_trial[box];
  S_trial[box] = st;

  double *gt = greenFunc[box];
  greenFunc[box] = greenFunc_trial[box];
  greenFunc_trial[box] = gt;

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
    XYZ s = currentAxes.TransformUnSlant(r, box);
    double sx = s.x / currentAxes.axis.Get(box).x;
    double sy = s.y / currentAxes.axis.Get(box).y;
    double sz = s.z / currentAxes.axis.Get(box).z;

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

  double volume = currentAxes.volume[box];
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

    double kx_cart = 2.0 * M_PI * kx_int / currentAxes.axis.Get(box).x;
    double ky_cart = 2.0 * M_PI * ky_int / currentAxes.axis.Get(box).y;
    double kz_cart = 2.0 * M_PI * iz / currentAxes.axis.Get(box).z;
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
      double kx_cart = 2.0 * M_PI * kx_int / currentAxes.axis.Get(box).x;
      double ky_cart = 2.0 * M_PI * ky_int / currentAxes.axis.Get(box).y;
      double kz_cart = 2.0 * M_PI * iz / currentAxes.axis.Get(box).z;
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

// ---------------------------------------------------------------------------
// Helper: accumulate ΔS(m) for a set of atoms into a pre-allocated complex
// array dS (same layout as S_ref: size (Kx*Ky*(Kz/2+1))).
// sign   = +1 (insert) or -1 (remove).
// ---------------------------------------------------------------------------
void EwaldPME::ComputeMolDeltaS(uint box, const XYZArray &coords,
                                const uint *atomIdx, const double *charges,
                                uint nAtoms, double sign,
                                fftw_complex *dS) const {
  int Kx = K[box][0], Ky = K[box][1], Kz = K[box][2];
  int halfKz = Kz / 2 + 1;

#ifdef _OPENMP
#pragma omp parallel for default(shared)
#endif
  for (int a = 0; a < (int)nAtoms; ++a) {
    AddAtomToDeltaS(box, atomIdx[a], sign * charges[a], coords, dS);
  }
}

void EwaldPME::AddAtomToDeltaS(uint box, uint atomIdx, double charge,
                               const XYZArray &coords, fftw_complex *dS) const {
  int Kx = K[box][0], Ky = K[box][1], Kz = K[box][2];
  int halfKz = Kz / 2 + 1;

  XYZ r = coords.Get(atomIdx);
  XYZ s = currentAxes.TransformUnSlant(r, box);
  double sx = s.x / currentAxes.axis.Get(box).x;
  double sy = s.y / currentAxes.axis.Get(box).y;
  double sz = s.z / currentAxes.axis.Get(box).z;
  sx -= std::floor(sx);
  sy -= std::floor(sy);
  sz -= std::floor(sz);

  double ux = sx * Kx, uy = sy * Ky, uz = sz * Kz;
  double bs_x[20], bs_y[20], bs_z[20];
  bspline::EvalAll(pmeOrder, ux - std::floor(ux), bs_x);
  bspline::EvalAll(pmeOrder, uy - std::floor(uy), bs_y);
  bspline::EvalAll(pmeOrder, uz - std::floor(uz), bs_z);

  int nx = (int)std::floor(ux) - pmeOrder + 1;
  int ny = (int)std::floor(uy) - pmeOrder + 1;
  int nz = (int)std::floor(uz) - pmeOrder + 1;

  // Since this is incremental and only for 1 atom at a time in trials,
  // we could use separability here too, but for one atom P^3 is small.
  // However, dS is SHARED, so we need atomic updates.
  for (int ix = 0; ix < pmeOrder; ++ix) {
    int gx = ((nx + ix) % Kx + Kx) % Kx;
    double wx = bs_x[pmeOrder - 1 - ix];
    for (int iy = 0; iy < pmeOrder; ++iy) {
      int gy = ((ny + iy) % Ky + Ky) % Ky;
      double wy = wx * bs_y[pmeOrder - 1 - iy];
      for (int iz = 0; iz < pmeOrder; ++iz) {
        int gz = ((nz + iz) % Kz + Kz) % Kz;
        double w = charge * wy * bs_z[pmeOrder - 1 - iz];

        for (int kmx = 0; kmx < Kx; ++kmx) {
          int kxi = (kmx <= Kx / 2) ? kmx : kmx - Kx;
          double phx = -2.0 * M_PI * kxi * gx / Kx;
          for (int kmy = 0; kmy < Ky; ++kmy) {
            int kyi = (kmy <= Ky / 2) ? kmy : kmy - Ky;
            double phy = phx - 2.0 * M_PI * kyi * gy / Ky;
            for (int kmz = 0; kmz < halfKz; ++kmz) {
              double phase = phy - 2.0 * M_PI * kmz * gz / Kz;
              int kidx = (kmx * Ky + kmy) * halfKz + kmz;
#ifdef _OPENMP
#pragma omp atomic update
#endif
              dS[kidx][0] += w * std::cos(phase);
#ifdef _OPENMP
#pragma omp atomic update
#endif
              dS[kidx][1] += w * std::sin(phase);
            }
          }
        }
      }
    }
  }
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

  int Kx = K[box][0], Ky = K[box][1], Kz = K[box][2];
  int halfKz = Kz / 2 + 1;
  int nk = Kx * Ky * halfKz;

  // Allocate ΔS buffer (zero-initialized)
  std::vector<fftw_complex> dS(nk);
  for (int i = 0; i < nk; ++i) {
    dS[i][0] = dS[i][1] = 0.0;
  }

  // Accumulate contributions from new and old positions
  if (newCoords)
    ComputeMolDeltaS(box, *newCoords, atomIndices, charges, nAtoms, sign_new,
                     dS.data());
  if (oldCoords)
    ComputeMolDeltaS(box, *oldCoords, atomIndices, charges, nAtoms, sign_old,
                     dS.data());

  double dE = 0.0;
  double *G = greenFunc[box];
  fftw_complex *stref = S_ref[box];

#ifdef _OPENMP
#pragma omp parallel for default(shared) reduction(+ : dE)
#endif
  for (int i = 0; i < nk; ++i) {
    // Reconstruct kmz for weight calculation
    int kmz = i % halfKz;
    // Reconstruct kmx, kmy to check for origin skip
    int kmy_kmz = i / halfKz;
    int kmx = kmy_kmz / Ky;
    int kmy = kmy_kmz % Ky;

    if (kmx == 0 && kmy == 0 && kmz == 0)
      continue;

    double dre = dS[i][0], dim = dS[i][1];
    double dSsq = dre * dre + dim * dim;
    double cross = stref[i][0] * dre + stref[i][1] * dim;

    double weight = (kmz == 0 || (Kz % 2 == 0 && kmz == Kz / 2)) ? 1.0 : 2.0;
    dE += weight * G[i] * (cross + 0.5 * dSsq);
  }

  // Optionally commit the update to S_ref (on move acceptance)
  if (updateSRef) {
#ifdef _OPENMP
#pragma omp parallel for default(shared)
#endif
    for (int i = 0; i < nk; ++i) {
      stref[i][0] += dS[i][0];
      stref[i][1] += dS[i][1];
    }
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
  return DeltaERecip(box, &molCoords, +1.0, &currentCoords, -1.0, idx.data(),
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

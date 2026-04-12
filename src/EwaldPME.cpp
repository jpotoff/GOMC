#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include "BSpline.h"
#include "BoxDimensions.h"
#include "EnergyTypes.h"
#include "EwaldPME.h"
#include "NumLib.h"
#include "StaticVals.h"
#include "System.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

EwaldPME::EwaldPME(StaticVals &stat, System &sys) : Ewald(stat, sys) {
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

  memset(fwdPlan, 0, sizeof(fwdPlan));
  memset(bwdPlan, 0, sizeof(bwdPlan));
  memset(scratchPlan, 0, sizeof(scratchPlan));

  for (uint b = 0; b < BOX_TOTAL; b++) {
    K[b][0] = K[b][1] = K[b][2] = 0;
    K_trial[b][0] = K_trial[b][1] = K_trial[b][2] = 0;
    K_allocated[b][0] = K_allocated[b][1] = K_allocated[b][2] = 0;
    tempEnergyRecip[b] = 0.0;
  }

  pendingUpdate = false;
  forceFullUpdate = false;
  cachedBox = 0;
  cachedSignNew = 0.0;
  cachedSignOld = 0.0;
  cachedNAtoms = 0;
}

EwaldPME::~EwaldPME() {
  for (uint b = 0; b < BOX_TOTAL; b++) {
    if (S_ref && S_ref[b]) fftw_free(S_ref[b]);
    if (greenFunc && greenFunc[b]) delete[] greenFunc[b];
    if (chargeMesh && chargeMesh[b]) delete[] chargeMesh[b];
    if (potentialMesh && potentialMesh[b]) delete[] potentialMesh[b];
    if (scratchMesh && scratchMesh[b]) delete[] scratchMesh[b];
    if (S_delta && S_delta[b]) fftw_free(S_delta[b]);
    if (S_trial && S_trial[b]) fftw_free(S_trial[b]);
    if (greenFunc_trial && greenFunc_trial[b]) delete[] greenFunc_trial[b];
    if (fwdPlan[b]) fftw_destroy_plan(fwdPlan[b]);
    if (bwdPlan[b]) fftw_destroy_plan(bwdPlan[b]);
    if (scratchPlan[b]) fftw_destroy_plan(scratchPlan[b]);
  }
  delete[] S_ref;
  delete[] greenFunc;
  delete[] chargeMesh;
  delete[] potentialMesh;
  delete[] scratchMesh;
  delete[] S_delta;
  delete[] S_trial;
  delete[] greenFunc_trial;
}

void EwaldPME::Init() {
  Ewald::Init();
}

void EwaldPME::AllocMem() {
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
    S_trial[b] = nullptr;
    greenFunc_trial[b] = nullptr;
    fwdPlan[b] = nullptr;
    bwdPlan[b] = nullptr;
    scratchPlan[b] = nullptr;
  }
}

void EwaldPME::BoxReciprocalSetup(uint box, XYZArray const &molCoords) {
  if (box >= BOXES_WITH_U_NB)
    return;

  int Kx = K_trial[box][0], Ky = K_trial[box][1], Kz = K_trial[box][2];
  bool kChanged = (Kx != K_allocated[box][0] || Ky != K_allocated[box][1] || Kz != K_allocated[box][2]);

  if (kChanged || fwdPlan[box] == nullptr) {
    if (fwdPlan[box]) fftw_destroy_plan(fwdPlan[box]);
    if (bwdPlan[box]) fftw_destroy_plan(bwdPlan[box]);
    if (scratchPlan[box]) fftw_destroy_plan(scratchPlan[box]);

    int N = Kx * Ky * Kz;
    int halfKz = Kz / 2 + 1;
    int N_complex = Kx * Ky * halfKz;

    if (chargeMesh[box]) delete[] chargeMesh[box];
    chargeMesh[box] = new double[N];
    if (potentialMesh[box]) delete[] potentialMesh[box];
    potentialMesh[box] = new double[N];
    if (scratchMesh[box]) delete[] scratchMesh[box];
    scratchMesh[box] = new double[N];

    if (S_ref[box]) fftw_free(S_ref[box]);
    S_ref[box] = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N_complex);
    if (S_delta[box]) fftw_free(S_delta[box]);
    S_delta[box] = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N_complex);
    if (S_trial[box]) fftw_free(S_trial[box]);
    S_trial[box] = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N_complex);

    if (greenFunc[box]) delete[] greenFunc[box];
    greenFunc[box] = new double[N_complex];
    if (greenFunc_trial[box]) delete[] greenFunc_trial[box];
    greenFunc_trial[box] = new double[N_complex];

    fwdPlan[box] = fftw_plan_dft_r2c_3d(Kx, Ky, Kz, chargeMesh[box],
                                        S_trial[box], FFTW_ESTIMATE);
    bwdPlan[box] = fftw_plan_dft_c2r_3d(Kx, Ky, Kz, S_trial[box],
                                        potentialMesh[box], FFTW_ESTIMATE);
    scratchPlan[box] = fftw_plan_dft_r2c_3d(Kx, Ky, Kz, scratchMesh[box],
                                            S_delta[box], FFTW_ESTIMATE);

    K_allocated[box][0] = Kx; K_allocated[box][1] = Ky; K_allocated[box][2] = Kz;

    // Initialize the committed greenFunc with valid data so it doesn't contain garbage zeros
    UpdateGreenFunction(box, trialAxes[box], greenFunc[box]);
  }

  // Always evaluate the trial green function for the new trial dimensions
  UpdateGreenFunction(box, trialAxes[box], greenFunc_trial[box]);

  int N = Kx * Ky * Kz;
  fill(chargeMesh[box], chargeMesh[box] + N, 0.0);

  MoleculeLookup::box_iterator thisMol = molLookup.BoxBegin(box);
  MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);
  while (thisMol != end) {
    uint m = *thisMol;
    double lambda = GetLambdaCoef(m, box);
    const MoleculeKind &kind = mols.GetKind(m);
    uint start = mols.MolStart(m);
    for (uint a = 0; a < kind.NumAtoms(); ++a) {
      double charge = kind.AtomCharge(a) * lambda;
      XYZ r = molCoords.Get(start + a);
      XYZ s = trialAxes[box].TransformUnSlant(r, box);
      double sx = s.x / trialAxes[box].axis.Get(box).x;
      double sy = s.y / trialAxes[box].axis.Get(box).y;
      double sz = s.z / trialAxes[box].axis.Get(box).z;
      sx -= floor(sx); sy -= floor(sy); sz -= floor(sz);
      double ux = sx * Kx, uy = sy * Ky, uz = sz * Kz;
      double bs_x[20], bs_y[20], bs_z[20];
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
            chargeMesh[box][(gx * Ky + gy) * Kz + gz] += charge * wy * bs_z[pmeOrder - 1 - iz];
          }
        }
      }
    }
    ++thisMol;
  }

  fftw_execute(fwdPlan[box]);
  tempEnergyRecip[box] = SumMeshEnergy(box, S_trial[box]);
}

void EwaldPME::UpdateGreenFunction(uint box, const BoxDimensions &axes, double *gf_out) {
  int Kx = K_trial[box][0], Ky = K_trial[box][1], Kz = K_trial[box][2];
  int halfKz = Kz / 2 + 1;
  double Lx = axes.axis.Get(box).x, Ly = axes.axis.Get(box).y, Lz = axes.axis.Get(box).z;
  double alpha = ff.alpha[box], pre = 1.0 / (4.0 * alpha * alpha), vol = Lx * Ly * Lz;
  vector<double> bx = bspline::BSplineModuli(Kx, pmeOrder);
  vector<double> by = bspline::BSplineModuli(Ky, pmeOrder);
  vector<double> bz = bspline::BSplineModuli(Kz, pmeOrder);
  for (int ix = 0; ix < Kx; ++ix) {
    int kx_int = (ix <= Kx / 2) ? ix : ix - Kx;
    double kx = 2.0 * M_PI * kx_int / Lx;
    for (int iy = 0; iy < Ky; ++iy) {
      int ky_int = (iy <= Ky / 2) ? iy : iy - Ky;
      double ky = 2.0 * M_PI * ky_int / Ly;
      for (int iz = 0; iz < halfKz; ++iz) {
        double kz = 2.0 * M_PI * iz / Lz;
        double kSq = kx * kx + ky * ky + kz * kz;
        int idx = (ix * Ky + iy) * halfKz + iz;
        if (kSq == 0) gf_out[idx] = 0.0;
        else gf_out[idx] = num::qqFact * (4.0 * M_PI / (vol * kSq)) * exp(-pre * kSq) * bx[ix] * by[iy] * bz[iz];
      }
    }
  }
}

double EwaldPME::SumMeshEnergy(uint box, fftw_complex *S, Virial *virial,
                               bool useTrial) const {
  int Kx = useTrial ? K_trial[box][0] : K[box][0];
  int Ky = useTrial ? K_trial[box][1] : K[box][1];
  int Kz = useTrial ? K_trial[box][2] : K[box][2];
  const double *gf = useTrial ? greenFunc_trial[box] : greenFunc[box];
  const BoxDimensions &axes = useTrial ? trialAxes[box] : currentAxes;
  int halfKz = Kz / 2 + 1;
  double energy = 0.0, wT11 = 0.0, wT22 = 0.0, wT33 = 0.0;
  double constVal = 1.0 / (4.0 * ff.alpha[box] * ff.alpha[box]);
  for (int i = 0; i < Kx * Ky * halfKz; ++i) {
    int iz = i % halfKz, iy_iz = i / halfKz, ix = iy_iz / Ky, iy = iy_iz % Ky;
    if (ix == 0 && iy == 0 && iz == 0) continue;
    double S_sq = S[i][0] * S[i][0] + S[i][1] * S[i][1];
    double weight = (iz == 0 || (Kz % 2 == 0 && iz == Kz / 2)) ? 1.0 : 2.0;
    double G = gf[i], term = 0.5 * weight * G * S_sq;
    energy += term;
    if (virial) {
      int kx_int = (ix <= Kx / 2) ? ix : ix - Kx, ky_int = (iy <= Ky / 2) ? iy : iy - Ky;
      double kx_cart = 2.0 * M_PI * kx_int / axes.axis.Get(box).x;
      double ky_cart = 2.0 * M_PI * ky_int / axes.axis.Get(box).y;
      double kz_cart = 2.0 * M_PI * iz / axes.axis.Get(box).z;
      double kSq = kx_cart * kx_cart + ky_cart * ky_cart + kz_cart * kz_cart;
      double common = 2.0 * (constVal + 1.0 / kSq);
      wT11 += term * (1.0 - common * kx_cart * kx_cart);
      wT22 += term * (1.0 - common * ky_cart * ky_cart);
      wT33 += term * (1.0 - common * kz_cart * kz_cart);
    }
  }
  if (virial) {
    virial->recipTens[0][0] = wT11; virial->recipTens[1][1] = wT22; virial->recipTens[2][2] = wT33;
    virial->recip = wT11 + wT22 + wT33;
  }
  return energy;
}

void EwaldPME::UpdatePotentialMesh(uint box) {
  if (box >= BOXES_WITH_U_NB || S_ref[box] == nullptr) return;
  int nk = K[box][0] * K[box][1] * (K[box][2] / 2 + 1);
  for (int i = 0; i < nk; ++i) {
    S_trial[box][i][0] = greenFunc[box][i] * S_ref[box][i][0];
    S_trial[box][i][1] = greenFunc[box][i] * S_ref[box][i][1];
  }
  fftw_execute(bwdPlan[box]);
}

void EwaldPME::UpdateAtomInMesh(uint box, const double *charges, uint nAtoms,
                                 const XYZArray &coords, double sign) {
  // Use committed K dims (not K_trial) for consistency with ComputeDeltaSsq
  int Kx = K[box][0], Ky = K[box][1], Kz = K[box][2];
  for (uint i = 0; i < nAtoms; ++i) {
    double charge = charges[i] * sign;
    XYZ r = coords.Get(i); XYZ s = currentAxes.TransformUnSlant(r, box);
    double sx = s.x / currentAxes.axis.Get(box).x, sy = s.y / currentAxes.axis.Get(box).y, sz = s.z / currentAxes.axis.Get(box).z;
    sx -= floor(sx); sy -= floor(sy); sz -= floor(sz);
    double ux = sx * Kx, uy = sy * Ky, uz = sz * Kz;
    double bs_x[20], bs_y[20], bs_z[20];
    bspline::EvalAll(pmeOrder, ux - floor(ux), bs_x); bspline::EvalAll(pmeOrder, uy - floor(uy), bs_y); bspline::EvalAll(pmeOrder, uz - floor(uz), bs_z);
    int nx = (int)floor(ux) - pmeOrder + 1, ny = (int)floor(uy) - pmeOrder + 1, nz = (int)floor(uz) - pmeOrder + 1;
    for (int ix = 0; ix < pmeOrder; ++ix) {
      int gx = ((nx + ix) % Kx + Kx) % Kx; double wx = bs_x[pmeOrder - 1 - ix];
      for (int iy = 0; iy < pmeOrder; ++iy) {
        int gy = ((ny + iy) % Ky + Ky) % Ky; double wy = wx * bs_y[pmeOrder - 1 - iy];
        for (int iz = 0; iz < pmeOrder; ++iz) {
          int gz = ((nz + iz) % Kz + Kz) % Kz;
          chargeMesh[box][(gx * Ky + gy) * Kz + gz] += charge * wy * bs_z[pmeOrder - 1 - iz];
        }
      }
    }
  }
}

double EwaldPME::InterpolatePotential(uint box, const XYZ &r) const {
  int Kx = K[box][0], Ky = K[box][1], Kz = K[box][2];
  XYZ s = currentAxes.TransformUnSlant(r, box);
  double sx = s.x / currentAxes.axis.Get(box).x, sy = s.y / currentAxes.axis.Get(box).y, sz = s.z / currentAxes.axis.Get(box).z;
  sx -= floor(sx); sy -= floor(sy); sz -= floor(sz);
  double ux = sx * Kx, uy = sy * Ky, uz = sz * Kz;
  double bs_x[20], bs_y[20], bs_z[20];
  bspline::EvalAll(pmeOrder, ux - floor(ux), bs_x); bspline::EvalAll(pmeOrder, uy - floor(uy), bs_y); bspline::EvalAll(pmeOrder, uz - floor(uz), bs_z);
  int nx = (int)floor(ux) - pmeOrder + 1, ny = (int)floor(uy) - pmeOrder + 1, nz = (int)floor(uz) - pmeOrder + 1;
  double pot = 0.0;
  for (int ix = 0; ix < pmeOrder; ++ix) {
    int gx = ((nx + ix) % Kx + Kx) % Kx; double wx = bs_x[pmeOrder - 1 - ix];
    for (int iy = 0; iy < pmeOrder; ++iy) {
      int gy = ((ny + iy) % Ky + Ky) % Ky; double wy = wx * bs_y[pmeOrder - 1 - iy];
      for (int iz = 0; iz < pmeOrder; ++iz) {
        int gz = ((nz + iz) % Kz + Kz) % Kz;
        pot += wy * bs_z[pmeOrder - 1 - iz] * potentialMesh[box][(gx * Ky + gy) * Kz + gz];
      }
    }
  }
  return pot;
}

double EwaldPME::ComputeDeltaSsq(uint box, const XYZArray *newCoords,
                                 double sign_new, const XYZArray *oldCoords,
                                 double sign_old, const uint *atomIndices,
                                 const double *charges, uint nAtoms) const {
  if (nAtoms == 0) return 0.0;
  int Kx = K[box][0], Ky = K[box][1], Kz = K[box][2];
  fill(scratchMesh[box], scratchMesh[box] + Kx * Ky * Kz, 0.0);
  auto add = [&](const XYZArray *coords, double sign) {
    if (!coords) return;
    double bs_x[20], bs_y[20], bs_z[20];
    for (uint i = 0; i < nAtoms; ++i) {
      if (particleHasNoCharge[atomIndices[i]]) continue;
      double chg = charges[i] * sign; 
      XYZ r = coords->Get(i); XYZ s = currentAxes.TransformUnSlant(r, box);
      double sx = s.x / currentAxes.axis.Get(box).x, sy = s.y / currentAxes.axis.Get(box).y, sz = s.z / currentAxes.axis.Get(box).z;
      sx -= floor(sx); sy -= floor(sy); sz -= floor(sz);
      double ux = sx * Kx, uy = sy * Ky, uz = sz * Kz;
      bspline::EvalAll(pmeOrder, ux - floor(ux), bs_x); bspline::EvalAll(pmeOrder, uy - floor(uy), bs_y); bspline::EvalAll(pmeOrder, uz - floor(uz), bs_z);
      int nx = (int)floor(ux) - pmeOrder + 1, ny = (int)floor(uy) - pmeOrder + 1, nz = (int)floor(uz) - pmeOrder + 1;
      for (int ix = 0; ix < pmeOrder; ++ix) {
        int gx = ((nx + ix) % Kx + Kx) % Kx; double wx = bs_x[pmeOrder - 1 - ix];
        for (int iy = 0; iy < pmeOrder; ++iy) {
          int gy = ((ny + iy) % Ky + Ky) % Ky; double wy = wx * bs_y[pmeOrder - 1 - iy];
          for (int iz = 0; iz < pmeOrder; ++iz) {
            int gz = ((nz + iz) % Kz + Kz) % Kz;
            scratchMesh[box][(gx * Ky + gy) * Kz + gz] += chg * wy * bs_z[pmeOrder - 1 - iz];
          }
        }
      }
    }
  };
  add(newCoords, sign_new); add(oldCoords, sign_old);
  fftw_execute(scratchPlan[box]);
  return SumMeshEnergy(box, S_delta[box], nullptr, false);
}

double EwaldPME::DeltaERecip(uint box, const XYZArray *newCoords,
                             double sign_new, const XYZArray *oldCoords,
                             double sign_old, const uint *atomIndices,
                             const double *charges, uint nAtoms,
                             bool updateSRef) {
  if (box >= BOXES_WITH_U_NB || S_ref[box] == nullptr) return 0.0;
  double dE = 0.0;
  if (newCoords) for (uint a = 0; a < nAtoms; ++a) if (!particleHasNoCharge[atomIndices[a]]) dE += sign_new * charges[a] * InterpolatePotential(box, newCoords->Get(a));
  if (oldCoords) for (uint a = 0; a < nAtoms; ++a) if (!particleHasNoCharge[atomIndices[a]]) dE += sign_old * charges[a] * InterpolatePotential(box, oldCoords->Get(a));
  dE += ComputeDeltaSsq(box, newCoords, sign_new, oldCoords, sign_old, atomIndices, charges, nAtoms);
  if (updateSRef) {
    if (oldCoords) UpdateAtomInMesh(box, charges, nAtoms, *oldCoords, sign_old);
    if (newCoords) UpdateAtomInMesh(box, charges, nAtoms, *newCoords, sign_new);
    fftw_execute(fwdPlan[box]);
    int nk = K[box][0] * K[box][1] * (K[box][2] / 2 + 1);
    memcpy(S_ref[box], S_trial[box], sizeof(fftw_complex) * nk);
    UpdatePotentialMesh(box);
    return SumMeshEnergy(box, S_ref[box], nullptr, false);
  }
  return dE;
}

void EwaldPME::BoxReciprocalSums(uint box, XYZArray const &molCoords) {
  if (box >= BOXES_WITH_U_NB) return;
  RecipInit(box, currentAxes);
  BoxReciprocalSetup(box, molCoords);
}

double EwaldPME::BoxReciprocal(uint box, bool isNewVolume) const {
  if (box >= BOXES_WITH_U_NB) return 0.0;
  return isNewVolume ? tempEnergyRecip[box] : sysPotRef.boxEnergy[box].recip;
}

void EwaldPME::RecipInit(uint box, BoxDimensions const &axes) {
  Ewald::RecipInit(box, axes);
  trialAxes[box] = axes;

  K_trial[box][0] = (int)round(axes.axis.Get(box).x / ff.pmeGridSpacing);
  K_trial[box][1] = (int)round(axes.axis.Get(box).y / ff.pmeGridSpacing);
  K_trial[box][2] = (int)round(axes.axis.Get(box).z / ff.pmeGridSpacing);

  if (K_trial[box][0] < 1) K_trial[box][0] = 1;
  if (K_trial[box][1] < 1) K_trial[box][1] = 1;
  if (K_trial[box][2] < 1) K_trial[box][2] = 1;

  auto nextFFTW = [](int d) {
    while (d % 2 != 0 && d % 3 != 0 && d % 5 != 0) d++;
    return d;
  };

  K_trial[box][0] = nextFFTW(K_trial[box][0]);
  K_trial[box][1] = nextFFTW(K_trial[box][1]);
  K_trial[box][2] = nextFFTW(K_trial[box][2]);
}

void EwaldPME::UpdateRecipVec(uint box) {
  if (box >= BOXES_WITH_U_NB) return;
  K[box][0] = K_trial[box][0]; K[box][1] = K_trial[box][1]; K[box][2] = K_trial[box][2];
  int nk = K[box][0] * K[box][1] * (K[box][2] / 2 + 1);
  memcpy(S_ref[box], S_trial[box], sizeof(fftw_complex) * nk);
  memcpy(greenFunc[box], greenFunc_trial[box], sizeof(double) * nk);
  const_cast<SystemPotential &>(sysPotRef).boxEnergy[box].recip = tempEnergyRecip[box];
  const_cast<SystemPotential &>(sysPotRef).boxVirial[box].recip = tempVirialRecip[box].recip;
  for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) const_cast<SystemPotential &>(sysPotRef).boxVirial[box].recipTens[i][j] = tempVirialRecip[box].recipTens[i][j];
  trialAxes[box] = currentAxes;
  UpdatePotentialMesh(box);
}

void EwaldPME::UpdateRecip(uint box) {
  if (box >= BOXES_WITH_U_NB || !pendingUpdate || box != cachedBox) return;
  if (forceFullUpdate) {
    // Ensure K_trial matches committed K before rebuilding
    K_trial[box][0] = K[box][0];
    K_trial[box][1] = K[box][1];
    K_trial[box][2] = K[box][2];
    trialAxes[box] = currentAxes;
    BoxReciprocalSetup(box, currentCoords);
    int nk = K[box][0] * K[box][1] * (K[box][2] / 2 + 1);
    memcpy(S_ref[box], S_trial[box], sizeof(fftw_complex) * nk);
    UpdatePotentialMesh(box);
    // Store the exact energy computed from the full mesh rebuild
    const_cast<SystemPotential &>(sysPotRef).boxEnergy[box].recip =
        SumMeshEnergy(box, S_ref[box], nullptr, false);
  } else {
    // DeltaERecip with updateSRef=true returns the exact total recip energy
    double exactEnergy = DeltaERecip(box, (cachedSignNew!=0.0?&cachedNewCoords:nullptr), cachedSignNew, (cachedSignOld!=0.0?&cachedOldCoords:nullptr), cachedSignOld, cachedAtomIndices.data(), cachedCharges.data(), cachedNAtoms, true);
    // Store exact energy, overwriting any incremental += dE from the caller
    const_cast<SystemPotential &>(sysPotRef).boxEnergy[box].recip = exactEnergy;
  }
  pendingUpdate = false; forceFullUpdate = false;
}

void EwaldPME::Maintain(const ulong step) {
  if (refreshInterval == 0 || (step + 1) % refreshInterval != 0) return;
  for (uint b = 0; b < BOXES_WITH_U_NB; b++) {
    BoxReciprocalSums(b, currentCoords);
    UpdateRecipVec(b);
  }
}

void EwaldPME::exgMolCache() {
  for (uint b = 0; b < BOXES_WITH_U_NB; ++b) {
    if (S_ref && S_ref[b]) {
      K_trial[b][0] = K[b][0];
      K_trial[b][1] = K[b][1];
      K_trial[b][2] = K[b][2];
      trialAxes[b] = currentAxes;

      if (K_allocated[b][0] != K[b][0] || K_allocated[b][1] != K[b][1] || K_allocated[b][2] != K[b][2]) {
        if (fwdPlan[b]) { fftw_destroy_plan(fwdPlan[b]); fwdPlan[b] = nullptr; }
        if (bwdPlan[b]) { fftw_destroy_plan(bwdPlan[b]); bwdPlan[b] = nullptr; }
        if (scratchPlan[b]) { fftw_destroy_plan(scratchPlan[b]); scratchPlan[b] = nullptr; }

        BoxReciprocalSetup(b, currentCoords);
        int nk = K[b][0] * K[b][1] * (K[b][2] / 2 + 1);
        memcpy(S_ref[b], S_trial[b], sizeof(fftw_complex) * nk);
        UpdatePotentialMesh(b);
      }
    }
  }
}

void EwaldPME::RestoreMol(int molIndex) {
  pendingUpdate = false; forceFullUpdate = false;
}

// PME uses mesh-based S_ref/S_trial;
// the base Ewald's sumR/sumI arrays are not relevant.
void EwaldPME::CopyRecip(uint box) { return; }
void EwaldPME::backupMolCache() { return; }


void EwaldPME::BoxForceReciprocal(XYZArray const &molCoords, XYZArray &atomForceRec, XYZArray &molForceRec, uint box) {
  if (box >= BOXES_WITH_U_NB || S_ref[box] == nullptr) return;
  UpdatePotentialMesh(box);
  int Kx = K[box][0], Ky = K[box][1], Kz = K[box][2];
  double Lx = currentAxes.axis.Get(box).x, Ly = currentAxes.axis.Get(box).y, Lz = currentAxes.axis.Get(box).z;
  double factorX = (double)Kx / Lx, factorY = (double)Ky / Ly, factorZ = (double)Kz / Lz;
  MoleculeLookup::box_iterator thisMol = molLookup.BoxBegin(box);
  MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);
  while (thisMol != end) {
    uint m = *thisMol;
    const MoleculeKind &kind = mols.GetKind(m);
    uint start = mols.MolStart(m);
    double lambda = GetLambdaCoef(m, box);
    XYZ f_mol;
    for (uint a = 0; a < kind.NumAtoms(); ++a) {
      uint globalIdx = start + a;
      if (particleHasNoCharge[globalIdx]) { atomForceRec.Set(globalIdx, 0,0,0); continue; }
      double charge = kind.AtomCharge(a) * lambda;
      XYZ r = molCoords.Get(globalIdx); XYZ s = currentAxes.TransformUnSlant(r, box);
      double sx = s.x / Lx, sy = s.y / Ly, sz = s.z / Lz;
      sx -= floor(sx); sy -= floor(sy); sz -= floor(sz);
      double ux = sx * Kx, uy = sy * Ky, uz = sz * Kz;
      double bs_x[20], bs_y[20], bs_z[20], dbs_x[20], dbs_y[20], dbs_z[20];
      bspline::EvalAll(pmeOrder, ux - floor(ux), bs_x); bspline::EvalAllDeriv(pmeOrder, ux - floor(ux), dbs_x);
      bspline::EvalAll(pmeOrder, uy - floor(uy), bs_y); bspline::EvalAllDeriv(pmeOrder, uy - floor(uy), dbs_y);
      bspline::EvalAll(pmeOrder, uz - floor(uz), bs_z); bspline::EvalAllDeriv(pmeOrder, uz - floor(uz), dbs_z);
      int nx = (int)floor(ux) - pmeOrder + 1, ny = (int)floor(uy) - pmeOrder + 1, nz = (int)floor(uz) - pmeOrder + 1;
      double dVdux = 0, dVduy = 0, dVduz = 0;
      for (int ix = 0; ix < pmeOrder; ++ix) {
        int gx = ((nx + ix) % Kx + Kx) % Kx; double wx = bs_x[pmeOrder-1-ix], dwx = dbs_x[pmeOrder-1-ix];
        for (int iy = 0; iy < pmeOrder; ++iy) {
          int gy = ((ny + iy) % Ky + Ky) % Ky; double wy = bs_y[pmeOrder-1-iy], dwy = dbs_y[pmeOrder-1-iy];
          for (int iz = 0; iz < pmeOrder; ++iz) {
            int gz = ((nz + iz) % Kz + Kz) % Kz; double wz = bs_z[pmeOrder-1-iz], dwz = dbs_z[pmeOrder-1-iz];
            double pot = potentialMesh[box][(gx * Ky + gy) * Kz + gz];
            dVdux += dwx * wy * wz * pot; dVduy += wx * dwy * wz * pot; dVduz += wx * wy * dwz * pot;
          }
        }
      }
      XYZ f_atom(-charge * dVdux * factorX, -charge * dVduy * factorY, -charge * dVduz * factorZ);
      atomForceRec.Set(globalIdx, f_atom); f_mol += f_atom;
    }
    molForceRec.Set(m, f_mol); ++thisMol;
  }
}

double EwaldPME::MolReciprocal(XYZArray const &molCoords, const uint molIndex, const uint box) {
  uint startAtom = mols.MolStart(molIndex), length = mols.MolLength(molIndex);
  vector<uint> atomIndices(length); vector<double> charges(length);
  for (uint i = 0; i < length; ++i) { atomIndices[i] = startAtom + i; charges[i] = particleCharge[startAtom + i]; }
  // Extract old coords first so we can pass the correctly-sized array
  cachedOldCoords.Uninit();
  cachedOldCoords.Init(length);
  currentCoords.CopyRange(cachedOldCoords, startAtom, 0, length);

  double dE = DeltaERecip(box, &molCoords, 1.0, &cachedOldCoords, -1.0, atomIndices.data(), charges.data(), length, false);

  // Cache remaining move data so UpdateRecip can apply it on acceptance
  pendingUpdate = true;
  forceFullUpdate = false;
  cachedBox = box;
  cachedNewCoords = molCoords;
  cachedAtomIndices.assign(atomIndices.begin(), atomIndices.end());
  cachedCharges.assign(charges.begin(), charges.end());
  cachedSignNew = 1.0;
  cachedSignOld = -1.0;
  cachedNAtoms = length;

  return dE;
}

double EwaldPME::SwapDestRecip(const cbmc::TrialMol &newMol, const uint box, const int molIndex) {
  uint length = newMol.GetKind().NumAtoms(); vector<uint> atomIndices(length); vector<double> charges(length);
  uint startAtom = mols.MolStart(molIndex);
  for (uint i = 0; i < length; ++i) { atomIndices[i] = startAtom + i; charges[i] = newMol.GetKind().AtomCharge(i); }
  double dE = DeltaERecip(box, &newMol.GetCoords(), 1.0, nullptr, 0.0, atomIndices.data(), charges.data(), length, false);

  pendingUpdate = true;
  forceFullUpdate = false;
  cachedBox = box;
  cachedNewCoords = newMol.GetCoords();
  cachedOldCoords.Uninit();
  cachedOldCoords.Init(0);
  cachedAtomIndices.assign(atomIndices.begin(), atomIndices.end());
  cachedCharges.assign(charges.begin(), charges.end());
  cachedSignNew = 1.0;
  cachedSignOld = 0.0;
  cachedNAtoms = length;

  return dE;
}

double EwaldPME::SwapSourceRecip(const cbmc::TrialMol &oldMol, const uint box, const int molIndex) {
  uint length = oldMol.GetKind().NumAtoms(); vector<uint> atomIndices(length); vector<double> charges(length);
  uint startAtom = mols.MolStart(molIndex);
  for (uint i = 0; i < length; ++i) { atomIndices[i] = startAtom + i; charges[i] = oldMol.GetKind().AtomCharge(i); }
  double dE = DeltaERecip(box, nullptr, 0.0, &oldMol.GetCoords(), -1.0, atomIndices.data(), charges.data(), length, false);

  pendingUpdate = true;
  forceFullUpdate = false;
  cachedBox = box;
  cachedNewCoords.Uninit();
  cachedNewCoords.Init(0);
  cachedOldCoords = oldMol.GetCoords();
  cachedAtomIndices.assign(atomIndices.begin(), atomIndices.end());
  cachedCharges.assign(charges.begin(), charges.end());
  cachedSignNew = 0.0;
  cachedSignOld = -1.0;
  cachedNAtoms = length;

  return dE;
}

double EwaldPME::ChangeLambdaRecip(XYZArray const &molCoords, const double lambdaOld, const double lambdaNew, const uint molIndex, const uint box) {
  uint length = mols.MolLength(molIndex), startAtom = mols.MolStart(molIndex);
  vector<uint> atomIndices(length); vector<double> charges(length);
  double sign_new = sqrt(lambdaNew), sign_old = -sqrt(lambdaOld);
  for (uint i = 0; i < length; ++i) { atomIndices[i] = startAtom + i; charges[i] = particleCharge[startAtom + i]; }
  return DeltaERecip(box, &molCoords, sign_new, &molCoords, sign_old, atomIndices.data(), charges.data(), length, false);
}

double EwaldPME::MolExchangeReciprocal(const std::vector<cbmc::TrialMol> &newMol, const std::vector<cbmc::TrialMol> &oldMol, const std::vector<uint> &molIndexNew, const std::vector<uint> &molIndexOld, bool first_call) {
  double dE = 0.0;
  for (uint i = 0; i < newMol.size(); ++i) dE += SwapDestRecip(newMol[i], newMol[i].GetBox(), molIndexNew[i]);
  for (uint i = 0; i < oldMol.size(); ++i) dE += SwapSourceRecip(oldMol[i], oldMol[i].GetBox(), molIndexOld[i]);
  return dE;
}

Virial EwaldPME::VirialReciprocal(Virial &virial, uint box) const {
  if (box >= BOXES_WITH_U_NB || S_ref[box] == nullptr) return virial;
  SumMeshEnergy(box, S_ref[box], &virial, false); return virial;
}

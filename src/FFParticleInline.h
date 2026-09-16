/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 2.80
Copyright (C) 2022 GOMC Group
A copy of the MIT License can be found in License.txt
along with this program, also can be found at
<https://opensource.org/licenses/MIT>.
********************************************************************************/
#ifndef FF_PARTICLE_INLINE_H
#define FF_PARTICLE_INLINE_H

//
// Inline bodies for FFParticle's pair kernels.
//
// These cannot live in FFParticle.h: that header and Forcefield.h include each
// other, so when FFParticle.h is parsed the Forcefield class is still
// incomplete and `forcefield.rCutSq` will not compile. The derived forcefields
// (FFShift.h and friends) avoid this only because their bodies are parsed after
// both headers have finished.
//
// Keeping them here, rather than in FFParticle.cpp, lets translation units that
// do the actual energy evaluation inline the pair math. Without it the
// VDW_STD_KIND kernels can be devirtualised but never inlined, because the
// definitions are not visible outside FFParticle.cpp.
//
// Include this wherever FFParticle's pair math should be inlined; include it
// after Forcefield.h is complete.
//

#include "FFParticle.h"
#include "Forcefield.h"
#include "NumLib.h"

inline double FFParticle::GetRmin(const uint i, const uint j) const {
  return 0.0;
}
inline double FFParticle::GetRmax(const uint i, const uint j) const {
  return 0.0;
}
inline double FFParticle::GetRmin_1_4(const uint i, const uint j) const {
  return 0.0;
}
inline double FFParticle::GetRmax_1_4(const uint i, const uint j) const {
  return 0.0;
}

// Defining the functions

inline void FFParticle::CalcAdd_1_4(double &en, const double distSq,
                                    const uint kind1, const uint kind2) const {
  if (forcefield.rCutSq < distSq)
    return;

  uint index = FlatIndex(kind1, kind2);
  double rRat2 = sigmaSq_1_4[index] / distSq;
  double rRat4 = rRat2 * rRat2;
  double attract = rRat4 * rRat2;
  
  double repulse;
  uint nh = nExp_1_4[index];
  if (nh == 12) {
    repulse = attract * attract;
  } else if (nh != 0xFFFFFFFF) {
    double rRat6 = attract;
    repulse = num::POW(rRat2, rRat4, rRat6, nh);
  } else {
    repulse = pow(rRat2, n_1_4[index] * 0.5);
  }

  en += epsilon_cn_1_4[index] * (repulse - attract);
}

inline void FFParticle::CalcCoulombAdd_1_4(double &en, const double distSq,
                                           const double qi_qj_Fact,
                                           const bool NB) const {
  if (forcefield.rCutSq < distSq)
    return;

  double dist = sqrt(distSq);
  if (NB)
    en += qi_qj_Fact / dist;
  else
    en += qi_qj_Fact * forcefield.scaling_14 / dist;
}

// mie potential
inline double FFParticle::CalcEn(const double distSq, const uint kind1,
                                 const uint kind2, const double lambda) const {
  if (forcefield.rCutSq < distSq)
    return 0.0;

  uint index = FlatIndex(kind1, kind2);
  if (lambda >= 0.999999) {
    // save computation time
    return CalcEn(distSq, index);
  }
  double sigma6 = sigmaSq[index] * sigmaSq[index] * sigmaSq[index];
  sigma6 = std::max(sigma6, forcefield.sc_sigma_6);
  double dist6 = distSq * distSq * distSq;
  double lambdaCoef =
      forcefield.sc_alpha * pow((1.0 - lambda), forcefield.sc_power);
  double softDist6 = lambdaCoef * sigma6 + dist6;
  double softRsq = cbrt(softDist6);

  double en = lambda * CalcEn(softRsq, index);
  return en;
}

inline double FFParticle::CalcEn(const double distSq, const uint index) const {
  double rRat2 = sigmaSq[index] / distSq;
  double rRat4 = rRat2 * rRat2;
  double attract = rRat4 * rRat2;
  
  double repulse;
  uint nh = nExp[index];
  if (nh == 12) {
    repulse = attract * attract;
  } else if (nh != 0xFFFFFFFF) {
    double rRat6 = attract;
    repulse = num::POW(rRat2, rRat4, rRat6, nh);
  } else {
    repulse = pow(rRat2, n[index] * 0.5);
  }

  return (epsilon_cn[index] * (repulse - attract));
}

inline double FFParticle::CalcVir(const double distSq, const uint kind1,
                                  const uint kind2, const double lambda) const {
  if (forcefield.rCutSq < distSq)
    return 0.0;

  uint index = FlatIndex(kind1, kind2);
  if (lambda >= 0.999999) {
    // save computation time
    return CalcVir(distSq, index);
  }
  double sigma6 = sigmaSq[index] * sigmaSq[index] * sigmaSq[index];
  sigma6 = std::max(sigma6, forcefield.sc_sigma_6);
  double dist6 = distSq * distSq * distSq;
  double lambdaCoef =
      forcefield.sc_alpha * pow((1.0 - lambda), forcefield.sc_power);
  double softDist6 = lambdaCoef * sigma6 + dist6;
  double softRsq = cbrt(softDist6);
  double correction = distSq / softRsq;
  // We need to fix the return value from calcVir
  double vir = lambda * correction * correction * CalcVir(softRsq, index);
  return vir;
}

inline double FFParticle::CalcVir(const double distSq, const uint index) const {
  double rNeg2 = 1.0 / distSq;
  double rRat2 = rNeg2 * sigmaSq[index];
  double rRat4 = rRat2 * rRat2;
  double attract = rRat4 * rRat2;
  
  double repulse;
  uint nh = nExp[index];
  if (nh == 12) {
    repulse = attract * attract;
  } else if (nh != 0xFFFFFFFF) {
    double rRat6 = attract;
    repulse = num::POW(rRat2, rRat4, rRat6, nh);
  } else {
    repulse = pow(rRat2, n[index] * 0.5);
  }
  
  // Virial is F.r = -dE/dr * 1/r
  return epsilon_cn_6[index] * (nOver6[index] * repulse - attract) * rNeg2;
}

inline double FFParticle::CalcCoulomb(const double distSq, const uint kind1,
                                      const uint kind2, const double qi_qj_Fact,
                                      const double lambda, const uint b) const {
  if (forcefield.rCutCoulombSq[b] < distSq)
    return 0.0;

  if (lambda >= 0.999999) {
    // save computation time
    return CalcCoulomb(distSq, qi_qj_Fact, b);
  }
  double en = 0.0;
  if (forcefield.sc_coul) {
    uint index = FlatIndex(kind1, kind2);
    double sigma6 = sigmaSq[index] * sigmaSq[index] * sigmaSq[index];
    sigma6 = std::max(sigma6, forcefield.sc_sigma_6);
    double dist6 = distSq * distSq * distSq;
    double lambdaCoef =
        forcefield.sc_alpha * pow((1.0 - lambda), forcefield.sc_power);
    double softDist6 = lambdaCoef * sigma6 + dist6;
    double softRsq = cbrt(softDist6);
    en = lambda * CalcCoulomb(softRsq, qi_qj_Fact, b);
  } else {
    en = lambda * CalcCoulomb(distSq, qi_qj_Fact, b);
  }
  return en;
}

inline double FFParticle::CalcCoulomb(const double distSq,
                                      const double qi_qj_Fact,
                                      const uint b) const {
  if (forcefield.ewald) {
    double tab;
    if (forcefield.realTable.Energy(distSq, b, tab))
      return qi_qj_Fact * tab;
    // below the table floor: overlapping pair, exact form
    double dist = sqrt(distSq);
    double val = forcefield.alpha[b] * dist;
    return qi_qj_Fact * std::erfc(val) / dist;
  } else {
    double dist = sqrt(distSq);
    return qi_qj_Fact / dist;
  }
}

inline double FFParticle::CalcCoulombVir(const double distSq, const uint kind1,
                                         const uint kind2, const double qi_qj,
                                         const double lambda,
                                         const uint b) const {
  if (forcefield.rCutCoulombSq[b] < distSq)
    return 0.0;

  if (lambda >= 0.999999) {
    // save computation time
    return CalcCoulombVir(distSq, qi_qj, b);
  }
  double vir = 0.0;
  if (forcefield.sc_coul) {
    uint index = FlatIndex(kind1, kind2);
    double sigma6 = sigmaSq[index] * sigmaSq[index] * sigmaSq[index];
    sigma6 = std::max(sigma6, forcefield.sc_sigma_6);
    double dist6 = distSq * distSq * distSq;
    double lambdaCoef =
        forcefield.sc_alpha * pow((1.0 - lambda), forcefield.sc_power);
    double softDist6 = lambdaCoef * sigma6 + dist6;
    double softRsq = cbrt(softDist6);
    double correction = distSq / softRsq;
    // We need to fix the return value from calcVir
    vir = lambda * correction * correction * CalcCoulombVir(softRsq, qi_qj, b);
  } else {
    vir = lambda * CalcCoulombVir(distSq, qi_qj, b);
  }
  return vir;
}

inline double FFParticle::CalcCoulombVir(const double distSq,
                                         const double qi_qj,
                                         const uint b) const {
  if (forcefield.ewald) {
    double tab;
    if (forcefield.realTable.Virial(distSq, b, tab))
      return qi_qj * tab;
    // below the table floor: overlapping pair, exact form
    double dist = sqrt(distSq);
    // M_2_SQRTPI is 2/sqrt(PI)
    double constValue = forcefield.alpha[b] * M_2_SQRTPI;
    double expConstValue = exp(-1.0 * forcefield.alphaSq[b] * distSq);
    double temp = std::erfc(forcefield.alpha[b] * dist);
    return qi_qj * (temp / dist + constValue * expConstValue) / distSq;
  } else {
    double dist = sqrt(distSq);
    double result = qi_qj / (distSq * dist);
    return result;
  }
}

// Calculate the dE/dlambda for vdw energy
inline double FFParticle::CalcdEndL(const double distSq, const uint kind1,
                                    const uint kind2,
                                    const double lambda) const {
  if (forcefield.rCutSq < distSq)
    return 0.0;

  uint index = FlatIndex(kind1, kind2);
  double sigma6 = sigmaSq[index] * sigmaSq[index] * sigmaSq[index];
  sigma6 = std::max(sigma6, forcefield.sc_sigma_6);
  double dist6 = distSq * distSq * distSq;
  double lambdaCoef =
      forcefield.sc_alpha * pow((1.0 - lambda), forcefield.sc_power);
  double softDist6 = lambdaCoef * sigma6 + dist6;
  double softRsq = cbrt(softDist6);
  double fCoef = lambda * forcefield.sc_alpha * forcefield.sc_power / 6.0;
  fCoef *= pow(1.0 - lambda, forcefield.sc_power - 1.0) * sigma6 /
           (softRsq * softRsq);
  double dhdl = CalcEn(softRsq, index) + fCoef * CalcVir(softRsq, index);
  return dhdl;
}

// Calculate the dE/dlambda for Coulomb energy
inline double FFParticle::CalcCoulombdEndL(const double distSq,
                                           const uint kind1, const uint kind2,
                                           const double qi_qj_Fact,
                                           const double lambda, uint b) const {
  if (forcefield.rCutCoulombSq[b] < distSq)
    return 0.0;

  double dhdl = 0.0;
  if (forcefield.sc_coul) {
    uint index = FlatIndex(kind1, kind2);
    double sigma6 = sigmaSq[index] * sigmaSq[index] * sigmaSq[index];
    sigma6 = std::max(sigma6, forcefield.sc_sigma_6);
    double dist6 = distSq * distSq * distSq;
    double lambdaCoef =
        forcefield.sc_alpha * pow((1.0 - lambda), forcefield.sc_power);
    double softDist6 = lambdaCoef * sigma6 + dist6;
    double softRsq = cbrt(softDist6);
    double fCoef = lambda * forcefield.sc_alpha * forcefield.sc_power / 6.0;
    fCoef *= pow(1.0 - lambda, forcefield.sc_power - 1.0) * sigma6 /
             (softRsq * softRsq);
    dhdl = CalcCoulomb(softRsq, qi_qj_Fact, b) +
           fCoef * CalcCoulombVir(softRsq, qi_qj_Fact, b);
  } else {
    dhdl = CalcCoulomb(distSq, qi_qj_Fact, b);
  }
  return dhdl;
}

#endif /*FF_PARTICLE_INLINE_H*/

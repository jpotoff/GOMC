/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 2.80
Copyright (C) 2022 GOMC Group
A copy of the MIT License can be found in License.txt
along with this program, also can be found at
<https://opensource.org/licenses/MIT>.
********************************************************************************/
#ifndef FF_SWITCH_H
#define FF_SWITCH_H

#include "BasicTypes.h" //for uint
#include "FFAdapter.h"
#include "FFConst.h" //constants related to particles.
#include "NumLib.h"  //For Cb, Sq

#include <cmath>

//    Switched Mie potential
//    U(rij) = cn * eps_ij * ( (sig_ij/rij)^n - (sig_ij/rij)^6) * fE
//
//    fE = 1                                              , if rij < rswitch
//       = (rcut^2 - rij^2)^2 * factor2 * (factor1 + 2 rij^2), otherwise
//    factor1 = rcut^2 - 3 * rswitch^2
//    factor2 = (rcut^2 - rswitch^2)^-3
//
//    The energy is untouched below rswitch and falls smoothly to zero at the
//    cutoff. Unlike a constant shift, the switching function has a non-zero
//    derivative, so the virial picks up an extra term -- see ff::SwitchVir.
//
//    Everything but the switch constants and their derivation comes from
//    FFAdapter; the pair expressions are ff::MieSwitch and ff::SwitchCoul.
//

struct FF_SWITCH final
    : public FFAdapter<FF_SWITCH, ff::MieSwitch, ff::SwitchCoul> {
  friend struct FFTestAccess;

public:
  explicit FF_SWITCH(Forcefield &ff) : FFAdapter(ff) {
    rOnSq = rOn = factor1 = factor2 = 0.0;
  }

  void Init(ff_setup::Particle const &mie,
            ff_setup::NBfix const &nbfix) override;

  //! Long-range corrections relative to the full (unswitched) Mie potential,
  //! assuming g(r) = 1 beyond rSwitch:
  //!   E = 2 pi Int_rOn^rCut r^2 u (1 - S) dr  +  2 pi Int_rCut^inf r^2 u dr
  //!   W = 2 pi Int_rOn^rCut r^3 [-u'(1 - S) + u S'] dr
  //!       - 2 pi Int_rCut^inf r^3 u' dr
  //! The second term of each is the plain truncated tail (FFParticle); the
  //! first restores what the switching function removes between rSwitch and
  //! the cutoff. Same convention as OpenMM's dispersion correction with a
  //! switching function. Same normalisation as FFParticle::EnergyLRC/VirialLRC.
  double EnergyLRC(const uint kind1, const uint kind2) const override;
  double VirialLRC(const uint kind1, const uint kind2) const override;
  //! Zero: the switched potential is continuous at the cutoff, so there is no
  //! impulsive force there.
  double ImpulsePressureCorrection(const uint, const uint) const override {
    return 0.0;
  }

  //! Parameter views; these add the switching constants.
  ff::VdwParams VdwView() const {
    ff::VdwParams p = FFParticle::VdwView();
    p.rOnSq = rOnSq;
    p.factor1 = factor1;
    p.factor2 = factor2;
    return p;
  }
  ff::VdwParams VdwView14() const {
    ff::VdwParams p = FFParticle::VdwView14();
    p.rOnSq = rOnSq;
    p.factor1 = factor1;
    p.factor2 = factor2;
    return p;
  }

protected:
  double rOn, rOnSq, factor1, factor2;

private:
  //! The two switching-region integrals above, for one kind pair.
  void SwitchRegionIntegrals(const uint idx, double &en, double &vir) const;
};

inline void FF_SWITCH::Init(ff_setup::Particle const &mie,
                            ff_setup::NBfix const &nbfix) {
  // Initialize sigma and epsilon
  FFParticle::Init(mie, nbfix);
  rOn = forcefield.rswitch;
  rOnSq = rOn * rOn;
  // calculate switch constant
  factor1 = forcefield.rCutSq - 3 * rOnSq;
  factor2 = 1.0 / ((forcefield.rCutSq - rOnSq) * (forcefield.rCutSq - rOnSq) *
                   (forcefield.rCutSq - rOnSq));
}

namespace ff_switch_detail {
//! Int_a^b r^m dr, including the logarithmic case m == -1.
inline double PowerIntegral(const double m, const double a, const double b) {
  if (std::fabs(m + 1.0) < 1e-12)
    return std::log(b / a);
  return (std::pow(b, m + 1.0) - std::pow(a, m + 1.0)) / (m + 1.0);
}
} // namespace ff_switch_detail

//
// Write the switching factor as a polynomial in r,
//   S(r) = s0 + s2 r^2 + s4 r^4 + s6 r^6,
// expanded from factor2 (rc^2 - r^2)^2 (factor1 + 2 r^2). With
// u(r) = A r^-n - B r^-6 (A = eps_cn sigma^n, B = eps_cn sigma^6):
//   r^2 u (1 - S)          = sum_k c_k r^(k+2) (A r^-n - B r^-6)
//   r^3 [-u'(1 - S) + u S'] = sum_k c_k r^(k+2) (n A r^-n - 6 B r^-6)
//                           + sum_k d_k r^(k+2) (A r^-n - B r^-6)
// with c_0 = 1 - s0, c_k = -s_k (k > 0) and d_k = k s_k, the latter from
// r S' = 2 s2 r^2 + 4 s4 r^4 + 6 s6 r^6. Each term integrates in closed form.
// The r^-n terms hit the logarithm for odd n = 3, 5, 7 or 9.
//
inline void FF_SWITCH::SwitchRegionIntegrals(const uint idx, double &en,
                                             double &vir) const {
  using ff_switch_detail::PowerIntegral;
  en = vir = 0.0;
  const double rc = forcefield.rCut;
  const double rcSq = forcefield.rCutSq;
  if (rOn >= rc)
    return;
  const double N = n[idx];
  const double sigma = sqrt(sigmaSq[idx]);
  const double A = epsilon_cn[idx] * pow(sigma, N);
  const double B = epsilon_cn[idx] * num::Cb(sigmaSq[idx]);

  const double s[4] = {factor2 * factor1 * rcSq * rcSq,
                       factor2 * (2.0 * rcSq * rcSq - 2.0 * factor1 * rcSq),
                       factor2 * (factor1 - 4.0 * rcSq), 2.0 * factor2};
  for (int j = 0; j < 4; ++j) {
    const double k = 2.0 * j;
    const double c = (j == 0 ? 1.0 - s[0] : -s[j]);
    const double d = k * s[j];
    const double In = PowerIntegral(k + 2.0 - N, rOn, rc);
    const double I6 = PowerIntegral(k - 4.0, rOn, rc);
    en += c * (A * In - B * I6);
    vir += c * (N * A * In - 6.0 * B * I6) + d * (A * In - B * I6);
  }
  en *= 2.0 * M_PI;
  vir *= 2.0 * M_PI;
}

inline double FF_SWITCH::EnergyLRC(const uint kind1, const uint kind2) const {
  double en, vir;
  SwitchRegionIntegrals(FlatIndex(kind1, kind2), en, vir);
  return FFParticle::EnergyLRC(kind1, kind2) + en;
}

inline double FF_SWITCH::VirialLRC(const uint kind1, const uint kind2) const {
  double en, vir;
  SwitchRegionIntegrals(FlatIndex(kind1, kind2), en, vir);
  return FFParticle::VirialLRC(kind1, kind2) + vir;
}

#endif /*FF_SWITCH_H*/

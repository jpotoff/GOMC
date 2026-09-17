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

  //! Returns zero: the switching function brings the energy to zero at the
  //! cutoff, so there is no tail beyond it to correct for.
  double EnergyLRC(const uint, const uint) const override { return 0.0; }
  double VirialLRC(const uint, const uint) const override { return 0.0; }
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

#endif /*FF_SWITCH_H*/

/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 2.80
Copyright (C) 2022 GOMC Group
A copy of the MIT License can be found in License.txt
along with this program, also can be found at
<https://opensource.org/licenses/MIT>.
********************************************************************************/
#ifndef FF_SHIFT_H
#define FF_SHIFT_H

#include "BasicTypes.h" //for uint
#include "FFAdapter.h"
#include "FFConst.h" //constants related to particles.
#include "NumLib.h"  //For Cb, Sq

//    Shifted Mie potential
//    U(rij) = cn * eps_ij * ( (sig_ij/rij)^n - (sig_ij/rij)^6) - shiftConst
//    shiftConst = cn * eps_ij * ( (sig_ij/rcut)^n - (sig_ij/rcut)^6)
//    cn = n/(n-6) * ((n/6)^(6/(n-6)))
//
//    The energy therefore vanishes at the cutoff. A constant offset does not
//    change the force, so the virial is the unshifted one -- see ff::ShiftVir.
//
//    Everything except the shift constant and its own parameter arrays comes
//    from FFAdapter; the pair expressions are ff::MieShift and ff::ShiftCoul.
//

struct FF_SHIFT final : public FFAdapter<FF_SHIFT, ff::MieShift, ff::ShiftCoul> {
  friend struct FFTestAccess;

public:
  explicit FF_SHIFT(Forcefield &ff)
      : FFAdapter(ff), shiftConst(NULL), shiftConst_1_4(NULL) {}

  ~FF_SHIFT() override {
    delete[] shiftConst;
    delete[] shiftConst_1_4;
  }

  void Init(ff_setup::Particle const &mie,
            ff_setup::NBfix const &nbfix) override;

  //! Returns zero: the shift already removes the energy at the cutoff, so
  //! there is no tail beyond it to correct for.
  double EnergyLRC(const uint, const uint) const override { return 0.0; }
  double VirialLRC(const uint, const uint) const override { return 0.0; }
  double ImpulsePressureCorrection(const uint, const uint) const override {
    return 0.0;
  }

  //! Parameter views; these add shiftConst to what FFParticle supplies.
  ff::VdwParams VdwView() const {
    ff::VdwParams p = FFParticle::VdwView();
    p.shiftConst = shiftConst;
    return p;
  }
  ff::VdwParams VdwView14() const {
    ff::VdwParams p = FFParticle::VdwView14();
    p.shiftConst = shiftConst_1_4;
    return p;
  }

protected:
  double *shiftConst, *shiftConst_1_4;
};

inline void FF_SHIFT::Init(ff_setup::Particle const &mie,
                           ff_setup::NBfix const &nbfix) {
  // Initialize sigma and epsilon
  FFParticle::Init(mie, nbfix);
  uint size = num::Sq(count);
  // allocate memory
  shiftConst = new double[size];
  shiftConst_1_4 = new double[size];
  // calculate shift constant
  for (uint i = 0; i < count; ++i) {
    for (uint j = 0; j < count; ++j) {
      uint idx = FlatIndex(i, j);
      double rRat2 = sigmaSq[idx] / forcefield.rCutSq;
      double rRat4 = rRat2 * rRat2;
      double attract = rRat4 * rRat2;
      // for 1-4 interaction
      double rRat2_1_4 = sigmaSq_1_4[idx] / forcefield.rCutSq;
      double rRat4_1_4 = rRat2_1_4 * rRat2_1_4;
      double attract_1_4 = rRat4_1_4 * rRat2_1_4;
      double repulse = pow(sqrt(rRat2), n[idx]);
      double repulse_1_4 = pow(sqrt(rRat2_1_4), n_1_4[idx]);

      shiftConst[idx] = epsilon_cn[idx] * (repulse - attract);
      shiftConst_1_4[idx] = epsilon_cn_1_4[idx] * (repulse_1_4 - attract_1_4);
    }
  }
}

#endif /*FF_SHIFT_H*/

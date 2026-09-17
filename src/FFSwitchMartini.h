/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#ifndef FF_SWITCH_MARTINI_H
#define FF_SWITCH_MARTINI_H

#include "BasicTypes.h" //for uint
#include "FFConst.h"    //constants related to particles.
#include "FFAdapter.h"
#include "MiePotential.h"
#include "CoulEvaluators.h"
#include "VdwEvaluators.h"
#include "NumLib.h" //For Cb, Sq

///////////////////////////////////////////////////////////////////////
////////////////////////// LJ Switch Martini Style ////////////////////////////
///////////////////////////////////////////////////////////////////////
// LJ potential calculation:
// Eij = cn * eps_ij * ( sig_ij^n * (1/rij^n + phi(n)) -
//       sig_ij^6 * (1/rij^6 + phi(6)))
// cn = n/(n-6) * ((n/6)^(6/(n-6)))
//
// Eelec = qi*qj*(1/rij + phi(1))
// Welec = qi*qj*(1/rij^3 + phiW(1)/r)
//
// phi(x) = -Cx , if r < rswitch
// phi(x) = -Ax *(r - rswitch)^3/3 - Bx * (r - rswitch)^4 / 4 - Cx ,if r>rswitch
//
// Ax = x * ((x + 1) * rswitch - (x + 4) * rcut) /
//      (rcut^(x + 2)) * (rcut - rswitch)^2)
// Bx = x * ((x + 1) * rswitch - (x + 3) * rcut) /
//      (rcut^(x + 2)) * (rcut - rswitch)^3)
// Cx = 1/rcut^x - Ax * (rcut - rswitch)^3 / 3 - Bx * (rcut - rswitch)^4 / 4
//
// Virial Calculation
//
// Wij = cn * eps_ij * ( sig_ij^n * (n/rij^(n+2) + phiW(n)/rij) -
//       sig_ij^6 * (6/rij^(6+2) + phiW(6)/r))
//
// phiW(x) = 0 , if r < rswitch
// phiW(x) = Ax *(r - rswitch)^2 + Bx * (r - rswitch)^3 ,if r > rswitch
//
//

// `final` lets the compiler resolve this class's own virtual calls --
// notably the 4-argument CalcEn/CalcCoulomb calling their 2-argument
// counterparts -- statically, which is what allows the templated energy
// kernels to inline the pair math. Nothing derives from these.
struct FF_SWITCH_MARTINI final
    : public FFAdapter<FF_SWITCH_MARTINI, ff::MartiniEval, ff::MartiniCoul> {
  friend struct FFTestAccess;

public:
  FF_SWITCH_MARTINI(Forcefield &ff)
      : FFAdapter(ff), An(NULL), Bn(NULL), Cn(NULL), An_1_4(NULL),
        Bn_1_4(NULL), Cn_1_4(NULL), sig6(NULL), sign(NULL), sig6_1_4(NULL),
        sign_1_4(NULL) {
    A1 = B1 = C1 = A6 = B6 = C6 = 0.0;
  }
  ~FF_SWITCH_MARTINI() override {
    delete[] An;
    delete[] Bn;
    delete[] Cn;
    delete[] An_1_4;
    delete[] Bn_1_4;
    delete[] Cn_1_4;
    delete[] sig6;
    delete[] sign;
    delete[] sig6_1_4;
    delete[] sign_1_4;
  }

  //! Parameter views; these add the Martini switching constants.
  ff::VdwParams VdwView() const {
    ff::VdwParams p = FFParticle::VdwView();
    p.An = An; p.Bn = Bn; p.Cn = Cn; p.sig6 = sig6; p.sign = sign;
    p.A6 = A6; p.B6 = B6; p.C6 = C6; p.rOn = rOn; p.rOnSq = rOnSq;
    return p;
  }
  ff::VdwParams VdwView14() const {
    ff::VdwParams p = FFParticle::VdwView14();
    p.An = An_1_4; p.Bn = Bn_1_4; p.Cn = Cn_1_4;
    p.sig6 = sig6_1_4; p.sign = sign_1_4;
    p.A6 = A6; p.B6 = B6; p.C6 = C6; p.rOn = rOn; p.rOnSq = rOnSq;
    return p;
  }
  //! Martini's electrostatics are dielectric-screened with their own shift.
  ff::CoulParams CoulView(const uint b) const {
    ff::CoulParams c = FFParticle::CoulView(b);
    c.diElectric_1 = diElectric_1;
    c.A1 = A1; c.B1 = B1; c.C1 = C1;
    return c;
  }

  void Init(ff_setup::Particle const &mie,
            ff_setup::NBfix const &nbfix) override;


  // coulomb interaction functions

  //! Returns Ezero, no energy correction
  double EnergyLRC(const uint kind1, const uint kind2) const override {
    return 0.0;
  }
  //!!Returns Ezero, no virial correction
  double VirialLRC(const uint kind1, const uint kind2) const override {
    return 0.0;
  }
  //! Returns zero for impulse pressure correction term for a kind pair
  double ImpulsePressureCorrection(const uint kind1,
                                   const uint kind2) const override {
    return 0.0;
  }

  // Calculate the dE/dlambda for vdw energy
  // Calculate the dE/dlambda for Coulomb energy



protected:

  double *An, *Bn, *Cn, *An_1_4, *Bn_1_4, *Cn_1_4;
  double *sig6, *sign, *sig6_1_4, *sign_1_4;

  double diElectric_1, rOn, rOnSq, rOnCoul, A1, B1, C1, A6, B6, C6;
};

inline void FF_SWITCH_MARTINI::Init(ff_setup::Particle const &mie,
                                    ff_setup::NBfix const &nbfix) {
  // Initialize sigma and epsilon
  FFParticle::Init(mie, nbfix);
  uint size = num::Sq(count);
  // allocate memory
  An = new double[size];
  Bn = new double[size];
  Cn = new double[size];
  sign = new double[size];
  sig6 = new double[size];
  An_1_4 = new double[size];
  Bn_1_4 = new double[size];
  Cn_1_4 = new double[size];
  sign_1_4 = new double[size];
  sig6_1_4 = new double[size];
  // Set martini constant
  diElectric_1 = 1.0 / forcefield.dielectric;
  rOn = forcefield.rswitch;
  rOnSq = rOn * rOn;
  // in Martini, Coulomb switching distance is zero
  rOnCoul = 0.0;
  double rCut = forcefield.rCut;
  // Original Unoptimized computation
  // Set LJ constants
  // A6 = 6.0 * ((6.0 + 1) * rOn - (6.0 + 4) * rCut) / (pow(rCut, 6.0 + 2) *
  // pow(rCut - rOn, 2));
  // B6 = -6.0 * ((6.0 + 1) * rOn - (6.0 + 3) * rCut) / (pow(rCut, 6.0 + 2) *
  // pow(rCut - rOn, 3));
  // C6 = 1.0 / pow(rCut, 6.0) - A6 / 3.0 * pow(rCut - rOn, 3) - B6 / 4.0 *
  // pow(rCut - rOn, 4);
  // // Set Coulomb constants
  // A1 = 1.0 * ((1.0 + 1) * rOnCoul - (1.0 + 4) * rCut) / (pow(rCut, 1.0 + 2) *
  // pow(rCut - rOnCoul, 2));
  // B1 = -1.0 * ((1.0 + 1) * rOnCoul - (1.0 + 3) * rCut) / (pow(rCut, 1.0 + 2)
  // * pow(rCut - rOnCoul, 3));
  // C1 = 1.0 / pow(rCut, 1.0) - A1 / 3.0 * pow(rCut - rOnCoul, 3) - B1 / 4.0 *
  // pow(rCut - rOnCoul, 4);

  // Optimized computation
  // Set LJ constants
  A6 = 6.0 * (7.0 * rOn - 10.0 * rCut) /
       (pow(rCut, 8.0) * (rCut - rOn) * (rCut - rOn));
  B6 = -6.0 * (7.0 * rOn - 9.0 * rCut) /
       (pow(rCut, 8.0) * (rCut - rOn) * (rCut - rOn) * (rCut - rOn));
  C6 = pow(rCut, -6.0) - A6 / 3.0 * (rCut - rOn) * (rCut - rOn) * (rCut - rOn) -
       B6 / 4.0 * (rCut - rOn) * (rCut - rOn) * (rCut - rOn) * (rCut - rOn);
  // Set Coulomb constants
  A1 = (2.0 * rOnCoul - 5.0 * rCut) /
       (rCut * rCut * rCut * (rCut - rOnCoul) * (rCut - rOnCoul));
  B1 = -1.0 * (2.0 * rOnCoul - 4.0 * rCut) /
       (rCut * rCut * rCut * (rCut - rOnCoul) * (rCut - rOnCoul) *
        (rCut - rOnCoul));
  C1 = 1.0 / rCut -
       A1 / 3.0 * (rCut - rOnCoul) * (rCut - rOnCoul) * (rCut - rOnCoul) -
       B1 / 4.0 * (rCut - rOnCoul) * (rCut - rOnCoul) * (rCut - rOnCoul) *
           (rCut - rOnCoul);

  for (uint i = 0; i < count; ++i) {
    for (uint j = 0; j < count; ++j) {
      uint idx = FlatIndex(i, j);
      double pn = n[idx];
      An[idx] = pn * ((pn + 1.0) * rOn - (pn + 4.0) * rCut) /
                (pow(rCut, pn + 2.0) * (rCut - rOn) * (rCut - rOn));
      Bn[idx] =
          -pn * ((pn + 1.0) * rOn - (pn + 3.0) * rCut) /
          (pow(rCut, pn + 2.0) * (rCut - rOn) * (rCut - rOn) * (rCut - rOn));
      Cn[idx] = 1.0 / pow(rCut, pn) -
                An[idx] / 3.0 * (rCut - rOn) * (rCut - rOn) * (rCut - rOn) -
                Bn[idx] / 4.0 * (rCut - rOn) * (rCut - rOn) * (rCut - rOn) *
                    (rCut - rOn);
      double sigma = sqrt(sigmaSq[idx]);
      sig6[idx] = pow(sigma, 6.0);
      sign[idx] = pow(sigma, pn);

      // for 1-4 interaction
      double pn_1_4 = n_1_4[idx];
      An_1_4[idx] = pn_1_4 * ((pn_1_4 + 1.0) * rOn - (pn_1_4 + 4.0) * rCut) /
                    (pow(rCut, pn_1_4 + 2.0) * (rCut - rOn) * (rCut - rOn));
      Bn_1_4[idx] = -pn_1_4 * ((pn_1_4 + 1.0) * rOn - (pn_1_4 + 3.0) * rCut) /
                    (pow(rCut, pn_1_4 + 2.0) * (rCut - rOn) * (rCut - rOn) *
                     (rCut - rOn));
      Cn_1_4[idx] =
          1.0 / pow(rCut, pn_1_4) -
          An_1_4[idx] / 3.0 * (rCut - rOn) * (rCut - rOn) * (rCut - rOn) -
          Bn_1_4[idx] / 4.0 * (rCut - rOn) * (rCut - rOn) * (rCut - rOn) *
              (rCut - rOn);
      double sigma_1_4 = sqrt(sigmaSq_1_4[idx]);
      sig6_1_4[idx] = pow(sigma_1_4, 6.0);
      sign_1_4[idx] = pow(sigma_1_4, pn_1_4);
    }
  }
}












// Calculate the dE/dlambda for Coulomb energy

#endif /*FF_SWITCH_MARTINI_H*/

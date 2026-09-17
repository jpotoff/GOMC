/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 2.80
Copyright (C) 2022 GOMC Group
A copy of the MIT License can be found in License.txt
along with this program, also can be found at
<https://opensource.org/licenses/MIT>.
********************************************************************************/
#ifndef COUL_EVALUATORS_H
#define COUL_EVALUATORS_H

#include "BasicTypes.h"
#include "EwaldRealTable.h"
#include "MiePotential.h" // GOMC_HOSTDEV
#include <cmath>

//
// Real-space electrostatics.
//
// This splits differently from the van der Waals side, and the split reflects
// the physics rather than convenience:
//
//   Ewald ON  -- every forcefield computes the same thing,
//                  qq * erfc(alpha*r) / r
//                so there is one evaluator. This was ten identical copies
//                (CalcCoulomb and CalcCoulombVir of all five flavors).
//
//   Ewald OFF -- the flavors genuinely differ: plain 1/r, shifted so it
//                vanishes at the cutoff, switched, and Martini's own form.
//                These stay separate because they are separate physics.
//
// NOTE ON GPU SHARING: unlike the vdW evaluators, EwaldReal should NOT be
// shared with the CUDA kernels. It reads a lookup table, which on a GPU is a
// random global-memory access and costs more than simply calling erfc() --
// the opposite of the CPU trade-off. The GPU kernels compute erfc directly,
// and that is correct for them. Hence no GOMC_HOSTDEV on EwaldReal.
//

namespace ff {

//! Parameters the real-space electrostatic kernels need.
struct CoulParams {
  const EwaldRealTable *table; //!< tabulated erfc(alpha*r)/r; CPU only
  double alpha;                //!< Ewald splitting parameter for this box
  double rCut;
  double rCutSq;
  uint box;
  // Martini only
  double diElectric_1;
  double A1;
  double B1;
  double C1;
};

//
// qq * erfc(alpha*r) / r, via the table where it covers the separation and the
// exact form otherwise (below the table floor, i.e. overlapping pairs).
//
struct EwaldReal {
  static double Energy(const CoulParams &c, const double distSq,
                       const double qi_qj_Fact) {
    double tab;
    if (c.table && c.table->Energy(distSq, c.box, tab))
      return qi_qj_Fact * tab;
    const double dist = sqrt(distSq);
    return qi_qj_Fact * erfc(c.alpha * dist) / dist;
  }

  static double Virial(const CoulParams &c, const double distSq,
                       const double qi_qj) {
    double tab;
    if (c.table && c.table->Virial(distSq, c.box, tab))
      return qi_qj * tab;
    const double dist = sqrt(distSq);
    const double constValue = c.alpha * M_2_SQRTPI;
    const double expConstValue = exp(-1.0 * c.alpha * c.alpha * distSq);
    const double temp = erfc(c.alpha * dist);
    return qi_qj * (temp / dist + constValue * expConstValue) / distSq;
  }
};

// ------------------------------------------- non-Ewald, per forcefield ----

//! qq / r
struct PlainCoul {
  GOMC_HOSTDEV static double Energy(const CoulParams &, const double distSq,
                                    const double qi_qj_Fact) {
    return qi_qj_Fact / sqrt(distSq);
  }
  GOMC_HOSTDEV static double Virial(const CoulParams &, const double distSq,
                                    const double qi_qj) {
    return qi_qj / (distSq * sqrt(distSq));
  }
};

//! qq * (1/r - 1/rCut), so it vanishes at the cutoff
struct ShiftCoul {
  GOMC_HOSTDEV static double Energy(const CoulParams &c, const double distSq,
                                    const double qi_qj_Fact) {
    return qi_qj_Fact * (1.0 / sqrt(distSq) - 1.0 / c.rCut);
  }
  //! A constant shift leaves the force unchanged, so this is the plain form.
  GOMC_HOSTDEV static double Virial(const CoulParams &, const double distSq,
                                    const double qi_qj) {
    return qi_qj / (distSq * sqrt(distSq));
  }
};

//! qq * (r^2/rCut^2 - 1)^2 / r
struct SwitchCoul {
  GOMC_HOSTDEV static double Energy(const CoulParams &c, const double distSq,
                                    const double qi_qj_Fact) {
    double switchVal = distSq / c.rCutSq - 1.0;
    switchVal *= switchVal;
    return qi_qj_Fact * switchVal / sqrt(distSq);
  }
  GOMC_HOSTDEV static double Virial(const CoulParams &c, const double distSq,
                                    const double qi_qj) {
    const double dist = sqrt(distSq);
    double switchVal = distSq / c.rCutSq - 1.0;
    switchVal *= switchVal;
    const double dSwitchVal =
        2.0 * (distSq / c.rCutSq - 1.0) * 2.0 * dist / c.rCutSq;
    return -qi_qj * (dSwitchVal / distSq - switchVal / (distSq * dist));
  }
};

//
// Martini electrostatics: a dielectric-screened Coulomb with its own shift.
// Its Coulomb switching distance is zero, so (r - rOnCoul) is just r --
// hence the bare powers of dist below rather than differences.
//
struct MartiniCoul {
  GOMC_HOSTDEV static double Energy(const CoulParams &c, const double distSq,
                                    const double qi_qj_Fact) {
    const double dist = sqrt(distSq);
    const double rij_ronCoul_3 = dist * distSq;
    const double rij_ronCoul_4 = distSq * distSq;
    const double coul =
        -(c.A1 / 3.0) * rij_ronCoul_3 - (c.B1 / 4.0) * rij_ronCoul_4 - c.C1;
    return qi_qj_Fact * c.diElectric_1 * (1.0 / dist + coul);
  }

  GOMC_HOSTDEV static double Virial(const CoulParams &c, const double distSq,
                                    const double qi_qj) {
    const double dist = sqrt(distSq);
    const double rij_ronCoul_2 = distSq;
    const double rij_ronCoul_3 = dist * distSq;
    // phiW(1) = A1*(r-rOnCoul)^2 + B1*(r-rOnCoul)^3, so these multiply.
    // Released versions divided by them instead -- the vdW half of the same
    // forcefield (ff::MartiniEval::Virial, dshifttempRep/dshifttempAtt) has
    // always multiplied, which is what made the typo visible. Caught by
    // MartiniCoulombVirialIsNegativeDerivative.
    const double virCoul = c.A1 * rij_ronCoul_2 + c.B1 * rij_ronCoul_3;
    return qi_qj * c.diElectric_1 * (1.0 / (dist * distSq) + virCoul / dist);
  }
};

} // namespace ff

#endif /*COUL_EVALUATORS_H*/

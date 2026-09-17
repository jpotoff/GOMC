/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 2.80
Copyright (C) 2022 GOMC Group
A copy of the MIT License can be found in License.txt
along with this program, also can be found at
<https://opensource.org/licenses/MIT>.
********************************************************************************/
#ifndef VDW_EVALUATORS_H
#define VDW_EVALUATORS_H

#include "BasicTypes.h"
#include "MiePotential.h"
#include <cmath>

//
// van der Waals pair energy, as composable evaluators instead of a class
// hierarchy.
//
// Three concerns are kept apart:
//
//   VdwParams  -- a non-owning view of the parameter arrays. Plain data: no
//                 virtuals, no ownership, nothing to construct or destroy.
//   Core       -- the raw pair terms of a potential (Mie, Exp-6).
//   Trunc      -- how those terms are combined and truncated (plain, shifted,
//                 switched).
//
// The forcefields the user selects are then compositions:
//
//   MiePlain  = PairEval<MieCore, PlainTrunc>     (VDW_STD)
//   MieShift  = PairEval<MieCore, ShiftTrunc>     (VDW_SHIFT)
//   MieSwitch = PairEval<MieCore, SwitchTrunc>    (VDW_SWITCH)
//
// Adding a truncation gives it to every core for free, and adding a core gives
// it every truncation for free -- N+M pieces instead of N*M classes. Adding
// force-shifted VdW historically cost 280 lines across 10 files; as a Trunc it
// is a handful.
//
// Martini and Exp-6 do not fit the Core x Trunc shape -- Martini combines the
// two Mie terms differently rather than truncating them, and Exp-6 is a
// different functional form -- so they are standalone evaluators. That is an
// honest description of the physics, not a gap.
//
// These take raw pointers and scalars deliberately: it keeps them usable from
// the CUDA kernels, which currently reimplement all of this (and never received
// the integer-exponent fast path). GOMC_HOSTDEV marks them for that.
//
// Cutoff checks, lambda handling and the 1-4 parameter choice stay with the
// caller; these evaluate the potential itself.
//

namespace ff {

//
// A view of one forcefield's parameters. Each evaluator reads a subset; the
// rest may be null. Cheap to build, so callers make one per call.
//
struct VdwParams {
  // Mie core
  const double *sigmaSq;
  const double *epsilon_cn;
  const double *epsilon_cn_6;
  const double *nOver6;
  const double *n;
  const uint *nExp;
  // shifted
  const double *shiftConst;
  // switched
  double rCutSq;
  double rOnSq;
  double factor1;
  double factor2;
  // Martini
  const double *An;
  const double *Bn;
  const double *Cn;
  const double *sig6;
  const double *sign;
  double A6;
  double B6;
  double C6;
  double rOn;
  // Exp-6
  const double *expConst;
  const double *rMin;
};

// ---------------------------------------------------------------- cores ----

struct MieCore {
  //! (sigma/r)^6 and (sigma/r)^n
  GOMC_HOSTDEV static MieTerms Terms(const VdwParams &p, const double distSq,
                                     const uint i) {
    return MiePair(p.sigmaSq[i] / distSq, p.nExp[i], p.n[i]);
  }
};

// ---------------------------------------------------------- truncations ----

struct PlainTrunc {
  GOMC_HOSTDEV static double Energy(const VdwParams &p, const MieTerms &t,
                                    const double, const uint i) {
    return p.epsilon_cn[i] * (t.repulse - t.attract);
  }
};

struct ShiftTrunc {
  //! Plain, less a constant so the energy vanishes at the cutoff.
  GOMC_HOSTDEV static double Energy(const VdwParams &p, const MieTerms &t,
                                    const double, const uint i) {
    return p.epsilon_cn[i] * (t.repulse - t.attract) - p.shiftConst[i];
  }
};

struct SwitchTrunc {
  //! Plain, multiplied by a factor that is 1 below rSwitch and 0 at the cutoff.
  GOMC_HOSTDEV static double Energy(const VdwParams &p, const MieTerms &t,
                                    const double distSq, const uint i) {
    const double rCutSq_rijSq = p.rCutSq - distSq;
    const double rCutSq_rijSq_Sq = rCutSq_rijSq * rCutSq_rijSq;
    const double fE = rCutSq_rijSq_Sq * p.factor2 * (p.factor1 + 2.0 * distSq);
    const double factE = (distSq > p.rOnSq ? fE : 1.0);
    return (p.epsilon_cn[i] * (t.repulse - t.attract)) * factE;
  }
};

// --------------------------------------------------- virial truncations ----
//
// GOMC's convention: Vir = F.r / r^2 = -(1/r) dU/dr. The Mie form gives
//   Vir = epsilon_cn_6 * (nOver6 * repulse - attract) / r^2
// A constant energy shift does not change the force, so ShiftTrunc reuses the
// plain virial rather than defining its own.
//

struct PlainVir {
  GOMC_HOSTDEV static double Virial(const VdwParams &p, const MieTerms &t,
                                    const double distSq, const uint i) {
    const double rNeg2 = 1.0 / distSq;
    return p.epsilon_cn_6[i] * (p.nOver6[i] * t.repulse - t.attract) * rNeg2;
  }
};

//! Shifting the energy by a constant leaves the force unchanged.
typedef PlainVir ShiftVir;

struct SwitchVir {
  GOMC_HOSTDEV static double Virial(const VdwParams &p, const MieTerms &t,
                                    const double distSq, const uint i) {
    const double rCutSq_rijSq = p.rCutSq - distSq;
    const double rCutSq_rijSq_Sq = rCutSq_rijSq * rCutSq_rijSq;
    const double rNeg2 = 1.0 / distSq;
    const double fE = rCutSq_rijSq_Sq * p.factor2 * (p.factor1 + 2.0 * distSq);
    const double fW = 12.0 * p.factor2 * rCutSq_rijSq * (p.rOnSq - distSq);
    const double factE = (distSq > p.rOnSq ? fE : 1.0);
    const double factW = (distSq > p.rOnSq ? fW : 0.0);
    const double Wij =
        p.epsilon_cn_6[i] * (p.nOver6[i] * t.repulse - t.attract) * rNeg2;
    const double Eij = p.epsilon_cn[i] * (t.repulse - t.attract);
    return (Wij * factE - Eij * factW);
  }
};

// ------------------------------------------------------------ composition --

template <class Core, class Trunc, class Vir> struct PairEval {
  GOMC_HOSTDEV static double Energy(const VdwParams &p, const double distSq,
                                    const uint i) {
    return Trunc::Energy(p, Core::Terms(p, distSq, i), distSq, i);
  }
  GOMC_HOSTDEV static double Virial(const VdwParams &p, const double distSq,
                                    const uint i) {
    return Vir::Virial(p, Core::Terms(p, distSq, i), distSq, i);
  }
};

typedef PairEval<MieCore, PlainTrunc, PlainVir> MiePlain;    //!< VDW_STD
typedef PairEval<MieCore, ShiftTrunc, ShiftVir> MieShift;    //!< VDW_SHIFT
typedef PairEval<MieCore, SwitchTrunc, SwitchVir> MieSwitch; //!< VDW_SWITCH

// --------------------------------------------------- standalone evaluators --

//
// Martini reuses the two Mie terms but weights them separately and adds its own
// shift, so it is not a truncation of the plain form.
//
struct MartiniEval {
  GOMC_HOSTDEV static double Energy(const VdwParams &p, const double distSq,
                                    const uint i) {
    const double r_2 = 1.0 / distSq;
    const double r_4 = r_2 * r_2;
    const double r_6 = r_4 * r_2;
    const double r_n = MiePair(r_2, p.nExp[i], p.n[i]).repulse;

    const double rij_ron = sqrt(distSq) - p.rOn;
    const double rij_ron_2 = rij_ron * rij_ron;
    const double rij_ron_3 = rij_ron_2 * rij_ron;
    const double rij_ron_4 = rij_ron_2 * rij_ron_2;

    const double shifttempRep = -(p.An[i] / 3.0) * rij_ron_3 -
                                (p.Bn[i] / 4.0) * rij_ron_4 - p.Cn[i];
    const double shifttempAtt =
        -(p.A6 / 3.0) * rij_ron_3 - (p.B6 / 4.0) * rij_ron_4 - p.C6;

    const double shiftRep = (distSq > p.rOnSq ? shifttempRep : -p.Cn[i]);
    const double shiftAtt = (distSq > p.rOnSq ? shifttempAtt : -p.C6);

    return p.epsilon_cn[i] *
           (p.sign[i] * (r_n + shiftRep) - p.sig6[i] * (r_6 + shiftAtt));
  }

  GOMC_HOSTDEV static double Virial(const VdwParams &p, const double distSq,
                                    const uint i) {
    const double n_ij = p.n[i];
    const double r_1 = 1.0 / sqrt(distSq);

    const double r_2 = 1.0 / distSq;
    // 1/r^8. This read `distSq * distSq * distSq * distSq` -- i.e. r^8 -- in
    // every released version of FF_SWITCH_MARTINI::CalcVir. The attractive
    // term of W = -(1/r) dU/dr for U = -sig6/r^6 is +6/r^8, so the sign of
    // the exponent was wrong and the Martini virial (hence the pressure) was
    // wrong by a factor of r^16. Caught by MartiniVirialIsNegativeEnergy-
    // Derivative, which compares against a finite difference of CalcEn.
    const double r_8 = r_2 * r_2 * r_2 * r_2;
    const double r_n = MiePair(r_2, p.nExp[i], p.n[i]).repulse;
    const double r_n2 = r_n * r_2;

    const double rij_ron = sqrt(distSq) - p.rOn;
    const double rij_ron_2 = rij_ron * rij_ron;
    const double rij_ron_3 = rij_ron_2 * rij_ron;

    const double dshifttempRep = p.An[i] * rij_ron_2 + p.Bn[i] * rij_ron_3;
    const double dshifttempAtt = p.A6 * rij_ron_2 + p.B6 * rij_ron_3;

    const double dshiftRep = (distSq > p.rOnSq ? dshifttempRep * r_1 : 0);
    const double dshiftAtt = (distSq > p.rOnSq ? dshifttempAtt * r_1 : 0);

    return p.epsilon_cn[i] * (p.sign[i] * (n_ij * r_n2 + dshiftRep) -
                              p.sig6[i] * (6.0 * r_8 + dshiftAtt));
  }
};

//
// Exp-6 (Buckingham): a different functional form, sharing no terms with Mie.
//   U(r) = expConst * [ (6/alpha) * exp(alpha * (1 - r/rMin)) - (rMin/r)^6 ]
// `n` carries alpha here.
//
struct Exp6Eval {
  GOMC_HOSTDEV static double Energy(const VdwParams &p, const double distSq,
                                    const uint i) {
    const double dist = sqrt(distSq);
    const double rRat = p.rMin[i] / dist;
    const double rRat2 = rRat * rRat;
    const double attract = rRat2 * rRat2 * rRat2;

    const uint alpha_ij = p.n[i];
    const double repulse =
        (6.0 / alpha_ij) * exp(alpha_ij * (1.0 - dist / p.rMin[i]));
    return p.expConst[i] * (repulse - attract);
  }

  //! Vir = 6 * expConst * [(r/rMin) exp(alpha (1 - r/rMin)) - (rMin/r)^6] / r^2
  GOMC_HOSTDEV static double Virial(const VdwParams &p, const double distSq,
                                    const uint i) {
    const double dist = sqrt(distSq);
    const double rRat = p.rMin[i] / dist;
    const double rRat2 = rRat * rRat;
    const double attract = rRat2 * rRat2 * rRat2;

    const uint alpha_ij = p.n[i];
    const double repulse =
        (dist / p.rMin[i]) * exp(alpha_ij * (1.0 - dist / p.rMin[i]));
    return 6.0 * p.expConst[i] * (repulse - attract) / distSq;
  }
};

// ---------------------------------------------- lambda-aware wrappers ----
//
// The soft-core entry points. At lambda = 1 -- the overwhelmingly common case,
// and every pair in a simulation that is not a free-energy calculation -- these
// short-circuit straight to the ordinary potential.
//
// The virial picks up a (distSq/softRsq)^2 correction because it is evaluated
// at the softened separation but must be reported for the real one.
//

template <class Eval>
GOMC_HOSTDEV inline double
SoftCoreEnergy(const VdwParams &p, const double distSq, const uint i,
               const double lambda, const double scAlpha, const uint scPower,
               const double scSigma6) {
  if (lambda >= 0.999999)
    return Eval::Energy(p, distSq, i);
  const SoftCoreDist sc =
      SoftenedDistance(distSq, p.sigmaSq[i], lambda, scAlpha, scPower, scSigma6);
  return lambda * Eval::Energy(p, sc.softRsq, i);
}

template <class Eval>
GOMC_HOSTDEV inline double
SoftCoreVirial(const VdwParams &p, const double distSq, const uint i,
               const double lambda, const double scAlpha, const uint scPower,
               const double scSigma6) {
  if (lambda >= 0.999999)
    return Eval::Virial(p, distSq, i);
  const SoftCoreDist sc =
      SoftenedDistance(distSq, p.sigmaSq[i], lambda, scAlpha, scPower, scSigma6);
  const double correction = distSq / sc.softRsq;
  return lambda * correction * correction * Eval::Virial(p, sc.softRsq, i);
}

// ------------------------------------------------------- free energy ----
//
// dE/dlambda for the soft-core path, parameterised on the evaluator so each
// forcefield gets its own while the derivative itself is written once. It was
// five copies, differing only in which CalcEn/CalcVir they reached by virtual
// dispatch.
//
//   dE/dlambda = U(softRsq) + fCoef * Vir(softRsq)
//   fCoef      = lambda * sc_alpha * sc_power/6
//                * (1-lambda)^(sc_power-1) * sigma6 / softRsq^2
//
// Verified against a numerical derivative of the energy, not just transcribed.
//
template <class Eval>
GOMC_HOSTDEV inline double
DEnergyDLambda(const VdwParams &p, const double distSq, const uint i,
               const double lambda, const double scAlpha, const uint scPower,
               const double scSigma6) {
  const SoftCoreDist sc =
      SoftenedDistance(distSq, p.sigmaSq[i], lambda, scAlpha, scPower, scSigma6);
  double fCoef = lambda * scAlpha * scPower / 6.0;
  fCoef *= pow(1.0 - lambda, scPower - 1.0) * sc.sigma6 /
           (sc.softRsq * sc.softRsq);
  return Eval::Energy(p, sc.softRsq, i) + fCoef * Eval::Virial(p, sc.softRsq, i);
}

} // namespace ff

#endif /*VDW_EVALUATORS_H*/

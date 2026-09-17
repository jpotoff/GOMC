/*******************************************************************************
Analytic reference tests for every van der Waals forcefield.

These state each potential's closed form independently and compare the kernel
against it. That is deliberately different from MieExponentTest, which compares
a kernel's fast path against its own fallback -- a self-comparison that passes
even if both paths are wrong in the same way.

This is the gate for restructuring the forcefields: FF_EXP6 previously had no
test coverage at all, and the switch/Martini variants had only self-comparison,
so a refactor of the pair math had nothing to check it against.

Derived parameters (shiftConst, factor1/2, expConst, rMin, the Martini
switching constants) are read back from the object rather than recomputed,
because some are produced by root-finding at Init time. What is pinned here is
the *energy expression*, which is what a refactor moves.
*******************************************************************************/

#include <cmath>
#include <gtest/gtest.h>

#include "FFExp6.h"
#include "FFParticle.h"
#include "FFVdwStd.h"
#include "FFSetup.h"
#include "FFShift.h"
#include "FFSwitch.h"
#include "FFSwitchMartini.h"
#include "Forcefield.h"
#include "NumLib.h"

//
// Reads the derived parameters the closed forms need.
//
struct FFTestAccess {
  // shared, declared on FFParticle
  static double SigmaSq(const FFParticle &p, uint i) { return p.sigmaSq[i]; }
  static double EpsilonCn(const FFParticle &p, uint i) { return p.epsilon_cn[i]; }
  static double N(const FFParticle &p, uint i) { return p.n[i]; }
  // per-flavor
  static double ShiftConst(const FF_SHIFT &p, uint i) { return p.shiftConst[i]; }
  // 1-4 variants: separate arrays, so a mix-up with the ordinary ones is a
  // real failure mode the tests must be able to see.
  static double SigmaSq14(const FFParticle &p, uint i) { return p.sigmaSq_1_4[i]; }
  static double EpsilonCn14(const FFParticle &p, uint i) { return p.epsilon_cn_1_4[i]; }
  static double N14(const FFParticle &p, uint i) { return p.n_1_4[i]; }
  static double ShiftConst14(const FF_SHIFT &p, uint i) { return p.shiftConst_1_4[i]; }
  static double Factor1(const FF_SWITCH &p) { return p.factor1; }
  static double Factor2(const FF_SWITCH &p) { return p.factor2; }
  static double ROnSq(const FF_SWITCH &p) { return p.rOnSq; }
  static double ExpConst(const FF_EXP6 &p, uint i) { return p.expConst[i]; }
  static double RMin(const FF_EXP6 &p, uint i) { return p.rMin[i]; }
  static double RMaxSq(const FF_EXP6 &p, uint i) { return p.rMaxSq[i]; }
};

namespace {

const double kSigma = 3.740;
const double kEpsilon = 161.0;
const double kRCut = 14.0;
const double kRSwitch = 10.0;

// Separations spanning the repulsive wall, the well, the switching region and
// the tail. All inside rCut so nothing is short-circuited to zero.
const double kDist[] = {4.0, 4.5, 5.0, 6.0, 8.0, 9.5, 10.5, 11.0, 12.0, 13.5};
const int kNumDist = sizeof(kDist) / sizeof(kDist[0]);

const double kTol = 1e-12; // relative; these are the same operations reordered

void InitForcefield(Forcefield &ff, bool martini = false) {
  ff.rCut = kRCut;
  ff.rCutSq = kRCut * kRCut;
  ff.rCutLow = 0.0;
  ff.rCutLowSq = 0.0;
  ff.rswitch = kRSwitch;
  ff.freeEnergy = false;
  ff.sc_coul = false;
  ff.sc_alpha = 0.5;
  ff.sc_sigma = 3.0;
  ff.sc_sigma_6 = 729.0;
  ff.sc_power = 2;
  ff.vdwGeometricSigma = false;
  ff.isMartini = martini;
  ff.exp6 = false;
  ff.dielectric = 1.0;
  ff.electrostatic = false;
  ff.ewald = false;
  // The Coulomb kernels guard on rCutCoulombSq; leaving it zero makes every
  // call return 0.0 and any comparison against it vacuous.
  for (uint b = 0; b < BOX_TOTAL; ++b) {
    ff.rCutCoulomb[b] = kRCut;
    ff.rCutCoulombSq[b] = kRCut * kRCut;
    ff.alpha[b] = 0.0;
    ff.alphaSq[b] = 0.0;
  }
}

// One kind, run through the production Init/Blend path.
//
// The 1-4 parameters are deliberately DIFFERENT from the ordinary ones. If they
// matched, a kernel that read sigmaSq where it meant sigmaSq_1_4 would still
// pass, and that is exactly the kind of mix-up these tests exist to catch.
void InitOneKind(FFParticle &p, double nVal, double sigma = kSigma,
                 double eps = kEpsilon) {
  ff_setup::Particle mie;
  mie.setIsCHARMM(false);
  mie.getnamelist().push_back("TST");
  mie.epsilon.push_back(eps);
  mie.sigma.push_back(sigma);
  mie.n.push_back(nVal);
  mie.epsilon_1_4.push_back(eps * 0.5);      // distinct on purpose
  mie.sigma_1_4.push_back(sigma * 0.9);
  mie.n_1_4.push_back(nVal);
  ff_setup::NBfix nbfix;
  p.Init(mie, nbfix);
}

// U_Mie using the 1-4 parameter set.
double AnalyticMie14(const FFParticle &p, double r, uint idx) {
  const double sr2 = FFTestAccess::SigmaSq14(p, idx) / (r * r);
  const double n = FFTestAccess::N14(p, idx);
  return FFTestAccess::EpsilonCn14(p, idx) *
         (std::pow(sr2, n * 0.5) - std::pow(sr2, 3.0));
}

// U_Mie(r) = epsilon_cn * [ (sigma/r)^n - (sigma/r)^6 ]
double AnalyticMie(const FFParticle &p, double r, uint idx) {
  const double sr2 = FFTestAccess::SigmaSq(p, idx) / (r * r);
  const double n = FFTestAccess::N(p, idx);
  return FFTestAccess::EpsilonCn(p, idx) *
         (std::pow(sr2, n * 0.5) - std::pow(sr2, 3.0));
}

} // namespace

//
// Plain Mie / LJ. n is swept, not fixed at 12, because the exponent fast path
// is only exercised away from the hand-expanded n=12 case.
//
TEST(ForcefieldAnalyticTest, MieMatchesClosedForm) {
  const double exponents[] = {7.0, 12.0, 14.0, 16.0, 20.0, 12.5};
  for (int e = 0; e < 6; ++e) {
    Forcefield ff;
    InitForcefield(ff);
    FF_VDW_STD p(ff);
    InitOneKind(p, exponents[e]);

    for (int d = 0; d < kNumDist; ++d) {
      const double r = kDist[d];
      const double want = AnalyticMie(p, r, 0);
      const double got = p.CalcEn(r * r, 0, 0, 1.0);
      EXPECT_NEAR(got, want, kTol * std::fabs(want))
          << "n=" << exponents[e] << " r=" << r;
    }
  }
}

//
// Shifted: the Mie energy less a constant chosen so U(rCut) == 0.
//
TEST(ForcefieldAnalyticTest, ShiftMatchesClosedForm) {
  const double exponents[] = {12.0, 16.0, 12.5};
  for (int e = 0; e < 3; ++e) {
    Forcefield ff;
    InitForcefield(ff);
    FF_SHIFT p(ff);
    InitOneKind(p, exponents[e]);

    for (int d = 0; d < kNumDist; ++d) {
      const double r = kDist[d];
      const double want = AnalyticMie(p, r, 0) - FFTestAccess::ShiftConst(p, 0);
      const double got = p.CalcEn(r * r, 0, 0, 1.0);
      EXPECT_NEAR(got, want, kTol * std::fabs(want))
          << "n=" << exponents[e] << " r=" << r;
    }
  }
}

//
// The shift really must vanish at the cutoff -- that is the point of it, and it
// is independent of how shiftConst is stored.
//
TEST(ForcefieldAnalyticTest, ShiftIsZeroAtCutoff) {
  Forcefield ff;
  InitForcefield(ff);
  FF_SHIFT p(ff);
  InitOneKind(p, 12.0);
  const double justInside = kRCut - 1e-9;
  EXPECT_NEAR(p.CalcEn(justInside * justInside, 0, 0, 1.0), 0.0, 1e-9);
}

//
// Switched: Mie multiplied by a smooth factor that is 1 below rSwitch and
// falls to 0 at rCut.
//
TEST(ForcefieldAnalyticTest, SwitchMatchesClosedForm) {
  const double exponents[] = {12.0, 16.0};
  for (int e = 0; e < 2; ++e) {
    Forcefield ff;
    InitForcefield(ff);
    FF_SWITCH p(ff);
    InitOneKind(p, exponents[e]);

    const double f1 = FFTestAccess::Factor1(p);
    const double f2 = FFTestAccess::Factor2(p);
    const double rOnSq = FFTestAccess::ROnSq(p);

    for (int d = 0; d < kNumDist; ++d) {
      const double r = kDist[d], rSq = r * r;
      const double d2 = ff.rCutSq - rSq;
      const double factE = (rSq > rOnSq) ? d2 * d2 * f2 * (f1 + 2.0 * rSq) : 1.0;
      const double want = AnalyticMie(p, r, 0) * factE;
      const double got = p.CalcEn(rSq, 0, 0, 1.0);
      EXPECT_NEAR(got, want, kTol * std::fabs(want))
          << "n=" << exponents[e] << " r=" << r;
    }
  }
}

//
// The switching function's defining properties, checked independently of the
// stored constants: untouched below rSwitch, zero at rCut.
//
TEST(ForcefieldAnalyticTest, SwitchIsIdentityBelowRSwitchAndZeroAtCutoff) {
  Forcefield ff;
  InitForcefield(ff);
  FF_SWITCH p(ff);
  InitOneKind(p, 12.0);

  for (double r = 4.0; r < kRSwitch; r += 1.5) {
    EXPECT_NEAR(p.CalcEn(r * r, 0, 0, 1.0), AnalyticMie(p, r, 0),
                kTol * std::fabs(AnalyticMie(p, r, 0)))
        << "switch must be inactive below rSwitch, r=" << r;
  }
  const double justInside = kRCut - 1e-9;
  EXPECT_NEAR(p.CalcEn(justInside * justInside, 0, 0, 1.0), 0.0, 1e-6);
}

//
// Exp-6. Previously untested entirely.
//   U(r) = expConst * [ (6/alpha) * exp(alpha * (1 - r/rMin)) - (rMin/r)^6 ]
// with a hard wall below rMax.
//
TEST(ForcefieldAnalyticTest, Exp6MatchesClosedForm) {
  const double alphas[] = {13.0, 15.0, 16.0};
  for (int a = 0; a < 3; ++a) {
    Forcefield ff;
    InitForcefield(ff);
    ff.exp6 = true;
    FF_EXP6 p(ff);
    InitOneKind(p, alphas[a]);

    const double expConst = FFTestAccess::ExpConst(p, 0);
    const double rMin = FFTestAccess::RMin(p, 0);
    const double alpha = FFTestAccess::N(p, 0);
    const double rMaxSq = FFTestAccess::RMaxSq(p, 0);

    ASSERT_GT(rMin, 0.0) << "rMin root-finding failed for alpha=" << alphas[a];

    for (int d = 0; d < kNumDist; ++d) {
      const double r = kDist[d], rSq = r * r;
      if (rSq < rMaxSq)
        continue; // inside the hard wall; covered separately
      const double rRat = rMin / r;
      const double attract = rRat * rRat * rRat * rRat * rRat * rRat;
      const double repulse =
          (6.0 / alpha) * std::exp(alpha * (1.0 - r / rMin));
      const double want = expConst * (repulse - attract);
      const double got = p.CalcEn(rSq, 0, 0, 1.0);
      EXPECT_NEAR(got, want, kTol * std::fabs(want))
          << "alpha=" << alphas[a] << " r=" << r;
    }
  }
}

//
// Exp-6 turns over at small r, so GOMC guards it with a hard wall. Without this
// the potential would become attractive as r -> 0 and swallow molecules.
//
TEST(ForcefieldAnalyticTest, Exp6HasHardWallBelowRMax) {
  Forcefield ff;
  InitForcefield(ff);
  ff.exp6 = true;
  FF_EXP6 p(ff);
  InitOneKind(p, 15.0);

  const double rMaxSq = FFTestAccess::RMaxSq(p, 0);
  ASSERT_GT(rMaxSq, 0.0);
  EXPECT_GE(p.CalcEn(rMaxSq * 0.5, 0, 0, 1.0), num::BIGNUM)
      << "must repel hard inside rMax";
  EXPECT_LT(p.CalcEn(rMaxSq * 1.5, 0, 0, 1.0), num::BIGNUM)
      << "must be finite outside rMax";
}

//
// Energy must be continuous where a variant switches behaviour, and must vanish
// beyond the cutoff for every flavor.
//
TEST(ForcefieldAnalyticTest, AllFlavorsVanishBeyondCutoff) {
  const double beyond = (kRCut + 1.0) * (kRCut + 1.0);
  {
    Forcefield ff; InitForcefield(ff); FF_VDW_STD p(ff); InitOneKind(p, 12.0);
    EXPECT_EQ(p.CalcEn(beyond, 0, 0, 1.0), 0.0) << "Mie";
  }
  {
    Forcefield ff; InitForcefield(ff); FF_SHIFT p(ff); InitOneKind(p, 12.0);
    EXPECT_EQ(p.CalcEn(beyond, 0, 0, 1.0), 0.0) << "shift";
  }
  {
    Forcefield ff; InitForcefield(ff); FF_SWITCH p(ff); InitOneKind(p, 12.0);
    EXPECT_EQ(p.CalcEn(beyond, 0, 0, 1.0), 0.0) << "switch";
  }
  {
    Forcefield ff; InitForcefield(ff); ff.exp6 = true;
    FF_EXP6 p(ff); InitOneKind(p, 15.0);
    EXPECT_EQ(p.CalcEn(beyond, 0, 0, 1.0), 0.0) << "exp6";
  }
}

//
// The soft-core free-energy path must reduce to the ordinary potential at
// lambda = 1 for every flavor. This block is currently duplicated ~30 times
// across the forcefield files, so it needs pinning before it is unified.
//
TEST(ForcefieldAnalyticTest, SoftCoreReducesToPlainAtLambdaOne) {
  {
    Forcefield ff; InitForcefield(ff); FF_VDW_STD p(ff); InitOneKind(p, 16.0);
    for (int d = 0; d < kNumDist; ++d) {
      const double rSq = kDist[d] * kDist[d];
      EXPECT_NEAR(p.CalcEn(rSq, 0, 0, 1.0), AnalyticMie(p, kDist[d], 0),
                  kTol * std::fabs(AnalyticMie(p, kDist[d], 0)));
    }
  }
  {
    Forcefield ff; InitForcefield(ff); FF_SHIFT p(ff); InitOneKind(p, 16.0);
    for (int d = 0; d < kNumDist; ++d) {
      const double rSq = kDist[d] * kDist[d];
      const double want = AnalyticMie(p, kDist[d], 0) -
                          FFTestAccess::ShiftConst(p, 0);
      EXPECT_NEAR(p.CalcEn(rSq, 0, 0, 1.0), want, kTol * std::fabs(want));
    }
  }
}

//
// Soft core must be continuous as lambda approaches 1, so a free-energy
// calculation does not see a discontinuity at the end point.
//
TEST(ForcefieldAnalyticTest, SoftCoreIsContinuousApproachingLambdaOne) {
  Forcefield ff;
  InitForcefield(ff);
  ff.freeEnergy = true;
  FF_VDW_STD p(ff);
  InitOneKind(p, 12.0);

  const double rSq = 5.0 * 5.0;
  const double atOne = p.CalcEn(rSq, 0, 0, 1.0);
  double prev = p.CalcEn(rSq, 0, 0, 0.99);
  for (double lambda : {0.999, 0.9999, 0.99999}) {
    const double cur = p.CalcEn(rSq, 0, 0, lambda);
    EXPECT_LT(std::fabs(cur - atOne), std::fabs(prev - atOne) + 1e-12)
        << "not converging to the lambda=1 value at lambda=" << lambda;
    prev = cur;
  }
}

// ===========================================================================
// 1-4 interactions.
//
// These use a SEPARATE parameter set (sigmaSq_1_4, epsilon_cn_1_4, n_1_4,
// nExp_1_4) and a separate copy of the same arithmetic, so they can drift from
// the ordinary path independently -- and nothing exercised them until now. The
// exponent bug lived in these copies too.
//
// Note the interface difference: these ACCUMULATE into `en` rather than
// returning, which is its own failure mode.
// ===========================================================================

TEST(ForcefieldAnalyticTest, Mie14MatchesClosedForm) {
  const double exponents[] = {12.0, 16.0, 12.5};
  for (int e = 0; e < 3; ++e) {
    Forcefield ff;
    InitForcefield(ff);
    FF_VDW_STD p(ff);
    InitOneKind(p, exponents[e]);

    for (int d = 0; d < kNumDist; ++d) {
      const double r = kDist[d];
      const double want = AnalyticMie14(p, r, 0);
      double got = 0.0;
      p.CalcAdd_1_4(got, r * r, 0, 0);
      EXPECT_NEAR(got, want, kTol * std::fabs(want))
          << "n=" << exponents[e] << " r=" << r;
    }
  }
}

TEST(ForcefieldAnalyticTest, Shift14MatchesClosedForm) {
  Forcefield ff;
  InitForcefield(ff);
  FF_SHIFT p(ff);
  InitOneKind(p, 16.0);

  for (int d = 0; d < kNumDist; ++d) {
    const double r = kDist[d];
    const double want = AnalyticMie14(p, r, 0) - FFTestAccess::ShiftConst14(p, 0);
    double got = 0.0;
    p.CalcAdd_1_4(got, r * r, 0, 0);
    EXPECT_NEAR(got, want, kTol * std::fabs(want)) << "r=" << r;
  }
}

//
// The 1-4 path must read its OWN parameters. With sigma_1_4 != sigma and
// epsilon_1_4 != epsilon (see InitOneKind), reading the ordinary arrays by
// mistake produces a visibly different number.
//
TEST(ForcefieldAnalyticTest, OneFourUsesItsOwnParameters) {
  Forcefield ff;
  InitForcefield(ff);
  FF_VDW_STD p(ff);
  InitOneKind(p, 12.0);

  const double r = 5.0;
  double got14 = 0.0;
  p.CalcAdd_1_4(got14, r * r, 0, 0);
  const double gotOrdinary = p.CalcEn(r * r, 0, 0, 1.0);

  EXPECT_NEAR(got14, AnalyticMie14(p, r, 0),
              kTol * std::fabs(AnalyticMie14(p, r, 0)));
  EXPECT_GT(std::fabs(got14 - gotOrdinary), 1e-6)
      << "1-4 and ordinary parameters are distinct; results must differ";
}

//
// It accumulates. Overwriting instead of adding would silently discard every
// earlier contribution in the caller's loop.
//
TEST(ForcefieldAnalyticTest, OneFourAccumulatesRatherThanOverwrites) {
  Forcefield ff;
  InitForcefield(ff);
  FF_VDW_STD p(ff);
  InitOneKind(p, 12.0);

  const double r = 5.0, seed = 123.456;
  double fresh = 0.0;
  p.CalcAdd_1_4(fresh, r * r, 0, 0);

  double accumulated = seed;
  p.CalcAdd_1_4(accumulated, r * r, 0, 0);
  EXPECT_NEAR(accumulated, seed + fresh, 1e-9 * std::fabs(seed + fresh));

  // ...and twice adds twice.
  double twice = 0.0;
  p.CalcAdd_1_4(twice, r * r, 0, 0);
  p.CalcAdd_1_4(twice, r * r, 0, 0);
  EXPECT_NEAR(twice, 2.0 * fresh, 1e-9 * std::fabs(2.0 * fresh));
}

//
// Beyond the cutoff it must contribute nothing -- not even change `en`.
//
TEST(ForcefieldAnalyticTest, OneFourIsSilentBeyondCutoff) {
  Forcefield ff;
  InitForcefield(ff);
  FF_VDW_STD p(ff);
  InitOneKind(p, 12.0);

  const double beyond = (kRCut + 1.0) * (kRCut + 1.0);
  double en = 77.0;
  p.CalcAdd_1_4(en, beyond, 0, 0);
  EXPECT_EQ(en, 77.0) << "must leave en untouched beyond rCut";
}

//
// The 1-4 electrostatic form: qq/r when NB, scaled by 1-4scaling otherwise.
//
TEST(ForcefieldAnalyticTest, CoulombOneFourAppliesScaling) {
  Forcefield ff;
  InitForcefield(ff);
  ff.scaling_14 = 0.5;
  FF_VDW_STD p(ff);
  InitOneKind(p, 12.0);

  const double qq = 332.0;
  for (int d = 0; d < kNumDist; ++d) {
    const double r = kDist[d];

    double nb = 0.0;
    p.CalcCoulombAdd_1_4(nb, r * r, qq, true);
    EXPECT_NEAR(nb, qq / r, kTol * std::fabs(qq / r)) << "NB, r=" << r;

    double scaled = 0.0;
    p.CalcCoulombAdd_1_4(scaled, r * r, qq, false);
    EXPECT_NEAR(scaled, qq * ff.scaling_14 / r,
                kTol * std::fabs(qq * ff.scaling_14 / r))
        << "1-4 scaled, r=" << r;
  }

  // and silent beyond the cutoff
  double en = 9.0;
  p.CalcCoulombAdd_1_4(en, (kRCut + 1.0) * (kRCut + 1.0), qq, true);
  EXPECT_EQ(en, 9.0);
}

// ===========================================================================
// Soft core at lambda < 1 -- the branch a normal simulation never takes, and
// therefore the one a bit-for-bit comparison of ordinary output cannot check.
//
//   sigma6     = max(sigmaSq^3, sc_sigma_6)
//   lambdaCoef = sc_alpha * (1 - lambda)^sc_power
//   softRsq    = cbrt(lambdaCoef * sigma6 + distSq^3)
//   U          = lambda * U_plain(softRsq)
// ===========================================================================

namespace {
double SoftRsqReference(const Forcefield &ff, double sigmaSqIJ, double distSq,
                        double lambda) {
  double sigma6 = sigmaSqIJ * sigmaSqIJ * sigmaSqIJ;
  sigma6 = std::max(sigma6, ff.sc_sigma_6);
  const double dist6 = distSq * distSq * distSq;
  const double lambdaCoef = ff.sc_alpha * std::pow(1.0 - lambda, ff.sc_power);
  return std::cbrt(lambdaCoef * sigma6 + dist6);
}
} // namespace

TEST(ForcefieldAnalyticTest, SoftCoreMatchesClosedFormBelowLambdaOne) {
  const double lambdas[] = {0.1, 0.3, 0.5, 0.75, 0.9, 0.99};
  const double exponents[] = {12.0, 16.0};
  for (int e = 0; e < 2; ++e) {
    Forcefield ff;
    InitForcefield(ff);
    ff.freeEnergy = true;
    FF_VDW_STD p(ff);
    InitOneKind(p, exponents[e]);
    const double sSq = FFTestAccess::SigmaSq(p, 0);

    for (int l = 0; l < 6; ++l) {
      for (int d = 0; d < kNumDist; ++d) {
        const double distSq = kDist[d] * kDist[d];
        const double softRsq = SoftRsqReference(ff, sSq, distSq, lambdas[l]);
        // U_plain evaluated at the SOFTENED separation, scaled by lambda
        const double want =
            lambdas[l] * AnalyticMie(p, std::sqrt(softRsq), 0);
        const double got = p.CalcEn(distSq, 0, 0, lambdas[l]);
        EXPECT_NEAR(got, want, kTol * std::fabs(want))
            << "n=" << exponents[e] << " lambda=" << lambdas[l]
            << " r=" << kDist[d];
      }
    }
  }
}

TEST(ForcefieldAnalyticTest, SoftCoreShiftMatchesClosedFormBelowLambdaOne) {
  Forcefield ff;
  InitForcefield(ff);
  ff.freeEnergy = true;
  FF_SHIFT p(ff);
  InitOneKind(p, 12.0);
  const double sSq = FFTestAccess::SigmaSq(p, 0);

  for (double lambda : {0.25, 0.5, 0.9}) {
    for (int d = 0; d < kNumDist; ++d) {
      const double distSq = kDist[d] * kDist[d];
      const double softRsq = SoftRsqReference(ff, sSq, distSq, lambda);
      const double want =
          lambda * (AnalyticMie(p, std::sqrt(softRsq), 0) -
                    FFTestAccess::ShiftConst(p, 0));
      const double got = p.CalcEn(distSq, 0, 0, lambda);
      EXPECT_NEAR(got, want, kTol * std::fabs(want))
          << "lambda=" << lambda << " r=" << kDist[d];
    }
  }
}

//
// sc_sigma_6 is a floor on sigma^6, so a tiny sigma must still be softened by
// at least that much. Without the clamp the soft core stops protecting small
// particles.
//
TEST(ForcefieldAnalyticTest, SoftCoreClampsSmallSigma) {
  Forcefield ff;
  InitForcefield(ff);
  ff.freeEnergy = true;
  ff.sc_sigma_6 = 1.0e6; // far above sigma^6, so the clamp must bind
  FF_VDW_STD p(ff);
  InitOneKind(p, 12.0);
  const double sSq = FFTestAccess::SigmaSq(p, 0);
  ASSERT_LT(sSq * sSq * sSq, ff.sc_sigma_6) << "clamp would not bind";

  const double distSq = 25.0, lambda = 0.5;
  const double softRsq = SoftRsqReference(ff, sSq, distSq, lambda);
  const double want = lambda * AnalyticMie(p, std::sqrt(softRsq), 0);
  EXPECT_NEAR(p.CalcEn(distSq, 0, 0, lambda), want, kTol * std::fabs(want));
}

//
// The soft core must actually soften: at lambda < 1 the pair is evaluated
// further apart than it really is, so the repulsion is finite where the plain
// potential would diverge.
//
TEST(ForcefieldAnalyticTest, SoftCoreIsFiniteAtContact) {
  Forcefield ff;
  InitForcefield(ff);
  ff.freeEnergy = true;
  FF_VDW_STD p(ff);
  InitOneKind(p, 12.0);

  const double tiny = 1e-8; // essentially overlapping
  const double soft = p.CalcEn(tiny, 0, 0, 0.5);
  const double hard = p.CalcEn(tiny, 0, 0, 1.0);
  EXPECT_TRUE(std::isfinite(soft)) << "soft core must stay finite at contact";
  EXPECT_LT(soft, hard) << "lambda<1 must be less repulsive than lambda=1";
}

// ===========================================================================
// The point of the evaluator split: a new truncation is a few lines and works
// with every core, rather than a new 175-line class (cf. commit b1d7ec92,
// which added force-shifted VdW in 280 lines across 10 files).
//
// This is force-shifted VdW, defined here in the test rather than in the
// library, precisely to show that nothing in the library has to change.
// ===========================================================================
namespace {

struct TestForceShiftTrunc {
  //! plain, less the value AND the slope at the cutoff
  static double Energy(const ff::VdwParams &p, const ff::MieTerms &t,
                       const double distSq, const uint i) {
    return p.epsilon_cn[i] * (t.repulse - t.attract) - p.shiftConst[i];
  }
};

// Note it reuses the library's PlainVir: a new truncation of the energy does
// not require reimplementing the virial.
typedef ff::PairEval<ff::MieCore, TestForceShiftTrunc, ff::PlainVir>
    TestMieForceShift;

} // namespace

TEST(ForcefieldAnalyticTest, NewTruncationComposesWithExistingCore) {
  Forcefield ff;
  InitForcefield(ff);
  FF_SHIFT p(ff); // reuse its parameters, incl. shiftConst
  InitOneKind(p, 16.0);

  const ff::VdwParams vp = p.VdwView();
  for (int d = 0; d < kNumDist; ++d) {
    const double r = kDist[d], rSq = r * r;
    // the new evaluator, composed from an existing core and a new truncation
    const double got = TestMieForceShift::Energy(vp, rSq, 0);
    const double want = AnalyticMie(p, r, 0) - FFTestAccess::ShiftConst(p, 0);
    EXPECT_NEAR(got, want, kTol * std::fabs(want)) << "r=" << r;
  }
}

//
// The library's own compositions must agree with the classes that delegate to
// them -- i.e. the evaluators really are what the forcefields compute.
//
TEST(ForcefieldAnalyticTest, EvaluatorsAgreeWithTheirForcefields) {
  {
    Forcefield ff; InitForcefield(ff); FF_VDW_STD p(ff); InitOneKind(p, 16.0);
    for (int d = 0; d < kNumDist; ++d) {
      const double rSq = kDist[d] * kDist[d];
      EXPECT_EQ(ff::MiePlain::Energy(p.VdwView(), rSq, 0),
                p.CalcEn(rSq, 0, 0, 1.0)) << "MiePlain vs FFParticle";
    }
  }
  {
    Forcefield ff; InitForcefield(ff); FF_SHIFT p(ff); InitOneKind(p, 16.0);
    for (int d = 0; d < kNumDist; ++d) {
      const double rSq = kDist[d] * kDist[d];
      EXPECT_EQ(ff::MieShift::Energy(p.VdwView(), rSq, 0),
                p.CalcEn(rSq, 0, 0, 1.0)) << "MieShift vs FF_SHIFT";
    }
  }
  {
    Forcefield ff; InitForcefield(ff); FF_SWITCH p(ff); InitOneKind(p, 16.0);
    for (int d = 0; d < kNumDist; ++d) {
      const double rSq = kDist[d] * kDist[d];
      EXPECT_EQ(ff::MieSwitch::Energy(p.VdwView(), rSq, 0),
                p.CalcEn(rSq, 0, 0, 1.0)) << "MieSwitch vs FF_SWITCH";
    }
  }
  {
    Forcefield ff; InitForcefield(ff); ff.exp6 = true;
    FF_EXP6 p(ff); InitOneKind(p, 15.0);
    const double rMaxSq = FFTestAccess::RMaxSq(p, 0);
    for (int d = 0; d < kNumDist; ++d) {
      const double rSq = kDist[d] * kDist[d];
      if (rSq < rMaxSq) continue;
      EXPECT_EQ(ff::Exp6Eval::Energy(p.VdwView(), rSq, 0),
                p.CalcEn(rSq, 0, 0, 1.0)) << "Exp6Eval vs FF_EXP6";
    }
  }
}

// ===========================================================================
// Virial.
//
// GOMC's convention (stated in FFParticle::CalcVir) is
//
//     Vir(r) = F.r / r^2 = -(1/r) dU/dr
//
// so the strongest available check is against a central difference of CalcEn,
// not against a transcribed formula. A copied-and-edited virial expression can
// agree with a copied-and-edited closed form while both are wrong; it cannot
// agree with the derivative of the energy the code actually computes.
// ===========================================================================
namespace {

//! -(1/r) dU/dr by central difference on whatever CalcEn returns.
template <class FF>
double VirialByFiniteDifference(const FF &p, double r) {
  const double h = 1e-5 * r;
  const double uPlus = p.CalcEn((r + h) * (r + h), 0, 0, 1.0);
  const double uMinus = p.CalcEn((r - h) * (r - h), 0, 0, 1.0);
  return -(uPlus - uMinus) / (2.0 * h) / r;
}

// Away from the steep repulsive wall, where a central difference is accurate.
const double kVirDist[] = {4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 12.0};
const int kNumVirDist = sizeof(kVirDist) / sizeof(kVirDist[0]);
const double kFdTol = 1e-5; // central-difference truncation, not code error

} // namespace

TEST(ForcefieldAnalyticTest, MieVirialIsNegativeEnergyDerivative) {
  const double exponents[] = {12.0, 16.0};
  for (int e = 0; e < 2; ++e) {
    Forcefield ff;
    InitForcefield(ff);
    FF_VDW_STD p(ff);
    InitOneKind(p, exponents[e]);
    for (int d = 0; d < kNumVirDist; ++d) {
      const double r = kVirDist[d];
      const double want = VirialByFiniteDifference(p, r);
      const double got = p.CalcVir(r * r, 0, 0, 1.0);
      EXPECT_NEAR(got, want, kFdTol * std::fabs(want) + 1e-12)
          << "n=" << exponents[e] << " r=" << r;
    }
  }
}

//
// The Mie virial in closed form, independent of the finite difference:
//   Vir = epsilon_cn * (n * repulse - 6 * attract) / r^2
//
TEST(ForcefieldAnalyticTest, MieVirialMatchesClosedForm) {
  const double exponents[] = {12.0, 16.0, 12.5};
  for (int e = 0; e < 3; ++e) {
    Forcefield ff;
    InitForcefield(ff);
    FF_VDW_STD p(ff);
    InitOneKind(p, exponents[e]);
    const double sSq = FFTestAccess::SigmaSq(p, 0);
    const double ecn = FFTestAccess::EpsilonCn(p, 0);
    const double n = FFTestAccess::N(p, 0);

    for (int d = 0; d < kNumVirDist; ++d) {
      const double r = kVirDist[d], rSq = r * r;
      const double sr2 = sSq / rSq;
      const double attract = std::pow(sr2, 3.0);
      const double repulse = std::pow(sr2, n * 0.5);
      const double want = ecn * (n * repulse - 6.0 * attract) / rSq;
      const double got = p.CalcVir(rSq, 0, 0, 1.0);
      EXPECT_NEAR(got, want, 1e-10 * std::fabs(want))
          << "n=" << exponents[e] << " r=" << r;
    }
  }
}

//
// Shifting the energy by a constant cannot change the force, so FF_SHIFT's
// virial must equal the plain one. This is a property the code should have, not
// a formula it happens to contain.
//
TEST(ForcefieldAnalyticTest, ShiftDoesNotChangeTheVirial) {
  Forcefield ffA, ffB;
  InitForcefield(ffA);
  InitForcefield(ffB);
  FF_VDW_STD plain(ffA);
  FF_SHIFT shifted(ffB);
  InitOneKind(plain, 16.0);
  InitOneKind(shifted, 16.0);

  for (int d = 0; d < kNumVirDist; ++d) {
    const double rSq = kVirDist[d] * kVirDist[d];
    EXPECT_NEAR(shifted.CalcVir(rSq, 0, 0, 1.0), plain.CalcVir(rSq, 0, 0, 1.0),
                1e-12 * std::fabs(plain.CalcVir(rSq, 0, 0, 1.0)))
        << "a constant shift must not change the force, r=" << kVirDist[d];
  }
}

//
// The switched virial must account for the derivative of the switching
// function, not just the potential -- the case a transcribed formula gets
// wrong most easily.
//
TEST(ForcefieldAnalyticTest, SwitchVirialIsNegativeEnergyDerivative) {
  Forcefield ff;
  InitForcefield(ff);
  FF_SWITCH p(ff);
  InitOneKind(p, 12.0);

  for (int d = 0; d < kNumVirDist; ++d) {
    const double r = kVirDist[d];
    const double want = VirialByFiniteDifference(p, r);
    const double got = p.CalcVir(r * r, 0, 0, 1.0);
    EXPECT_NEAR(got, want, kFdTol * std::fabs(want) + 1e-12)
        << "r=" << r << (r * r > kRSwitch * kRSwitch ? "  (inside switch)" : "");
  }
}

TEST(ForcefieldAnalyticTest, Exp6VirialIsNegativeEnergyDerivative) {
  Forcefield ff;
  InitForcefield(ff);
  ff.exp6 = true;
  FF_EXP6 p(ff);
  InitOneKind(p, 15.0);
  const double rMaxSq = FFTestAccess::RMaxSq(p, 0);

  for (int d = 0; d < kNumVirDist; ++d) {
    const double r = kVirDist[d];
    if (r * r < rMaxSq * 1.05)
      continue; // keep the stencil clear of the hard wall
    const double want = VirialByFiniteDifference(p, r);
    const double got = p.CalcVir(r * r, 0, 0, 1.0);
    EXPECT_NEAR(got, want, kFdTol * std::fabs(want) + 1e-12) << "r=" << r;
  }
}

// ===========================================================================
// Electrostatics.
//
// Two distinct situations, and they factor differently:
//
//   Ewald on  -- every flavor computes the SAME thing, qq * erfc(alpha*r)/r
//                (now via the tabulated kernel). Ten identical copies.
//   Ewald off -- the flavors genuinely differ: plain qq/r, shifted
//                qq(1/r - 1/rCut), switched, and Martini's own form.
//
// So the Ewald branch is duplication to remove; the rest is real physics.
// ===========================================================================
namespace {

void InitForcefieldEwald(Forcefield &ff) {
  InitForcefield(ff);
  ff.electrostatic = true;
  ff.ewald = true;
  ff.tolerance = 1e-5;
  for (uint b = 0; b < BOX_TOTAL; ++b) {
    ff.rCutCoulomb[b] = kRCut;
    ff.rCutCoulombSq[b] = kRCut * kRCut;
    ff.alpha[b] = std::sqrt(-std::log(ff.tolerance)) / kRCut;
    ff.alphaSq[b] = ff.alpha[b] * ff.alpha[b];
  }
  ff.realTable.Init(ff.alpha, ff.rCutCoulombSq, ff.rCutLowSq);
}

const double kQQ = 332.0636;   // e^2/(4 pi eps0) in GOMC's units, roughly
// The tabulated kernel is accurate to ~1e-7 relative by construction.
const double kCoulTol = 1e-6;

} // namespace

TEST(ForcefieldAnalyticTest, EwaldCoulombMatchesErfcOverR) {
  Forcefield ff;
  InitForcefieldEwald(ff);
  FF_VDW_STD p(ff);
  InitOneKind(p, 12.0);
  const double alpha = ff.alpha[0];

  for (int d = 0; d < kNumDist; ++d) {
    const double r = kDist[d];
    const double want = kQQ * std::erfc(alpha * r) / r;
    const double got = p.CalcCoulomb(r * r, 0, 0, kQQ, 1.0, 0);
    EXPECT_NEAR(got, want, kCoulTol * std::fabs(want)) << "r=" << r;
  }
}

//
// With Ewald on, every flavor must agree -- that is what makes the ten copies
// duplication rather than physics.
//
TEST(ForcefieldAnalyticTest, EwaldCoulombIsTheSameForEveryFlavor) {
  Forcefield f1, f2, f3, f4;
  InitForcefieldEwald(f1); InitForcefieldEwald(f2);
  InitForcefieldEwald(f3); InitForcefieldEwald(f4);
  FF_VDW_STD a(f1); FF_SHIFT b(f2); FF_SWITCH c(f3);
  f4.exp6 = true; FF_EXP6 d(f4);
  InitOneKind(a, 12.0); InitOneKind(b, 12.0);
  InitOneKind(c, 12.0); InitOneKind(d, 15.0);

  for (int i = 0; i < kNumDist; ++i) {
    const double rSq = kDist[i] * kDist[i];
    const double ref = a.CalcCoulomb(rSq, 0, 0, kQQ, 1.0, 0);
    EXPECT_EQ(b.CalcCoulomb(rSq, 0, 0, kQQ, 1.0, 0), ref) << "shift, r=" << kDist[i];
    EXPECT_EQ(c.CalcCoulomb(rSq, 0, 0, kQQ, 1.0, 0), ref) << "switch, r=" << kDist[i];
    EXPECT_EQ(d.CalcCoulomb(rSq, 0, 0, kQQ, 1.0, 0), ref) << "exp6, r=" << kDist[i];
  }
}

//
// Ewald off: each flavor's own form.
//
TEST(ForcefieldAnalyticTest, PlainCoulombMatchesClosedForm) {
  Forcefield ff;
  InitForcefield(ff);          // ewald stays off
  ff.electrostatic = true;
  FF_VDW_STD p(ff);
  InitOneKind(p, 12.0);
  for (int d = 0; d < kNumDist; ++d) {
    const double r = kDist[d];
    EXPECT_NEAR(p.CalcCoulomb(r * r, 0, 0, kQQ, 1.0, 0), kQQ / r,
                kTol * std::fabs(kQQ / r)) << "r=" << r;
  }
}

TEST(ForcefieldAnalyticTest, ShiftedCoulombVanishesAtCutoff) {
  Forcefield ff;
  InitForcefield(ff);
  ff.electrostatic = true;
  FF_SHIFT p(ff);
  InitOneKind(p, 12.0);
  for (int d = 0; d < kNumDist; ++d) {
    const double r = kDist[d];
    const double want = kQQ * (1.0 / r - 1.0 / kRCut);
    EXPECT_NEAR(p.CalcCoulomb(r * r, 0, 0, kQQ, 1.0, 0), want,
                kTol * std::fabs(want)) << "r=" << r;
  }
  // its defining property
  const double atCut = kRCut - 1e-9;
  EXPECT_NEAR(p.CalcCoulomb(atCut * atCut, 0, 0, kQQ, 1.0, 0), 0.0, 1e-6);
}

TEST(ForcefieldAnalyticTest, SwitchedCoulombMatchesClosedForm) {
  Forcefield ff;
  InitForcefield(ff);
  ff.electrostatic = true;
  FF_SWITCH p(ff);
  InitOneKind(p, 12.0);
  for (int d = 0; d < kNumDist; ++d) {
    const double r = kDist[d], rSq = r * r;
    double sw = rSq / ff.rCutSq - 1.0;
    sw *= sw;
    const double want = kQQ * sw / r;
    EXPECT_NEAR(p.CalcCoulomb(rSq, 0, 0, kQQ, 1.0, 0), want,
                kTol * std::fabs(want)) << "r=" << r;
  }
}

//
// The Coulomb virial follows the same convention as the vdW one, so the same
// finite-difference check applies.
//
TEST(ForcefieldAnalyticTest, EwaldCoulombVirialIsNegativeDerivative) {
  Forcefield ff;
  InitForcefieldEwald(ff);
  FF_VDW_STD p(ff);
  InitOneKind(p, 12.0);

  for (int d = 0; d < kNumVirDist; ++d) {
    const double r = kVirDist[d], h = 1e-5 * r;
    const double uP = p.CalcCoulomb((r + h) * (r + h), 0, 0, kQQ, 1.0, 0);
    const double uM = p.CalcCoulomb((r - h) * (r - h), 0, 0, kQQ, 1.0, 0);
    // A finite difference of a kernel that returns zero everywhere is zero,
    // and would agree with a virial kernel that also returns zero. That is
    // precisely what a missed CoulView override produced, so require the
    // energy to be non-trivial before believing the comparison.
    ASSERT_GT(std::fabs(uP), 1e-6) << "vacuous comparison at r=" << r;
    const double want = -(uP - uM) / (2.0 * h) / r;
    const double got = p.CalcCoulombVir(r * r, 0, 0, kQQ, 1.0, 0);
    EXPECT_NEAR(got, want, kFdTol * std::fabs(want) + 1e-10) << "r=" << r;
  }
}

// ===========================================================================
// dE/dlambda -- the free-energy derivative.
//
// As with the virial, the strong check is not a transcribed formula but the
// defining relationship: dE/dlambda must equal the numerical derivative of the
// energy with respect to lambda. This is the most intricate of the soft-core
// paths and the one a copied formula is most likely to get subtly wrong.
// ===========================================================================
namespace {

template <class FF>
double DEnergyDLambdaByFiniteDifference(const FF &p, double distSq,
                                        double lambda) {
  const double h = 1e-6;
  return (p.CalcEn(distSq, 0, 0, lambda + h) -
          p.CalcEn(distSq, 0, 0, lambda - h)) /
         (2.0 * h);
}

} // namespace

TEST(ForcefieldAnalyticTest, DEnergyDLambdaIsDerivativeOfEnergy) {
  const double lambdas[] = {0.2, 0.4, 0.6, 0.8};
  Forcefield ff;
  InitForcefield(ff);
  ff.freeEnergy = true;
  FF_VDW_STD p(ff);
  InitOneKind(p, 12.0);

  for (int l = 0; l < 4; ++l) {
    for (int d = 0; d < kNumVirDist; ++d) {
      const double distSq = kVirDist[d] * kVirDist[d];
      const double want = DEnergyDLambdaByFiniteDifference(p, distSq, lambdas[l]);
      const double got = p.CalcdEndL(distSq, 0, 0, lambdas[l]);
      EXPECT_NEAR(got, want, 1e-4 * std::fabs(want) + 1e-9)
          << "lambda=" << lambdas[l] << " r=" << kVirDist[d];
    }
  }
}

TEST(ForcefieldAnalyticTest, ShiftDEnergyDLambdaIsDerivativeOfEnergy) {
  Forcefield ff;
  InitForcefield(ff);
  ff.freeEnergy = true;
  FF_SHIFT p(ff);
  InitOneKind(p, 16.0);

  for (double lambda : {0.3, 0.5, 0.7}) {
    for (int d = 0; d < kNumVirDist; ++d) {
      const double distSq = kVirDist[d] * kVirDist[d];
      const double want = DEnergyDLambdaByFiniteDifference(p, distSq, lambda);
      const double got = p.CalcdEndL(distSq, 0, 0, lambda);
      EXPECT_NEAR(got, want, 1e-4 * std::fabs(want) + 1e-9)
          << "lambda=" << lambda << " r=" << kVirDist[d];
    }
  }
}

TEST(ForcefieldAnalyticTest, SwitchDEnergyDLambdaIsDerivativeOfEnergy) {
  Forcefield ff;
  InitForcefield(ff);
  ff.freeEnergy = true;
  FF_SWITCH p(ff);
  InitOneKind(p, 12.0);

  for (double lambda : {0.3, 0.6}) {
    for (int d = 0; d < kNumVirDist; ++d) {
      const double distSq = kVirDist[d] * kVirDist[d];
      const double want = DEnergyDLambdaByFiniteDifference(p, distSq, lambda);
      const double got = p.CalcdEndL(distSq, 0, 0, lambda);
      EXPECT_NEAR(got, want, 1e-4 * std::fabs(want) + 1e-9)
          << "lambda=" << lambda << " r=" << kVirDist[d];
    }
  }
}

// ===========================================================================
// Martini.
//
// Martini weights the two Mie terms separately and adds its own shift, so it is
// not a truncation of the plain form -- it is a standalone evaluator. Its
// electrostatics are dielectric-screened with a Coulomb switching distance of
// zero.
//
// The virial check here is the finite-difference one, which does not depend on
// my transcription of the formula being right -- the point being that these
// expressions were moved into ff::MartiniEval / ff::MartiniCoul by hand.
// ===========================================================================
namespace {

void InitForcefieldMartini(Forcefield &ff) {
  InitForcefield(ff);
  ff.isMartini = true;
  ff.electrostatic = true;
  ff.dielectric = 15.0; // Martini's usual screened value
}

} // namespace

TEST(ForcefieldAnalyticTest, MartiniEnergyMatchesClosedForm) {
  Forcefield ff;
  InitForcefieldMartini(ff);
  FF_SWITCH_MARTINI p(ff);
  InitOneKind(p, 12.0);

  const ff::VdwParams vp = p.VdwView();
  for (int d = 0; d < kNumDist; ++d) {
    const double r = kDist[d], rSq = r * r;
    const double r_2 = 1.0 / rSq;
    const double r_6 = r_2 * r_2 * r_2;
    const double r_n = std::pow(r_2, FFTestAccess::N(p, 0) * 0.5);

    const double rij_ron = r - vp.rOn;
    const double c3 = rij_ron * rij_ron * rij_ron;
    const double c4 = c3 * rij_ron;
    const double shiftRep = (rSq > vp.rOnSq)
        ? -(vp.An[0] / 3.0) * c3 - (vp.Bn[0] / 4.0) * c4 - vp.Cn[0]
        : -vp.Cn[0];
    const double shiftAtt = (rSq > vp.rOnSq)
        ? -(vp.A6 / 3.0) * c3 - (vp.B6 / 4.0) * c4 - vp.C6
        : -vp.C6;

    const double want = FFTestAccess::EpsilonCn(p, 0) *
        (vp.sign[0] * (r_n + shiftRep) - vp.sig6[0] * (r_6 + shiftAtt));
    const double got = p.CalcEn(rSq, 0, 0, 1.0);
    EXPECT_NEAR(got, want, 1e-9 * std::fabs(want)) << "r=" << r;
  }
}

TEST(ForcefieldAnalyticTest, MartiniVirialIsNegativeEnergyDerivative) {
  Forcefield ff;
  InitForcefieldMartini(ff);
  FF_SWITCH_MARTINI p(ff);
  InitOneKind(p, 12.0);

  for (int d = 0; d < kNumVirDist; ++d) {
    const double r = kVirDist[d];
    const double want = VirialByFiniteDifference(p, r);
    const double got = p.CalcVir(r * r, 0, 0, 1.0);
    EXPECT_NEAR(got, want, kFdTol * std::fabs(want) + 1e-10) << "r=" << r;
  }
}

TEST(ForcefieldAnalyticTest, MartiniCoulombIsDielectricScreened) {
  Forcefield ff;
  InitForcefieldMartini(ff);
  FF_SWITCH_MARTINI p(ff);
  InitOneKind(p, 12.0);

  const ff::CoulParams cp = p.CoulView(0);
  ASSERT_NEAR(cp.diElectric_1, 1.0 / ff.dielectric, 1e-12)
      << "the screening factor must reach the evaluator";

  for (int d = 0; d < kNumDist; ++d) {
    const double r = kDist[d], rSq = r * r;
    const double c3 = r * rSq, c4 = rSq * rSq;
    const double coul = -(cp.A1 / 3.0) * c3 - (cp.B1 / 4.0) * c4 - cp.C1;
    const double want = kQQ * cp.diElectric_1 * (1.0 / r + coul);
    const double got = p.CalcCoulomb(rSq, 0, 0, kQQ, 1.0, 0);
    EXPECT_NEAR(got, want, 1e-9 * std::fabs(want)) << "r=" << r;
  }
}

TEST(ForcefieldAnalyticTest, MartiniCoulombVirialIsNegativeDerivative) {
  Forcefield ff;
  InitForcefieldMartini(ff);
  FF_SWITCH_MARTINI p(ff);
  InitOneKind(p, 12.0);

  for (int d = 0; d < kNumVirDist; ++d) {
    const double r = kVirDist[d], h = 1e-5 * r;
    const double uP = p.CalcCoulomb((r + h) * (r + h), 0, 0, kQQ, 1.0, 0);
    const double uM = p.CalcCoulomb((r - h) * (r - h), 0, 0, kQQ, 1.0, 0);
    // A finite difference of a kernel that returns zero everywhere is zero,
    // and would agree with a virial kernel that also returns zero. That is
    // precisely what a missed CoulView override produced, so require the
    // energy to be non-trivial before believing the comparison.
    ASSERT_GT(std::fabs(uP), 1e-6) << "vacuous comparison at r=" << r;
    const double want = -(uP - uM) / (2.0 * h) / r;
    const double got = p.CalcCoulombVir(r * r, 0, 0, kQQ, 1.0, 0);
    EXPECT_NEAR(got, want, kFdTol * std::fabs(want) + 1e-10) << "r=" << r;
  }
}

/*******************************************************************************
Unit tests for the Mie-exponent fast path in the van der Waals kernels.

Each forcefield flavor computes the repulsive term (sigma/r)^n three ways,
selected by the precomputed integer exponent `nExp`:

  nExp == 12          -> a hand-expanded special case
  nExp in [7,50]      -> num::POW integer-exponent table
  nExp == 0xFFFFFFFF  -> std::pow fallback (non-integer or out-of-range n)

All three must agree. The tests below pin that invariant by evaluating each
kernel twice for the same n -- once on the fast path, once with `nExp` forced
to the sentinel so the kernel takes its own std::pow fallback -- and comparing.
Everything else in the expression is identical between the two evaluations, so
any difference isolates the repulsive term.

This is the coverage that was missing: every system under test/input/Systems
uses n=12, so a fast path that is wrong only for n != 12 passes the rest of
the suite unnoticed.
*******************************************************************************/

#include <cmath>
#include <gtest/gtest.h>

#include "FFParticle.h"
#include "FFSetup.h"
#include "FFShift.h"
#include "FFSwitch.h"
#include "FFSwitchMartini.h"
#include "Forcefield.h"
#include "NumLib.h"

namespace {

const uint POW_SENTINEL = 0xFFFFFFFF;

// Lowest and highest integer exponents the fast path is enabled for; must track
// the guard in FFParticle::Blend / AdjNBfix.
const uint N_MIN = 7;
const uint N_MAX = 50;

// CH4-like united atom, so the numbers stay in a physically sensible range.
const double kSigma = 3.740;
const double kEpsilon = 161.0;
const double kRCut = 14.0;

// Separations spanning the repulsive wall, the well, and the tail.
const double kDist[] = {3.2, 3.5, 3.74, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0};
const int kNumDist = sizeof(kDist) / sizeof(kDist[0]);

double MieCn(double n) {
  return n / (n - 6.0) * std::pow(n / 6.0, 6.0 / (n - 6.0));
}

// A minimal Forcefield carrying only what the CalcEn paths read.
void InitForcefield(Forcefield &ff) {
  ff.rCut = kRCut;
  ff.rCutSq = kRCut * kRCut;
  ff.rCutLow = 0.0;
  ff.rCutLowSq = 0.0;
  ff.rswitch = 10.0;
  ff.freeEnergy = false;
  ff.sc_coul = false;
  ff.sc_alpha = 0.5;
  ff.sc_sigma = 3.0;
  ff.sc_sigma_6 = 729.0;
  ff.sc_power = 2;
  ff.vdwGeometricSigma = false;
  ff.isMartini = false;
  ff.exp6 = false;
}

// The forcefield parameter arrays are protected, so the fixtures below derive
// from each flavor to populate them. Arrays left alone stay NULL from
// FFParticle's constructor, which its destructor tolerates.
template <class Base> struct TestFF : public Base {
  explicit TestFF(Forcefield &ff) : Base(ff) {}

  // Populate the single-kind arrays shared by every flavor.
  void InitCommon(double nVal) {
    this->count = 1;
    this->n = new double[1];
    this->n_1_4 = new double[1];
    this->sigmaSq = new double[1];
    this->sigmaSq_1_4 = new double[1];
    this->epsilon = new double[1];
    this->epsilon_1_4 = new double[1];
    this->epsilon_cn = new double[1];
    this->epsilon_cn_1_4 = new double[1];
    this->epsilon_cn_6 = new double[1];
    this->epsilon_cn_6_1_4 = new double[1];
    this->nOver6 = new double[1];
    this->nOver6_1_4 = new double[1];
    this->nExp = new uint[1];
    this->nExp_1_4 = new uint[1];

    const double cn = MieCn(nVal);
    this->n[0] = this->n_1_4[0] = nVal;
    this->sigmaSq[0] = this->sigmaSq_1_4[0] = kSigma * kSigma;
    this->epsilon[0] = this->epsilon_1_4[0] = kEpsilon;
    this->epsilon_cn[0] = this->epsilon_cn_1_4[0] = cn * kEpsilon;
    this->epsilon_cn_6[0] = this->epsilon_cn_6_1_4[0] = cn * kEpsilon * 6.0;
    this->nOver6[0] = this->nOver6_1_4[0] = nVal / 6.0;

    // Mirror the selection FFParticle::Blend performs.
    const uint n_int = static_cast<uint>(nVal);
    const bool integral =
        (nVal == static_cast<double>(n_int)) && n_int >= N_MIN && n_int <= N_MAX;
    this->nExp[0] = this->nExp_1_4[0] = integral ? n_int : POW_SENTINEL;
  }

  // Flavors with extra parameter arrays override this.
  void InitExtra() {}

  void SetExponent(uint e) { this->nExp[0] = e; }
  uint Exponent() const { return this->nExp[0]; }

  // Build the parameters a one-kind Mie system would be read from file and
  // run them through the production Init/Blend path, so nExp is computed by
  // real code rather than mirrored by this fixture.
  void SetupViaInit(double nVal) {
    this->exp6 = false;
    ff_setup::Particle mie;
    mie.setIsCHARMM(false);
    mie.getnamelist().push_back("TST");
    mie.epsilon.push_back(kEpsilon);
    mie.sigma.push_back(kSigma);
    mie.n.push_back(nVal);
    mie.epsilon_1_4.push_back(kEpsilon);
    mie.sigma_1_4.push_back(kSigma);
    mie.n_1_4.push_back(nVal);
    ff_setup::NBfix nbfix;
    this->Init(mie, nbfix);
  }

  void Setup(double nVal) {
    InitCommon(nVal);
    InitExtra();
  }
};

struct TestStd : public TestFF<FFParticle> {
  explicit TestStd(Forcefield &ff) : TestFF<FFParticle>(ff) {}
};

struct TestShift : public TestFF<FF_SHIFT> {
  explicit TestShift(Forcefield &ff) : TestFF<FF_SHIFT>(ff) {}
  void InitExtra() {
    this->shiftConst = new double[1];
    this->shiftConst_1_4 = new double[1];
    // Cancels between the two evaluations; only needs to be finite.
    this->shiftConst[0] = this->shiftConst_1_4[0] = 0.0;
  }
  void Setup(double nVal) {
    InitCommon(nVal);
    InitExtra();
  }
};

struct TestSwitch : public TestFF<FF_SWITCH> {
  explicit TestSwitch(Forcefield &ff) : TestFF<FF_SWITCH>(ff) {}
  void InitExtra() {
    this->rOn = this->forcefield.rswitch;
    this->rOnSq = this->rOn * this->rOn;
    const double d = this->forcefield.rCutSq - this->rOnSq;
    this->factor1 = this->forcefield.rCutSq - 3.0 * this->rOnSq;
    this->factor2 = 1.0 / (d * d * d);
  }
  void Setup(double nVal) {
    InitCommon(nVal);
    InitExtra();
  }
};

struct TestMartini : public TestFF<FF_SWITCH_MARTINI> {
  explicit TestMartini(Forcefield &ff) : TestFF<FF_SWITCH_MARTINI>(ff) {}
  void InitExtra() {
    this->An = new double[1];
    this->Bn = new double[1];
    this->Cn = new double[1];
    this->An_1_4 = new double[1];
    this->Bn_1_4 = new double[1];
    this->Cn_1_4 = new double[1];
    this->sig6 = new double[1];
    this->sign = new double[1];
    this->sig6_1_4 = new double[1];
    this->sign_1_4 = new double[1];
    // Switching constants cancel between the two evaluations.
    this->An[0] = this->An_1_4[0] = 1.0;
    this->Bn[0] = this->Bn_1_4[0] = 1.0;
    this->Cn[0] = this->Cn_1_4[0] = 1.0;
    this->sig6[0] = this->sig6_1_4[0] = 1.0;
    this->sign[0] = this->sign_1_4[0] = 1.0;
    this->A6 = this->B6 = this->C6 = 1.0;
    this->rOn = this->forcefield.rswitch;
    this->rOnSq = this->rOn * this->rOn;
    this->rOnCoul = this->rOn;
    this->diElectric_1 = 1.0;
  }
  void Setup(double nVal) {
    InitCommon(nVal);
    InitExtra();
  }
};

} // namespace

//
// The contract num::POW is expected to satisfy. Callers pass (d2, d2^2, d2^3)
// and the full exponent e, and must get back d2^(e/2). Passing e/2 instead of e
// silently yields d2^(e/4), which is the defect this pins.
//
TEST(MieExponentTest, NumLibPowMatchesStdPow) {
  const double bases[] = {0.3, 0.7, 1.0, 1.4, 2.0};
  for (int b = 0; b < 5; ++b) {
    const double d2 = bases[b];
    const double d4 = d2 * d2;
    const double d6 = d4 * d2;
    for (uint e = N_MIN; e <= N_MAX; ++e) {
      const double got = num::POW(d2, d4, d6, e);
      const double want = std::pow(d2, e * 0.5);
      EXPECT_NEAR(got, want, 1e-11 * std::fabs(want))
          << "num::POW(d2=" << d2 << ", e=" << e << ")";
    }
  }
}

//
// Fast path vs. the kernel's own std::pow fallback, for every enabled exponent.
// Everything outside the repulsive term is identical between the two
// evaluations, so any difference isolates the fast path.
//
namespace {

template <class Fixture> void CheckFastPathMatchesFallback() {
  for (uint n_int = N_MIN; n_int <= N_MAX; ++n_int) {
    Forcefield ff;
    InitForcefield(ff);
    Fixture particle(ff);
    particle.Setup(static_cast<double>(n_int));
    ASSERT_EQ(particle.Exponent(), n_int) << "fast path not selected";

    for (int d = 0; d < kNumDist; ++d) {
      const double distSq = kDist[d] * kDist[d];
      const double fast = particle.CalcEn(distSq, 0, 0, 1.0);
      particle.SetExponent(POW_SENTINEL);
      const double ref = particle.CalcEn(distSq, 0, 0, 1.0);
      particle.SetExponent(n_int);

      ASSERT_TRUE(std::isfinite(ref))
          << "reference not finite, n=" << n_int << " r=" << kDist[d];
      EXPECT_NEAR(fast, ref, 1e-10 * std::fabs(ref))
          << "n=" << n_int << " r=" << kDist[d];
    }
  }
}

} // namespace

TEST(MieExponentTest, StdKernelFastPathMatchesFallback) {
  CheckFastPathMatchesFallback<TestStd>();
}

TEST(MieExponentTest, ShiftKernelFastPathMatchesFallback) {
  CheckFastPathMatchesFallback<TestShift>();
}

TEST(MieExponentTest, SwitchKernelFastPathMatchesFallback) {
  CheckFastPathMatchesFallback<TestSwitch>();
}

TEST(MieExponentTest, MartiniKernelFastPathMatchesFallback) {
  CheckFastPathMatchesFallback<TestMartini>();
}

//
// Independent analytic check, so the comparisons above cannot pass by having
// both paths wrong in the same way. U(r) = Cn*eps*[(sigma/r)^n - (sigma/r)^6].
//
TEST(MieExponentTest, StdKernelMatchesAnalyticMie) {
  for (uint n_int = N_MIN; n_int <= N_MAX; ++n_int) {
    const double n = static_cast<double>(n_int);
    Forcefield ff;
    InitForcefield(ff);
    TestStd particle(ff);
    particle.Setup(n);

    for (int d = 0; d < kNumDist; ++d) {
      const double r = kDist[d];
      const double sr = kSigma / r;
      const double want =
          MieCn(n) * kEpsilon * (std::pow(sr, n) - std::pow(sr, 6.0));
      const double got = particle.CalcEn(r * r, 0, 0, 1.0);
      EXPECT_NEAR(got, want, 1e-9 * std::fabs(want))
          << "n=" << n_int << " r=" << r;
    }
  }
}

//
// Non-integer and out-of-range exponents must fall back rather than index the
// integer table.
//
TEST(MieExponentTest, OutOfRangeExponentSelectsFallback) {
  // Non-integral, and integral but outside [N_MIN, N_MAX] in both directions.
  const double exponents[] = {12.5, 16.4, 6.0, 51.0};
  for (int e = 0; e < 4; ++e) {
    Forcefield ff;
    InitForcefield(ff);
    TestStd particle(ff);
    particle.Setup(exponents[e]);
    EXPECT_EQ(particle.Exponent(), POW_SENTINEL)
        << "n=" << exponents[e] << " should not use the integer table";
  }
}

//
// The fallback itself must still reproduce the analytic Mie form. n=6 is
// excluded: the Mie prefactor n/(n-6) diverges there, so the potential is
// undefined rather than merely outside the fast path.
//
TEST(MieExponentTest, FallbackMatchesAnalyticMie) {
  const double exponents[] = {12.5, 16.4, 51.0};
  for (int e = 0; e < 3; ++e) {
    Forcefield ff;
    InitForcefield(ff);
    TestStd particle(ff);
    particle.Setup(exponents[e]);
    ASSERT_EQ(particle.Exponent(), POW_SENTINEL);

    for (int d = 0; d < kNumDist; ++d) {
      const double r = kDist[d];
      const double sr = kSigma / r;
      const double want = MieCn(exponents[e]) * kEpsilon *
                          (std::pow(sr, exponents[e]) - std::pow(sr, 6.0));
      const double got = particle.CalcEn(r * r, 0, 0, 1.0);
      EXPECT_NEAR(got, want, 1e-9 * std::fabs(want))
          << "n=" << exponents[e] << " r=" << r;
    }
  }
}

//
// End-to-end: parameters go through the production Init/Blend path, which is
// where the stored exponent is derived. This is the check that fails if Blend
// records n/2 instead of n, independently of how the kernel consumes it.
//
TEST(MieExponentTest, InitPathMatchesAnalyticMie) {
  for (uint n_int = N_MIN; n_int <= N_MAX; ++n_int) {
    const double n = static_cast<double>(n_int);
    Forcefield ff;
    InitForcefield(ff);
    TestStd particle(ff);
    particle.SetupViaInit(n);

    EXPECT_EQ(particle.Exponent(), n_int)
        << "Blend must store the full Mie exponent, n=" << n_int;

    for (int d = 0; d < kNumDist; ++d) {
      const double r = kDist[d];
      const double sr = kSigma / r;
      const double want =
          MieCn(n) * kEpsilon * (std::pow(sr, n) - std::pow(sr, 6.0));
      const double got = particle.CalcEn(r * r, 0, 0, 1.0);
      EXPECT_NEAR(got, want, 1e-9 * std::fabs(want))
          << "n=" << n_int << " r=" << r;
    }
  }
}

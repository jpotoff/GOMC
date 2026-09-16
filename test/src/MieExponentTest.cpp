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

// The concrete forcefields are `final` (see FFParticle.h) so the parameter
// arrays cannot be reached by deriving a fixture. Instead each test builds the
// parameters a one-kind Mie system would be read from file and runs them
// through the production Init/Blend path, which populates every array --
// including the derived classes' own. Only the precomputed exponent has to be
// poked directly, via the named friend below.
void InitOneKind(FFParticle &p, double nVal) {
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
  p.Init(mie, nbfix);
}

} // namespace

struct MieExponentTestAccess {
  static void SetExponent(FFParticle &p, uint e) { p.nExp[0] = e; }
  static uint Exponent(const FFParticle &p) { return p.nExp[0]; }
};

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

template <class FFT> void CheckFastPathMatchesFallback() {
  for (uint n_int = N_MIN; n_int <= N_MAX; ++n_int) {
    Forcefield ff;
    InitForcefield(ff);
    FFT particle(ff);
    InitOneKind(particle, static_cast<double>(n_int));
    ASSERT_EQ(MieExponentTestAccess::Exponent(particle), n_int)
        << "fast path not selected";

    for (int d = 0; d < kNumDist; ++d) {
      const double distSq = kDist[d] * kDist[d];
      const double fast = particle.CalcEn(distSq, 0, 0, 1.0);
      MieExponentTestAccess::SetExponent(particle, POW_SENTINEL);
      const double ref = particle.CalcEn(distSq, 0, 0, 1.0);
      MieExponentTestAccess::SetExponent(particle, n_int);

      ASSERT_TRUE(std::isfinite(ref))
          << "reference not finite, n=" << n_int << " r=" << kDist[d];
      EXPECT_NEAR(fast, ref, 1e-10 * std::fabs(ref))
          << "n=" << n_int << " r=" << kDist[d];
    }
  }
}

} // namespace

TEST(MieExponentTest, StdKernelFastPathMatchesFallback) {
  CheckFastPathMatchesFallback<FFParticle>();
}

TEST(MieExponentTest, ShiftKernelFastPathMatchesFallback) {
  CheckFastPathMatchesFallback<FF_SHIFT>();
}

TEST(MieExponentTest, SwitchKernelFastPathMatchesFallback) {
  CheckFastPathMatchesFallback<FF_SWITCH>();
}

TEST(MieExponentTest, MartiniKernelFastPathMatchesFallback) {
  CheckFastPathMatchesFallback<FF_SWITCH_MARTINI>();
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
    FFParticle particle(ff);
    InitOneKind(particle, n);

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
    FFParticle particle(ff);
    InitOneKind(particle, exponents[e]);
    EXPECT_EQ(MieExponentTestAccess::Exponent(particle), POW_SENTINEL)
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
    FFParticle particle(ff);
    InitOneKind(particle, exponents[e]);
    ASSERT_EQ(MieExponentTestAccess::Exponent(particle), POW_SENTINEL);

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
    FFParticle particle(ff);
    InitOneKind(particle, n);

    EXPECT_EQ(MieExponentTestAccess::Exponent(particle), n_int)
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

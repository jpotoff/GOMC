/*******************************************************************************
Unit tests for the tabulated real-space Ewald kernels (EwaldRealTable.h).

The table replaces `erfc`, `exp`, `sqrt` and a division in the innermost
electrostatics loop, so its accuracy has to be pinned explicitly -- the rest of
the suite would not notice a subtly wrong interpolation.

Accuracy bar: GOMC sets alpha = sqrt(-log(tolerance))/rCutCoulomb, so the
real-space sum is already truncated at ~`tolerance` relative error. Interpolation
error is required to sit well below that.
*******************************************************************************/

#include <cmath>
#include <gtest/gtest.h>

#include "EwaldRealTable.h"

namespace {

// Representative of a production setup: tolerance 1e-5, 10 A Coulomb cutoff.
const double kTolerance = 1e-5;
const double kRCut = 10.0;
const double kAlpha = 1.0; // replaced in SetUp below
const double kRCutLowSq = 1.0;

double AlphaFor(double rCut, double tol) { return std::sqrt(-std::log(tol)) / rCut; }

// The accuracy the default table size must achieve. Measured ~1.4e-7; assert an
// order of margin so the test flags a real regression, not noise.
const double kMaxRelErr = 1e-6;

class EwaldRealTableTest : public ::testing::Test {
protected:
  void SetUp() override {
    alpha = AlphaFor(kRCut, kTolerance);
    for (uint b = 0; b < BOX_TOTAL; ++b) {
      alphas[b] = alpha;
      rCutSq[b] = kRCut * kRCut;
    }
    table.Init(alphas, rCutSq, kRCutLowSq);
  }

  double alpha;
  double alphas[BOX_TOTAL];
  double rCutSq[BOX_TOTAL];
  EwaldRealTable table;
};

} // namespace

TEST_F(EwaldRealTableTest, EnergyMatchesExactAcrossDomain) {
  double worst = 0.0;
  const int N = 200000;
  for (int i = 0; i <= N; ++i) {
    const double distSq =
        kRCutLowSq + (rCutSq[0] - kRCutLowSq) * (double(i) / N) * 0.9999999;
    double got;
    ASSERT_TRUE(table.Energy(distSq, 0, got)) << "distSq=" << distSq;
    const double want = EwaldRealTable::ExactEnergy(distSq, alpha);
    const double rel = std::fabs(got - want) / std::fabs(want);
    worst = std::max(worst, rel);
  }
  EXPECT_LT(worst, kMaxRelErr) << "worst relative error " << worst;
}

TEST_F(EwaldRealTableTest, VirialMatchesExactAcrossDomain) {
  double worst = 0.0;
  const int N = 200000;
  for (int i = 0; i <= N; ++i) {
    const double distSq =
        kRCutLowSq + (rCutSq[0] - kRCutLowSq) * (double(i) / N) * 0.9999999;
    double got;
    ASSERT_TRUE(table.Virial(distSq, 0, got)) << "distSq=" << distSq;
    const double want = EwaldRealTable::ExactVirial(distSq, alpha);
    const double rel = std::fabs(got - want) / std::fabs(want);
    worst = std::max(worst, rel);
  }
  EXPECT_LT(worst, kMaxRelErr) << "worst relative error " << worst;
}

//
// The interpolation is built from analytic derivatives. If either derivative
// formula is wrong the table still looks plausible at the knots but sags
// between them, which the sweeps above would catch only weakly. Check the
// derivatives directly against central differences.
//
TEST_F(EwaldRealTableTest, AnalyticDerivativesAreConsistent) {
  for (double distSq = 2.0; distSq < 99.0; distSq += 3.7) {
    const double h = 1e-6 * distSq;

    const double fdEn = (EwaldRealTable::ExactEnergy(distSq + h, alpha) -
                         EwaldRealTable::ExactEnergy(distSq - h, alpha)) /
                        (2.0 * h);
    const double fdVir = (EwaldRealTable::ExactVirial(distSq + h, alpha) -
                          EwaldRealTable::ExactVirial(distSq - h, alpha)) /
                         (2.0 * h);

    // Recover the table's slope at a knot by evaluating either side of it.
    double a, c;
    ASSERT_TRUE(table.Energy(distSq - h, 0, a));
    ASSERT_TRUE(table.Energy(distSq + h, 0, c));
    EXPECT_NEAR((c - a) / (2.0 * h), fdEn, 1e-5 * std::fabs(fdEn) + 1e-12)
        << "energy slope at distSq=" << distSq;

    ASSERT_TRUE(table.Virial(distSq - h, 0, a));
    ASSERT_TRUE(table.Virial(distSq + h, 0, c));
    EXPECT_NEAR((c - a) / (2.0 * h), fdVir, 1e-5 * std::fabs(fdVir) + 1e-12)
        << "virial slope at distSq=" << distSq;
  }
}

//
// Outside the tabulated range the caller must fall back to the exact form.
// Reporting a hit there would silently return garbage for overlapping pairs.
//
TEST_F(EwaldRealTableTest, ReportsMissOutsideTable) {
  double out = 0.0;
  // Below the floor: the kernel diverges, these are overlaps.
  EXPECT_FALSE(table.Energy(0.01, 0, out));
  EXPECT_FALSE(table.Energy(0.24, 0, out));
  EXPECT_FALSE(table.Virial(0.01, 0, out));
  // Beyond the Coulomb cutoff.
  EXPECT_FALSE(table.Energy(rCutSq[0] * 1.001, 0, out));
  EXPECT_FALSE(table.Virial(rCutSq[0] * 1.001, 0, out));
  // Just inside each end must hit.
  EXPECT_TRUE(table.Energy(kRCutLowSq + 1e-9, 0, out));
  EXPECT_TRUE(table.Energy(rCutSq[0] * 0.9999, 0, out));
}

//
// An uninitialised box (electrostatics off => rCutCoulombSq == 0) must report a
// miss rather than index an empty table.
//
TEST_F(EwaldRealTableTest, EmptyBoxReportsMiss) {
  double zeroCut[BOX_TOTAL];
  double a[BOX_TOTAL];
  for (uint b = 0; b < BOX_TOTAL; ++b) {
    zeroCut[b] = 0.0;
    a[b] = alpha;
  }
  EwaldRealTable empty;
  empty.Init(a, zeroCut, kRCutLowSq);
  double out = 0.0;
  EXPECT_FALSE(empty.Energy(25.0, 0, out));
  EXPECT_FALSE(empty.Virial(25.0, 0, out));
}

//
// The tabulated domain shape is set by the tolerance alone (alpha*rCut =
// sqrt(-log(tol))), so accuracy must hold across cutoffs and tolerances.
//
TEST_F(EwaldRealTableTest, AccuracyHoldsAcrossCutoffsAndTolerances) {
  const double cutoffs[] = {8.0, 10.0, 12.0, 14.0};
  const double tols[] = {1e-4, 1e-5, 1e-6};
  for (int ci = 0; ci < 4; ++ci) {
    for (int ti = 0; ti < 3; ++ti) {
      const double aa = AlphaFor(cutoffs[ci], tols[ti]);
      double as[BOX_TOTAL], rs[BOX_TOTAL];
      for (uint b = 0; b < BOX_TOTAL; ++b) {
        as[b] = aa;
        rs[b] = cutoffs[ci] * cutoffs[ci];
      }
      EwaldRealTable t;
      t.Init(as, rs, kRCutLowSq);

      double worst = 0.0;
      for (int i = 0; i <= 20000; ++i) {
        const double distSq =
            kRCutLowSq + (rs[0] - kRCutLowSq) * (double(i) / 20000) * 0.9999999;
        double got;
        ASSERT_TRUE(t.Energy(distSq, 0, got));
        const double want = EwaldRealTable::ExactEnergy(distSq, aa);
        worst = std::max(worst, std::fabs(got - want) / std::fabs(want));
      }
      EXPECT_LT(worst, kMaxRelErr)
          << "rCut=" << cutoffs[ci] << " tol=" << tols[ti] << " worst=" << worst;
    }
  }
}

//
// Guard: past EwaldRealTable::MaxCutoff the table would no longer stay
// cache-resident, so those boxes must report misses and let callers use the
// stock erfc. Typical GEMC gas-phase cutoffs land above the limit.
//
TEST_F(EwaldRealTableTest, LongCutoffFallsBackToExactErfc) {
  const double longCut = EwaldRealTable::MaxCutoff() + 5.0;
  double a[BOX_TOTAL], rs[BOX_TOTAL];
  for (uint b = 0; b < BOX_TOTAL; ++b) {
    a[b] = AlphaFor(longCut, kTolerance);
    rs[b] = longCut * longCut;
  }
  EwaldRealTable t;
  t.Init(a, rs, kRCutLowSq);

  double out = 0.0;
  EXPECT_FALSE(t.Energy(25.0, 0, out)) << "table must be disabled past MaxCutoff";
  EXPECT_FALSE(t.Virial(25.0, 0, out));
  EXPECT_FALSE(t.Energy(rs[0] * 0.5, 0, out));
}

//
// Just under the limit the table must still be built, so the guard cannot
// silently disable the common case.
//
TEST_F(EwaldRealTableTest, JustUnderLimitStaysEnabled) {
  const double cut = EwaldRealTable::MaxCutoff() - 1.0;
  double a[BOX_TOTAL], rs[BOX_TOTAL];
  for (uint b = 0; b < BOX_TOTAL; ++b) {
    a[b] = AlphaFor(cut, kTolerance);
    rs[b] = cut * cut;
  }
  EwaldRealTable t;
  t.Init(a, rs, kRCutLowSq);

  double got = 0.0;
  ASSERT_TRUE(t.Energy(25.0, 0, got));
  EXPECT_NEAR(got, EwaldRealTable::ExactEnergy(25.0, a[0]),
              kMaxRelErr * std::fabs(EwaldRealTable::ExactEnergy(25.0, a[0])));
}

//
// Mixed cutoffs, the GEMC case: a short-cutoff liquid box keeps its table
// while a long-cutoff gas box falls back, independently.
//
TEST_F(EwaldRealTableTest, GuardIsPerBox) {
  if (BOX_TOTAL < 2)
    GTEST_SKIP() << "needs two boxes";
  double a[BOX_TOTAL], rs[BOX_TOTAL];
  a[0] = AlphaFor(12.0, kTolerance);
  rs[0] = 12.0 * 12.0;
  a[1] = AlphaFor(100.0, kTolerance);
  rs[1] = 100.0 * 100.0;
  EwaldRealTable t;
  t.Init(a, rs, kRCutLowSq);

  double out = 0.0;
  EXPECT_TRUE(t.Energy(25.0, 0, out)) << "box 0 (12 A) should keep its table";
  EXPECT_FALSE(t.Energy(25.0, 1, out)) << "box 1 (100 A) should fall back";
}

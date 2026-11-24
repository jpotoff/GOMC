/*******************************************************************************
Distribution tests for the flexible-bond length sampler (cbmc/FlexibleBond.h).

SampleBondLength must draw exactly from p(l) ~ l^2 exp(-beta Kb (l - b0)^2),
l > 0, since CBMC adds no Rosenbluth weight for the draw. The sample mean,
variance and a binned histogram are compared with p integrated numerically,
for a stiff C-C bond at room temperature (nearly Gaussian), a soft bond (the
l^2 Jacobian visibly skews p) and a very soft bond (p reaches l = 0).
*******************************************************************************/

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "FlexibleBond.h"

namespace {

struct Rng {
  explicit Rng(unsigned seed) : gen(seed) {}
  double operator()() { return uni(gen); }
  double Gaussian(double mean, double sd) { return mean + sd * norm(gen); }
  std::mt19937_64 gen;
  std::uniform_real_distribution<double> uni{0.0, 1.0};
  std::normal_distribution<double> norm{0.0, 1.0};
};

// Unnormalised log density of the bond length
double LogP(double l, double b0, double kb, double beta) {
  return 2.0 * std::log(l) - beta * kb * (l - b0) * (l - b0);
}

void CheckDistribution(double b0, double kb, double beta, unsigned seed) {
  // reference moments and bin probabilities by quadrature on (0, hi]
  const double s = std::sqrt(0.5 / (beta * kb));
  const double hi = b0 + 12.0 * s + 12.0 * s * s / b0 + 1.0;
  const double lo = std::max(1e-12, b0 - 12.0 * s);
  const int nq = 400000;
  const double h = (hi - lo) / nq;
  const double ref = LogP(std::max(lo, std::min(hi, b0)), b0, kb, beta);
  double z = 0.0, m1 = 0.0, m2 = 0.0;
  const int nbin = 40;
  const double blo = std::max(lo, b0 - 4.0 * s - 2.0 * s * s / b0);
  const double bhi = b0 + 5.0 * s + 4.0 * s * s / b0;
  const double bw = (bhi - blo) / nbin;
  std::vector<double> pbin(nbin, 0.0);
  for (int i = 0; i < nq; ++i) {
    double l = lo + (i + 0.5) * h;
    double w = std::exp(LogP(l, b0, kb, beta) - ref) * h;
    z += w;
    m1 += w * l;
    m2 += w * l * l;
    int b = (int)std::floor((l - blo) / bw);
    if (b >= 0 && b < nbin)
      pbin[b] += w;
  }
  m1 /= z;
  m2 /= z;
  for (double &p : pbin)
    p /= z;
  const double var = m2 - m1 * m1;

  Rng rng(seed);
  const long n = 2000000;
  double s1 = 0.0, s2 = 0.0;
  std::vector<double> count(nbin, 0.0);
  for (long i = 0; i < n; ++i) {
    double l = cbmc::SampleBondLength(rng, b0, kb, beta);
    ASSERT_GT(l, 0.0);
    s1 += l;
    s2 += (l - m1) * (l - m1);
    int b = (int)std::floor((l - blo) / bw);
    if (b >= 0 && b < nbin)
      count[b] += 1.0;
  }
  const double mean = s1 / n, sampleVar = s2 / n;
  // 5 standard errors
  EXPECT_NEAR(mean, m1, 5.0 * std::sqrt(var / n)) << "b0 " << b0 << " kb " << kb;
  EXPECT_NEAR(sampleVar, var, 5.0 * var * std::sqrt(2.0 / n))
      << "b0 " << b0 << " kb " << kb;
  // histogram: chi^2 over bins with enough expected counts
  double chi2 = 0.0;
  int dof = 0;
  for (int b = 0; b < nbin; ++b) {
    double e = pbin[b] * n;
    if (e < 50.0)
      continue;
    chi2 += (count[b] - e) * (count[b] - e) / e;
    ++dof;
  }
  ASSERT_GT(dof, 10);
  // mean dof, sd sqrt(2 dof): allow 5 sd
  EXPECT_LT(chi2, dof + 5.0 * std::sqrt(2.0 * dof))
      << "b0 " << b0 << " kb " << kb << " dof " << dof;
}

TEST(FlexibleBondTest, StiffCarbonCarbon) {
  // TraPPE-like C-C: Kb = 302134 K/A^2 (600 kcal/mol/A^2), 300 K
  CheckDistribution(1.54, 302133.817, 1.0 / 300.0, 11u);
}

TEST(FlexibleBondTest, SoftBondJacobianSkew) {
  // s = 0.22 A on b0 = 1: the l^2 factor shifts the mean by about 2 s^2/b0
  CheckDistribution(1.0, 5000.0, 1.0 / 500.0, 12u);
}

TEST(FlexibleBondTest, VerySoftBondNearZero) {
  // s = 1.2 A on b0 = 1: much of the Gaussian lies below l = 0
  CheckDistribution(1.0, 100.0, 1.0 / 300.0, 13u);
}

} // namespace

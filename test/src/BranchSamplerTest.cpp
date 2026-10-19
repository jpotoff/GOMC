/*******************************************************************************
Distribution tests for the CBMC branch sampler (BranchSampler).

Each case builds a branch point with a given set of bond angle parameters and
runs the sampler the way DCHedron uses it: N trials are generated, one is
selected with probability proportional to its weight, and the move is
accepted with min(1, W_new / W_old), where W_old includes the weight of the
current geometry. The stationary distribution of that chain must be the
bending Boltzmann distribution, so the moments of every prev-focus-b angle
and every b-focus-c angle are compared against an independent reference: a
Metropolis random walk of the bond directions on the unit sphere, whose
target is simply exp(-beta U) with respect to the uniform sphere measure.

The cases cover tetrahedral, trigonal planar, linear, trigonal bipyramidal and
octahedral centers, soft and asymmetric angles, the Martini cosine form, and
fixed bond angles.
*******************************************************************************/

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "BranchSampler.h"
#include "FFAngles.h"

namespace {

using cbmc::BranchSampler;
using cbmc::MAX_BONDS;

struct Rng {
  explicit Rng(unsigned seed) : gen(seed) {}
  double operator()() { return dist(gen); }
  std::mt19937_64 gen;
  std::uniform_real_distribution<double> dist{0.0, 1.0};
};

// Angle parameters set directly (FFAngles normally reads them from a file)
template <class Base> struct TestAngles : public Base {
  void Set(const std::vector<double> &k, const std::vector<double> &t0Deg,
           const std::vector<bool> &fix) {
    this->count = k.size();
    this->Ktheta = new double[k.size()];
    this->theta0 = new double[k.size()];
    this->fixed = new bool[k.size()];
    for (size_t i = 0; i < k.size(); ++i) {
      this->Ktheta[i] = k[i];
      this->theta0[i] = t0Deg[i] * M_PI / 180.0;
      this->fixed[i] = fix[i];
    }
  }
};

// A branch point: nBonds atoms bonded to the focus besides prev, and the
// angle kind of every prev-focus-b (diagonal) and b-focus-c pair.
struct Branch {
  uint n;
  uint kinds[MAX_BONDS][MAX_BONDS];
};

struct Moments {
  std::vector<double> mean, sd, seMean;
};

// Angles of a geometry in degrees: n prev angles, then pairs (b, c), c < b
std::vector<double> Angles(const Branch &br, const double *th,
                           const double *ph) {
  std::vector<double> a;
  for (uint b = 0; b < br.n; ++b)
    a.push_back(th[b] * 180.0 / M_PI);
  for (uint b = 1; b < br.n; ++b)
    for (uint c = 0; c < b; ++c)
      a.push_back(BranchSampler::AngleBetween(th[b], ph[b], th[c], ph[c]) *
                  180.0 / M_PI);
  return a;
}

Moments Summarize(const std::vector<std::vector<double>> &samples) {
  const size_t m = samples[0].size(), n = samples.size(), blocks = 20;
  Moments r;
  r.mean.assign(m, 0.0);
  r.sd.assign(m, 0.0);
  r.seMean.assign(m, 0.0);
  for (size_t j = 0; j < m; ++j) {
    double s = 0.0, s2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
      s += samples[i][j];
      s2 += samples[i][j] * samples[i][j];
    }
    r.mean[j] = s / n;
    r.sd[j] = sqrt(std::max(0.0, s2 / n - r.mean[j] * r.mean[j]));
    // block standard error of the mean, robust to correlation in the chains
    double bs2 = 0.0;
    for (size_t k = 0; k < blocks; ++k) {
      double bm = 0.0;
      size_t lo = k * n / blocks, hi = (k + 1) * n / blocks;
      for (size_t i = lo; i < hi; ++i)
        bm += samples[i][j];
      bm /= (hi - lo);
      bs2 += (bm - r.mean[j]) * (bm - r.mean[j]);
    }
    r.seMean[j] = sqrt(bs2 / (blocks * (blocks - 1)));
  }
  return r;
}

double Energy(const FFAngles &ang, const Branch &br, const double *th,
              const double *ph) {
  double e = 0.0;
  for (uint b = 0; b < br.n; ++b) {
    e += ang.Calc(br.kinds[b][b], th[b]);
    for (uint c = 0; c < b; ++c)
      e += ang.Calc(br.kinds[b][c],
                    BranchSampler::AngleBetween(th[b], ph[b], th[c], ph[c]));
  }
  return e;
}

// Reference: Metropolis random walk of the bond directions on the sphere
std::vector<std::vector<double>> Reference(const FFAngles &ang,
                                           const Branch &br, double beta,
                                           size_t nSamples, unsigned seed) {
  Rng rng(seed);
  std::normal_distribution<double> gauss(0.0, 1.0);
  double th[MAX_BONDS], ph[MAX_BONDS];
  for (uint b = 0; b < br.n; ++b) {
    th[b] = ang.Angle(br.kinds[b][b]);
    ph[b] = 2.0 * M_PI * b / br.n;
  }
  double e = Energy(ang, br, th, ph);
  const double step = 0.08;
  std::vector<std::vector<double>> out;
  const size_t sweepsPerSample = 5, equil = 20000;
  for (size_t s = 0; s < equil + nSamples * sweepsPerSample; ++s) {
    for (uint b = 0; b < br.n; ++b) {
      // perturb the unit vector isotropically and renormalize (symmetric)
      double x = sin(th[b]) * cos(ph[b]) + step * gauss(rng.gen);
      double y = sin(th[b]) * sin(ph[b]) + step * gauss(rng.gen);
      double z = cos(th[b]) + step * gauss(rng.gen);
      double r = sqrt(x * x + y * y + z * z);
      double oldT = th[b], oldP = ph[b];
      th[b] = acos(std::max(-1.0, std::min(1.0, z / r)));
      ph[b] = atan2(y, x);
      double eNew = Energy(ang, br, th, ph);
      if (rng() < exp(-beta * (eNew - e))) {
        e = eNew;
      } else {
        th[b] = oldT;
        ph[b] = oldP;
      }
    }
    if (s >= equil && (s - equil) % sweepsPerSample == 0)
      out.push_back(Angles(br, th, ph));
  }
  return out;
}

// Chain driven by the sampler exactly as DCHedron uses it
std::vector<std::vector<double>> SamplerChain(const FFAngles &ang,
                                              const Branch &br, double beta,
                                              uint nTrials, size_t nSamples,
                                              unsigned seed,
                                              double &acceptance) {
  Rng rng(seed);
  BranchSampler sampler;
  sampler.Init(br.n, br.kinds, &ang, nullptr);
  double th[MAX_BONDS], ph[MAX_BONDS], tt[MAX_BONDS], tp[MAX_BONDS];
  double nt[MAX_BONDS], np[MAX_BONDS];
  EXPECT_TRUE(sampler.Sample(rng, beta, th, ph));
  std::vector<std::vector<double>> out;
  size_t accepted = 0;
  const size_t equil = 1000;
  for (size_t s = 0; s < equil + nSamples; ++s) {
    double wNew = 0.0;
    for (uint i = 0; i < nTrials; ++i) {
      EXPECT_TRUE(sampler.Sample(rng, beta, tt, tp));
      double w = sampler.Weight(beta, tt, tp);
      wNew += w;
      if (i == 0 || rng() * wNew < w) {
        std::copy(tt, tt + br.n, nt);
        std::copy(tp, tp + br.n, np);
      }
    }
    double wOld = sampler.Weight(beta, th, ph);
    for (uint i = 1; i < nTrials; ++i) {
      sampler.Sample(rng, beta, tt, tp);
      wOld += sampler.Weight(beta, tt, tp);
    }
    if (rng() * wOld < wNew) {
      std::copy(nt, nt + br.n, th);
      std::copy(np, np + br.n, ph);
      if (s >= equil)
        ++accepted;
    }
    if (s >= equil)
      out.push_back(Angles(br, th, ph));
  }
  acceptance = double(accepted) / nSamples;
  return out;
}

template <class AngleType = FFAngles>
void CheckBranch(const char *name, const std::vector<double> &k,
                 const std::vector<double> &t0, const Branch &br,
                 uint nTrials = 4, size_t nSamples = 60000) {
  TestAngles<AngleType> ang;
  ang.Set(k, t0, std::vector<bool>(k.size(), false));
  const double beta = 1.0; // energies in units of kT
  double acc = 0.0;
  Moments ref = Summarize(Reference(ang, br, beta, nSamples, 11));
  Moments got =
      Summarize(SamplerChain(ang, br, beta, nTrials, nSamples, 23, acc));
  std::cout << "[ " << name << " ] N = " << nTrials
            << ", chain acceptance = " << acc << "\n";
  for (size_t j = 0; j < ref.mean.size(); ++j) {
    double se = sqrt(ref.seMean[j] * ref.seMean[j] +
                     got.seMean[j] * got.seMean[j]);
    EXPECT_NEAR(got.mean[j], ref.mean[j], 5.0 * se + 0.05)
        << name << ": mean of angle " << j;
    EXPECT_NEAR(got.sd[j], ref.sd[j], 0.03 * ref.sd[j] + 5.0 * se + 0.05)
        << name << ": width of angle " << j;
  }
}

// Fill a branch whose angle kinds are given row by row (lower triangle)
Branch MakeBranch(uint n, const std::vector<std::vector<uint>> &rows) {
  Branch br;
  br.n = n;
  for (uint b = 0; b < n; ++b)
    for (uint c = 0; c <= b; ++c) {
      br.kinds[b][c] = rows[b][c];
      br.kinds[c][b] = rows[b][c];
    }
  return br;
}

} // namespace

// sp3 center with three branches (e.g. CH3 or CH2 grown from a carbon)
TEST(BranchSamplerTest, Tetrahedral3) {
  Branch br = MakeBranch(3, {{0}, {1, 0}, {1, 1, 0}});
  CheckBranch("tetrahedral, 3 branches", {80.0, 50.0}, {109.5, 108.0}, br, 1);
  CheckBranch("tetrahedral, 3 branches", {80.0, 50.0}, {109.5, 108.0}, br, 4);
}

// sp3 center with two branches (e.g. an ether oxygen's neighbor)
TEST(BranchSamplerTest, Tetrahedral2) {
  Branch br = MakeBranch(2, {{0}, {1, 0}});
  CheckBranch("tetrahedral, 2 branches", {60.0, 40.0}, {110.0, 112.0}, br);
}

// Asymmetric angles and stiffnesses
TEST(BranchSamplerTest, Asymmetric) {
  Branch br = MakeBranch(3, {{0}, {3, 1}, {4, 5, 2}});
  CheckBranch("asymmetric", {70.0, 30.0, 90.0, 45.0, 60.0, 20.0},
              {100.0, 115.0, 125.0, 104.0, 112.0, 118.0}, br);
}

// sp2 (trigonal planar) center: the twist is planar, so phi is uniform
// with rejection; also the in-plane (120, 120, 120) case
TEST(BranchSamplerTest, TrigonalPlanar) {
  Branch br = MakeBranch(2, {{0}, {0, 0}});
  CheckBranch("trigonal planar", {40.0}, {120.0}, br);
}

// Linear center, one branch (e.g. CO2, nitrile)
TEST(BranchSamplerTest, Linear) {
  Branch br = MakeBranch(1, {{0}});
  CheckBranch("linear", {40.0}, {180.0}, br);
}

// Trigonal bipyramid with prev axial: one axial and three equatorial
TEST(BranchSamplerTest, TrigonalBipyramid) {
  // kinds: 0 axial-axial 180, 1 axial-equatorial 90, 2 equatorial pair 120
  // rows list c = 0..b; the last entry is the prev-focus-b (diagonal) kind
  Branch br = MakeBranch(4, {{0}, {1, 1}, {1, 2, 1}, {1, 2, 2, 1}});
  CheckBranch("trigonal bipyramid", {30.0, 30.0, 30.0}, {180.0, 90.0, 120.0},
              br, 4, 40000);
}

// Octahedron with prev axial: one axial and four equatorial atoms
TEST(BranchSamplerTest, Octahedral) {
  // kinds: 0 trans 180, 1 cis 90
  // equatorial atoms 1-4 in order around the ring: 1-3 and 2-4 are trans
  Branch br = MakeBranch(5, {{0}, {1, 1}, {1, 1, 1}, {1, 0, 1, 1},
                             {1, 1, 0, 1, 1}});
  CheckBranch("octahedral", {30.0, 30.0}, {180.0, 90.0}, br, 4, 40000);
}

// Very soft angles: Gaussian proposal too wide, uniform phi is used
TEST(BranchSamplerTest, Soft) {
  Branch br = MakeBranch(3, {{0}, {1, 0}, {1, 1, 0}});
  CheckBranch("soft", {2.0, 1.0}, {109.5, 109.5}, br);
}

// Martini cosine-harmonic bending: Ktheta (cos(theta) - cos(theta0))^2
TEST(BranchSamplerTest, Martini) {
  Branch br = MakeBranch(3, {{0}, {1, 0}, {1, 1, 0}});
  CheckBranch<FFAngleMartini>("martini", {60.0, 40.0}, {109.5, 108.0}, br);
}

// One fixed bond angle: the constrained twist is solved for and held
TEST(BranchSamplerTest, FixedPairAngle) {
  TestAngles<FFAngles> ang;
  ang.Set({60.0, 40.0, 40.0}, {109.5, 108.0, 108.0}, {false, true, false});
  Branch br = MakeBranch(3, {{0}, {1, 0}, {2, 2, 0}});
  BranchSampler sampler;
  sampler.Init(br.n, br.kinds, &ang, nullptr);
  Rng rng(5);
  double th[MAX_BONDS], ph[MAX_BONDS];
  for (int i = 0; i < 10000; ++i) {
    ASSERT_TRUE(sampler.Sample(rng, 1.0, th, ph));
    EXPECT_NEAR(BranchSampler::AngleBetween(th[1], ph[1], th[0], ph[0]),
                108.0 * M_PI / 180.0, 1e-9);
    double w = sampler.Weight(1.0, th, ph);
    EXPECT_TRUE(std::isfinite(w) && w > 0.0);
  }
}

// All angles fixed (rigid branch): always the same geometry, weight 1.
// Uses the exact tetrahedral angle so all three pair constraints can hold.
TEST(BranchSamplerTest, AllFixed) {
  const double tet = acos(-1.0 / 3.0) * 180.0 / M_PI;
  TestAngles<FFAngles> ang;
  ang.Set({1.0, 1.0}, {tet, tet}, {true, true});
  Branch br = MakeBranch(3, {{0}, {1, 0}, {1, 1, 0}});
  BranchSampler sampler;
  sampler.Init(br.n, br.kinds, &ang, nullptr);
  Rng rng(9);
  double th[MAX_BONDS], ph[MAX_BONDS];
  for (int i = 0; i < 100; ++i) {
    ASSERT_TRUE(sampler.Sample(rng, 1.0, th, ph));
    for (uint b = 1; b < 3; ++b)
      for (uint c = 0; c < b; ++c)
        EXPECT_NEAR(BranchSampler::AngleBetween(th[b], ph[b], th[c], ph[c]),
                    tet * M_PI / 180.0, 1e-6);
    EXPECT_DOUBLE_EQ(sampler.Weight(1.0, th, ph), 1.0);
  }
}
